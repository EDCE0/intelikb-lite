# 文件名：src/embeddings.py
"""
本地向量化模型（Embedding）

DeepSeek 只提供对话 / 推理接口，**没有 embedding 接口**，
所以原来用的 OpenAIEmbedding 必须换掉。这里改用 sentence-transformers
加载本地中文向量模型（默认 BAAI/bge-small-zh-v1.5，也可换 BAAI/bge-m3），
好处是：完全离线、中文效果好、不需要额外的向量 API Key。

模型加载结果做进程内缓存，Streamlit 反复 rerun 时不会重复加载。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import Field, PrivateAttr

from .config import (
    EMBED_BATCH_SIZE,
    EMBED_DEVICE,
    EMBED_MODEL,
    EMBED_QUERY_INSTRUCTION,
)

# (model_name, device) -> SentenceTransformer 实例
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def load_sentence_transformer(model_name: str, device: str = "cpu"):
    """加载并缓存 sentence-transformers 模型

    优先按「只用本地缓存」加载：模型已下载时启动更快，也不依赖外网；
    缓存里没有时再联网下载。
    """
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(
                model_name, device=device, local_files_only=True
            )
        except Exception:
            print(f"本地未找到模型 {model_name}，尝试联网下载（首次较慢）...")
            model = SentenceTransformer(model_name, device=device)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """释放缓存（切换模型 / 测试时用）"""
    _MODEL_CACHE.clear()


class BGEEmbedding(BaseEmbedding):
    """基于 sentence-transformers 的 BGE 中文向量模型"""

    model_name: str = Field(default=EMBED_MODEL, description="向量模型名称或本地路径")
    device: str = Field(default=EMBED_DEVICE, description="运行设备：cpu / cuda")
    normalize: bool = Field(default=True, description="是否 L2 归一化（配合余弦相似度）")
    query_instruction: str = Field(
        default=EMBED_QUERY_INSTRUCTION,
        description="BGE 检索时给 query 加的指令前缀",
    )

    _model: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        data.setdefault("embed_batch_size", EMBED_BATCH_SIZE)
        super().__init__(**data)

    @property
    def st_model(self):
        """惰性加载底层的 SentenceTransformer"""
        if self._model is None:
            self._model = load_sentence_transformer(self.model_name, self.device)
        return self._model

    # ------------------------------------------------------------------
    # 内部编码
    # ------------------------------------------------------------------
    def _encode(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        vectors = self.st_model.encode(
            texts,
            batch_size=self.embed_batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    # ------------------------------------------------------------------
    # 需要实现的抽象方法
    # ------------------------------------------------------------------
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._encode([query], is_query=True)[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._encode([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 真正批量编码，建索引时速度快很多
        return self._encode(list(texts))

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


def build_embedding(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> BGEEmbedding:
    """按需构造向量模型"""
    params: Dict[str, Any] = {}
    if model_name:
        params["model_name"] = model_name
    if device:
        params["device"] = device
    params.update(kwargs)
    return BGEEmbedding(**params)
