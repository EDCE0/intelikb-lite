# 文件名：src/rag_engine.py
"""
InteliKB-Lite RAG 引擎

流程：data/raw 下的 PDF / TXT / Markdown / DOCX
      -> 解析为 Document -> 清洗 -> 分块
      -> 本地 BGE 向量化 -> 存入 Chroma
      -> 检索 top_k 片段 -> 交给 DeepSeek 生成答案

说明：
  * 大模型：DeepSeek（OpenAI 兼容接口，见 src/deepseek_llm.py）
  * 向量化：本地 BGE 中文模型（DeepSeek 无 embedding 接口，见 src/embeddings.py）
  * 向量库：Chroma（持久化目录见 .env 的 CHROMA_PERSIST_DIR）
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import (
    ANSWER_MODE,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    SIMILARITY_TOP_K,
    normalize_answer_mode,
)
from .deepseek_llm import DeepSeekLLM, build_llm
from .embeddings import BGEEmbedding, build_embedding

# 支持的文件后缀
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".docx"}

# =====================================================================
# 提示词模板
#   QA_TEMPLATE / REFINE_TEMPLATE              : 严格模式（只用资料）
#   QA_TEMPLATE_HYBRID / REFINE_TEMPLATE_HYBRID: 混合模式（资料优先 + 模型补充）
#   GENERAL_SYSTEM_PROMPT                      : 通用模式（纯模型知识）
# =====================================================================

# 严格模式：只依据资料回答，并按「文件名 第X页」标注来源
QA_TEMPLATE = PromptTemplate(
    "以下是检索到的参考资料：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请只依据以上参考资料回答问题，不要使用资料之外的知识，也不要编造。\n"
    "如果资料中没有相关信息，请直接说明「资料中未提及」，并指出还缺少哪方面的信息。\n"
    "回答请使用中文，条理清晰；引用具体依据时，直接沿用资料开头标注的来源，"
    "写成【文件名 第X页】的形式。\n"
    "问题：{query_str}\n"
    "回答："
)

# 多片段合并时使用的提示词
REFINE_TEMPLATE = PromptTemplate(
    "已有一个初步答案：\n"
    "---------------------\n"
    "{existing_answer}\n"
    "---------------------\n"
    "下面还有新的参考资料：\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "请判断新资料是否能补充或修正已有答案。\n"
    "如果可以，请整合成更完整、更准确的答案（仍然只依据资料，保持中文，"
    "并保留【文件名 第X页】形式的引用）；"
    "如果新资料没有帮助，请原样返回已有答案。\n"
    "问题：{query_str}\n"
    "答案："
)

# 混合模式：资料优先；资料不足时用模型自身知识补充，并单独标注
QA_TEMPLATE_HYBRID = PromptTemplate(
    "以下是检索到的参考资料：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请按下面的要求回答（用中文，条理清晰）：\n"
    "1. 先判断资料是否足以回答问题：\n"
    "   - 资料能回答：只依据资料作答，并在依据处标注【文件名 第X页】（沿用资料开头标注的来源）；\n"
    "   - 资料只能回答一部分：先说明资料覆盖了什么并标注来源，再指出资料缺少什么。\n"
    "2. 当资料不足以完整回答时，另起一节，标题固定为「模型补充（非知识库内容）」，"
    "用你自己的通用知识继续回答，并说明这部分内容未经知识库验证、仅供扩展参考。\n"
    "3. 严禁把通用知识伪装成资料内容：没有资料依据时不要标注【文件名 第X页】。\n"
    "问题：{query_str}\n"
    "回答："
)

# 混合模式下多片段合并
REFINE_TEMPLATE_HYBRID = PromptTemplate(
    "已有一个初步答案：\n"
    "---------------------\n"
    "{existing_answer}\n"
    "---------------------\n"
    "下面还有新的参考资料：\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "请整合成更完整、更准确的答案（中文，条理清晰），规则：\n"
    "1. 资料能支持的内容标注【文件名 第X页】；\n"
    "2. 若资料仍不足以完整回答，保留或补充「模型补充（非知识库内容）」一节；\n"
    "3. 不要把通用知识当成资料结论，也不要编造资料中不存在的内容。\n"
    "问题：{query_str}\n"
    "答案："
)

# 通用模式：完全不看知识库，让 DeepSeek 用自己的知识回答
GENERAL_SYSTEM_PROMPT = (
    "你是 DeepSeek，林业、遥感、生态等领域的知识助手。\n"
    "用户的问题没有（或不需要）知识库资料支撑，请直接基于你自己的通用知识回答，"
    "使用中文，条理清晰。\n"
    "如果涉及具体文献、数据、课程设置等细节，请说明这属于通用知识、未经本地知识库验证。"
)


def clean_text(text: str) -> str:
    """对 PDF 抽取出来的文本做轻量清洗，减少噪声、保留语义"""
    if not text:
        return ""
    # 全角空格、不间断空格
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    # 去掉控制字符
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    # 行内多余空格
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    # 合并被硬换行打断的中文句子（上一行没有句末标点，下一行以中文开头）
    merged: List[str] = []
    for line in lines:
        if not line:
            merged.append("")
            continue
        if (
            merged
            and merged[-1]
            and not re.search(r"[。！？；：.!?;:）)】」]$", merged[-1])
            and re.match(r"[\u4e00-\u9fff（(《“]", line)
        ):
            merged[-1] += line
        else:
            merged.append(line)
    text = "\n".join(merged)
    # 连续空行压缩
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class RAGEngine:
    """RAG 引擎（DeepSeek + 本地 BGE + Chroma）"""

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embed_model: Optional[BGEEmbedding] = None,
        llm: Optional[DeepSeekLLM] = None,
        similarity_top_k: int = SIMILARITY_TOP_K,
        answer_mode: Optional[str] = None,
    ) -> None:
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        # 默认回答模式：strict / hybrid / general（见 .env 的 ANSWER_MODE）
        self.answer_mode = normalize_answer_mode(answer_mode)
        # 惰性初始化，避免只建索引时也去校验 API Key
        self._embed_model = embed_model
        self._llm = llm
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
        # 每种回答模式一个查询引擎（提示词不同）
        self._query_engines: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 组件
    # ------------------------------------------------------------------
    @property
    def embed_model(self) -> BGEEmbedding:
        if self._embed_model is None:
            self._embed_model = build_embedding()
        return self._embed_model

    @property
    def llm(self) -> DeepSeekLLM:
        if self._llm is None:
            self._llm = build_llm()
        return self._llm

    # ------------------------------------------------------------------
    # 文档加载
    # ------------------------------------------------------------------
    @staticmethod
    def _load_pdf(path: Path) -> List[Document]:
        """逐页解析 PDF，保留页码，方便答案溯源"""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        documents: List[Document] = []
        for page_no, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = clean_text(text)
            # 跳过扫描页 / 空白页（少于 20 个字符）
            if len(text) < 20:
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": path.name,
                        "file_path": str(path),
                        "file_type": "pdf",
                        "page_number": page_no,
                    },
                )
            )
        return documents

    @staticmethod
    def _load_text_file(path: Path) -> List[Document]:
        raw = ""
        for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                raw = path.read_text(encoding=encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        text = clean_text(raw)
        if not text:
            return []
        return [
            Document(
                text=text,
                metadata={
                    "file_name": path.name,
                    "file_path": str(path),
                    "file_type": path.suffix.lstrip("."),
                    "page_number": 1,
                },
            )
        ]

    @staticmethod
    def _load_docx(path: Path) -> List[Document]:
        import docx2txt

        text = clean_text(docx2txt.process(str(path)) or "")
        if not text:
            return []
        return [
            Document(
                text=text,
                metadata={
                    "file_name": path.name,
                    "file_path": str(path),
                    "file_type": "docx",
                    "page_number": 1,
                },
            )
        ]

    def load_documents(
        self,
        data_path: str = DATA_DIR,
        recursive: bool = True,
        limit: Optional[int] = None,
    ) -> List[Document]:
        """
        加载知识库文档

        Args:
            data_path: 知识库目录（默认 data/raw）
            recursive: 是否递归子目录
            limit: 只读取前 N 个文件（调试用，None 表示全部）

        Returns:
            Document 列表；目录为空时返回空列表
        """
        root = Path(data_path).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"知识库目录不存在：{root}")
        if root.is_file():
            files = [root]
        else:
            pattern = "**/*" if recursive else "*"
            files = sorted(
                p
                for p in root.glob(pattern)
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            )
        if limit:
            files = files[:limit]

        documents: List[Document] = []
        skipped: List[str] = []
        for path in files:
            suffix = path.suffix.lower()
            try:
                if suffix == ".pdf":
                    docs = self._load_pdf(path)
                elif suffix == ".docx":
                    docs = self._load_docx(path)
                else:
                    docs = self._load_text_file(path)
            except Exception as exc:  # 单个坏文件不影响整体建库
                print(f"⚠️  跳过 {path.name}：{exc}")
                skipped.append(path.name)
                continue
            if not docs:
                skipped.append(path.name)
            documents.extend(docs)

        self.last_skipped = skipped
        return documents

    # ------------------------------------------------------------------
    # 建索引
    # ------------------------------------------------------------------
    def _open_collection(self, rebuild: bool = False):
        chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        if rebuild:
            try:
                chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass  # 集合不存在时忽略
        # 归一化向量 + 余弦相似度
        return chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(
        self,
        documents: List[Document],
        rebuild: bool = True,
        show_progress: bool = False,
    ) -> int:
        """
        构建向量索引

        Args:
            documents: 文档列表
            rebuild: True 时先清空旧集合（避免重复入库）
            show_progress: 是否显示向量化进度条

        Returns:
            入向量库的 chunk 数量
        """
        if not documents:
            raise ValueError("没有可用的文档，请检查知识库目录")

        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes: List[BaseNode] = splitter.get_nodes_from_documents(documents)
        if not nodes:
            raise ValueError("文档分块结果为空，请检查文档内容")
        self._inject_source_header(nodes)

        collection = self._open_collection(rebuild=rebuild)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=show_progress,
        )
        # 查询引擎延迟到第一次提问时再创建：这样没有 API Key 也能先建索引
        self.query_engine = None
        return len(nodes)

    @staticmethod
    def _inject_source_header(nodes: List[BaseNode]) -> None:
        """给每个 chunk 开头加上【来源：文件名 第X页】

        这样模型能在答案里准确标注依据，也能增强「标题/主题」类问题的检索效果。
        """
        for node in nodes:
            metadata = node.metadata or {}
            file_name = metadata.get("file_name")
            if not file_name:
                continue
            page = metadata.get("page_number")
            label = (
                f"【来源：{file_name} 第{page}页】"
                if page
                else f"【来源：{file_name}】"
            )
            content = node.get_content()
            if not content.startswith("【来源："):
                node.text = f"{label}\n{content}"

    def _get_query_engine(self, mode: str):
        """按回答模式获取（并缓存）查询引擎"""
        if self.index is None:
            raise ValueError("索引未初始化，请先调用 build_index() 或 load_index()")
        if mode == "general":
            raise ValueError("通用模式不需要检索，请直接调用 query(..., mode='general')")

        if mode not in self._query_engines:
            if mode == "hybrid":
                qa_template, refine_template = QA_TEMPLATE_HYBRID, REFINE_TEMPLATE_HYBRID
            else:  # strict
                qa_template, refine_template = QA_TEMPLATE, REFINE_TEMPLATE
            self._query_engines[mode] = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=self.similarity_top_k,
                response_mode="compact",
                text_qa_template=qa_template,
                refine_template=refine_template,
            )
        self.query_engine = self._query_engines[mode]
        return self.query_engine

    def reset_query_engines(self) -> None:
        """清空查询引擎缓存（切换模型 / top_k / 回答模式后调用）"""
        self._query_engines.clear()
        self.query_engine = None

    def _query_general(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """通用模式：不检索，完全用 DeepSeek 自身知识回答"""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=GENERAL_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=question),
        ]
        response = self.llm.chat(messages)
        return (response.message.content or "").strip(), []

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def query(
        self, question: str, mode: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        查询

        Args:
            question: 用户问题
            mode: 回答模式
                - "strict" : 只依据知识库资料
                - "hybrid" : 资料优先，资料不足时用 DeepSeek 通用知识补充并标注
                - "general": 完全用 DeepSeek 通用知识回答（不检索）
                不传则使用 self.answer_mode（默认来自 .env 的 ANSWER_MODE）

        Returns:
            (答案, 来源片段列表)；general 模式来源为空
        """
        mode = normalize_answer_mode(mode or self.answer_mode)

        if mode == "general":
            return self._query_general(question)

        if self.index is None:
            raise ValueError("引擎未初始化，请先调用 build_index() 或 load_index()")

        response = self._get_query_engine(mode).query(question)

        sources: List[Dict[str, Any]] = []
        for node in getattr(response, "source_nodes", []) or []:
            metadata = node.node.metadata or {}
            sources.append(
                {
                    # 保留完整片段，评估时需要完整上下文；界面显示时再截断
                    "content": node.node.get_content(),
                    "file_name": metadata.get("file_name", "未知来源"),
                    "page_number": metadata.get("page_number"),
                    "score": float(node.score) if node.score is not None else 0.0,
                    "metadata": metadata,
                }
            )

        return str(response).strip(), sources

    # ------------------------------------------------------------------
    # 增量 / 持久化
    # ------------------------------------------------------------------
    def add_documents(self, new_documents: List[Document]) -> int:
        """向已有索引追加文档"""
        if not new_documents:
            return 0
        if self.index is None:
            return self.build_index(new_documents, rebuild=False)

        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(new_documents)
        self._inject_source_header(nodes)
        self.index.insert_nodes(nodes)
        return len(nodes)

    def save_index(self) -> None:
        """保存索引（Chroma 会自动持久化，这里兼容旧调用）"""
        if self.index is not None and self.index.storage_context is not None:
            try:
                self.index.storage_context.persist(persist_dir=self.persist_dir)
            except Exception:
                # Chroma 持久化失败不影响已写入的数据
                pass

    def count_nodes(self) -> int:
        """当前集合中的 chunk 数量"""
        try:
            chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            collection = chroma_client.get_collection(self.collection_name)
            return int(collection.count())
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # 加载已有索引
    # ------------------------------------------------------------------
    @classmethod
    def load_index(
        cls,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embed_model: Optional[BGEEmbedding] = None,
        llm: Optional[DeepSeekLLM] = None,
        similarity_top_k: int = SIMILARITY_TOP_K,
        answer_mode: Optional[str] = None,
    ) -> "RAGEngine":
        """加载 Chroma 中已建好的索引"""
        engine = cls(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embed_model=embed_model,
            llm=llm,
            similarity_top_k=similarity_top_k,
            answer_mode=answer_mode,
        )

        chroma_client = chromadb.PersistentClient(path=engine.persist_dir)
        try:
            collection = chroma_client.get_collection(engine.collection_name)
        except Exception as exc:
            raise FileNotFoundError(
                f"向量库中不存在集合「{engine.collection_name}」"
                f"（目录：{engine.persist_dir}），请先构建索引。"
            ) from exc

        if collection.count() == 0:
            raise ValueError("向量库为空，请先构建索引。")

        vector_store = ChromaVectorStore(chroma_collection=collection)
        engine.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=engine.embed_model,
        )
        return engine

    @classmethod
    def index_exists(
        cls,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> bool:
        """判断向量库里是否已经有可用索引"""
        try:
            chroma_client = chromadb.PersistentClient(path=str(persist_dir))
            collection = chroma_client.get_collection(collection_name)
            return collection.count() > 0
        except Exception:
            return False
