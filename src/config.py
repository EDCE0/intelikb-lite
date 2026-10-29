# 文件名：src/config.py
"""
InteliKB-Lite 全局配置

所有密钥 / 模型 / 路径都集中在这里，统一从 .env 读取：
  * 项目根目录 .env          （intelikb_lite/.env）
  * 上级目录 .env            （RAG/.env，默认密钥就放这里）
  * 环境变量 INTELIKB_ENV_FILE 指定的文件

环境变量优先级：真实环境变量 > .env 文件 > 代码里的默认值
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 项目根目录： intelikb_lite/
PROJECT_DIR = Path(__file__).resolve().parent.parent
# 上级目录： RAG/ （.env 默认放在这里）
PARENT_DIR = PROJECT_DIR.parent

# 使用说明里的默认知识库目录
DEFAULT_DATA_DIR = "data/raw"
# 默认向量库目录
DEFAULT_CHROMA_DIR = "./chroma_db"
# 默认向量集合名
DEFAULT_COLLECTION = "intelikb"

# 默认模型：
#   DeepSeek 对话模型（OpenAI 兼容接口），可选 deepseek-chat / deepseek-reasoner
DEFAULT_LLM_MODEL = "deepseek-chat"
#   DeepSeek 官方接口地址（也可以写 https://api.deepseek.com/v1）
DEFAULT_API_BASE = "https://api.deepseek.com"
#   DeepSeek 没有 embedding 接口，向量化改用本地 BGE 中文模型
DEFAULT_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


def _find_env_file() -> Optional[Path]:
    """按优先级查找 .env 文件"""
    candidates = []
    explicit = os.getenv("INTELIKB_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates += [
        PROJECT_DIR / ".env",
        PARENT_DIR / ".env",
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


# 载入 .env（不覆盖已存在的真实环境变量）
ENV_FILE = _find_env_file()
if ENV_FILE is not None:
    load_dotenv(ENV_FILE, override=False, encoding="utf-8")


def _get(key: str, default: str = "") -> str:
    """读取字符串配置，自动去掉首尾空白"""
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip()


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


def _resolve_path(raw: str, base: Path = PROJECT_DIR) -> str:
    """相对路径统一相对项目根目录解析，避免在不同工作目录下跑出两套数据"""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path)


def _fix_legacy_chroma_key() -> None:
    """
    兼容历史 .env 里的错误写法：
        CHROMA persist_directory=./chroma_db
    这是「变量名里带空格」的无效行，python-dotenv 会报警告并丢弃。
    这里把它当作 CHROMA_PERSIST_DIR 的同义写法处理。
    """
    for key, value in list(os.environ.items()):
        if key.strip() == "CHROMA persist_directory" or key.strip().startswith(
            "CHROMA persist_directory"
        ):
            os.environ.setdefault("CHROMA_PERSIST_DIR", value.strip())


_fix_legacy_chroma_key()

# ---------------------------------------------------------------------------
# DeepSeek 配置
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY: str = _get("DEEPSEEK_API_KEY") or _get("DS_API_KEY")
DEEPSEEK_API_BASE: str = _get("DEEPSEEK_API_BASE", DEFAULT_API_BASE).rstrip("/")
DEEPSEEK_MODEL: str = _get("DEEPSEEK_MODEL", DEFAULT_LLM_MODEL)
DEEPSEEK_TEMPERATURE: float = _get_float("DEEPSEEK_TEMPERATURE", 0.3)
DEEPSEEK_MAX_TOKENS: int = _get_int("DEEPSEEK_MAX_TOKENS", 2048)
DEEPSEEK_TIMEOUT: float = _get_float("DEEPSEEK_TIMEOUT", 120.0)
DEEPSEEK_MAX_RETRIES: int = _get_int("DEEPSEEK_MAX_RETRIES", 2)
# deepseek-chat 上下文 64K（reasoner 也是 64K），按需可在 .env 调整
DEEPSEEK_CONTEXT_WINDOW: int = _get_int("DEEPSEEK_CONTEXT_WINDOW", 65536)

# ---------------------------------------------------------------------------
# 向量化（Embedding）配置：DeepSeek 未提供 embedding 接口，改用本地模型
# ---------------------------------------------------------------------------
EMBED_MODEL: str = _get("EMBED_MODEL", DEFAULT_EMBED_MODEL)
EMBED_DEVICE: str = _get("EMBED_DEVICE", "cpu")
EMBED_BATCH_SIZE: int = _get_int("EMBED_BATCH_SIZE", 16)
# BGE 中文模型检索时的推荐指令前缀（bge-m3 可留空）
EMBED_QUERY_INSTRUCTION: str = _get(
    "EMBED_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章："
)

# ---------------------------------------------------------------------------
# 回答模式
#   strict  : 只用知识库资料；资料没有就回答「资料中未提及」
#   hybrid  : 先依据资料，资料不足时用 DeepSeek 自身知识补充，并明确标注
#   general : 完全用 DeepSeek 通用知识回答（不做检索）
# ---------------------------------------------------------------------------
VALID_ANSWER_MODES = ("strict", "hybrid", "general")
ANSWER_MODE: str = _get("ANSWER_MODE", "hybrid")

# ---------------------------------------------------------------------------
# 检索 / 分块配置
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = _resolve_path(
    _get("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR) or DEFAULT_CHROMA_DIR
)
COLLECTION_NAME: str = _get("COLLECTION_NAME", DEFAULT_COLLECTION)
DATA_DIR: str = _resolve_path(_get("DATA_DIR", DEFAULT_DATA_DIR))
SIMILARITY_TOP_K: int = _get_int("SIMILARITY_TOP_K", 5)
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 100)


def normalize_answer_mode(mode: Optional[str] = None) -> str:
    """把回答模式规范化为 strict / hybrid / general"""
    value = (mode or ANSWER_MODE or "hybrid").strip().lower()
    return value if value in VALID_ANSWER_MODES else "hybrid"


def require_api_key(api_key: Optional[str] = None) -> str:
    """
    获取 DeepSeek API Key，缺失时给出可操作的报错信息
    """
    key = (api_key or DEEPSEEK_API_KEY or "").strip()
    if not key:
        env_hint = str(ENV_FILE) if ENV_FILE else str(PARENT_DIR / ".env")
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。\n"
            f"请在 {env_hint} 中写入：\n"
            "    DEEPSEEK_API_KEY=sk-你的密钥\n"
            "或设置同名环境变量后再运行。"
        )
    return key


def describe() -> str:
    """返回一份便于排查的配置概览（密钥打码）"""
    key = DEEPSEEK_API_KEY
    masked = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else ("未配置" if not key else "***")
    lines = [
        "InteliKB-Lite 配置",
        f"  .env 文件        : {ENV_FILE or '未找到'}",
        f"  DEEPSEEK_API_KEY : {masked}",
        f"  DEEPSEEK_API_BASE: {DEEPSEEK_API_BASE}",
        f"  DEEPSEEK_MODEL   : {DEEPSEEK_MODEL}",
        f"  知识库目录       : {DATA_DIR}",
        f"  向量库目录       : {CHROMA_PERSIST_DIR}",
        f"  向量集合         : {COLLECTION_NAME}",
        f"  Embedding 模型   : {EMBED_MODEL} ({EMBED_DEVICE}, batch={EMBED_BATCH_SIZE})",
        f"  回答模式         : {ANSWER_MODE}（strict / hybrid / general）",
        f"  检索 top_k       : {SIMILARITY_TOP_K}",
        f"  分块 chunk_size  : {CHUNK_SIZE} / overlap {CHUNK_OVERLAP}",
    ]
    return "\n".join(lines)
