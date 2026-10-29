# 文件名：app.py
"""
InteliKB-Lite Streamlit 应用

运行：
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

from src.config import (
    ANSWER_MODE,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DATA_DIR,
    EMBED_MODEL,
    ENV_FILE,
    SIMILARITY_TOP_K,
)
from src.deepseek_llm import build_llm
from src.evaluator import RAGEvaluator
from src.rag_engine import SUPPORTED_SUFFIXES, RAGEngine

# 回答模式：界面文案 -> 内部取值
ANSWER_MODE_LABELS = {
    "混合（资料优先 + 模型补充）": "hybrid",
    "严格（仅知识库资料）": "strict",
    "通用（纯模型知识）": "general",
}
ANSWER_MODE_NAMES = {v: k for k, v in ANSWER_MODE_LABELS.items()}


# 页面配置
st.set_page_config(
    page_title="InteliKB-Lite",
    page_icon="🌲",
    layout="wide",
)

# 自定义CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# session state
# ----------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("engine", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("eval_result", None)
    st.session_state.setdefault("collection", COLLECTION_NAME)
    st.session_state.setdefault("persist_dir", CHROMA_PERSIST_DIR)
    st.session_state.setdefault("data_dir", DATA_DIR)
    st.session_state.setdefault("top_k", SIMILARITY_TOP_K)
    st.session_state.setdefault("llm_model", DEEPSEEK_MODEL)
    st.session_state.setdefault("answer_mode", ANSWER_MODE)
    st.session_state.setdefault("load_error", "")


def sync_engine(engine: RAGEngine) -> RAGEngine:
    """把侧边栏配置同步到引擎实例"""
    engine.persist_dir = st.session_state.persist_dir
    engine.collection_name = st.session_state.collection
    engine.answer_mode = st.session_state.answer_mode
    # top_k 变化后需要重建查询引擎才生效
    if engine.similarity_top_k != st.session_state.top_k:
        engine.similarity_top_k = st.session_state.top_k
        engine.reset_query_engines()
    return engine


def ensure_llm(engine: RAGEngine) -> None:
    """模型下拉框切换后重建 DeepSeek 客户端"""
    model = st.session_state.llm_model
    llm = engine._llm
    if llm is None or llm.model_name != model:
        engine._llm = build_llm(model=model)
        # 让查询引擎在下次提问时用新模型重建
        engine.reset_query_engines()


def get_engine() -> RAGEngine:
    """按页面配置构造引擎（同一会话复用）"""
    engine = st.session_state.engine
    if engine is None:
        engine = RAGEngine(
            persist_dir=st.session_state.persist_dir,
            collection_name=st.session_state.collection,
            similarity_top_k=st.session_state.top_k,
        )
        st.session_state.engine = engine
    return sync_engine(engine)


def load_engine_or_none() -> RAGEngine | None:
    """优先复用内存里的引擎，否则从 Chroma 加载"""
    engine = st.session_state.engine
    if engine is not None and engine.index is not None:
        return sync_engine(engine)
    try:
        engine = RAGEngine.load_index(
            persist_dir=st.session_state.persist_dir,
            collection_name=st.session_state.collection,
            similarity_top_k=st.session_state.top_k,
        )
        st.session_state.engine = engine
        st.session_state.load_error = ""
        return engine
    except Exception as exc:
        st.session_state.load_error = str(exc)
        return None


# ----------------------------------------------------------------------
# 侧边栏
# ----------------------------------------------------------------------
def sidebar() -> str:
    st.sidebar.title("⚙️ 系统设置")

    st.sidebar.caption(f"配置文件：{ENV_FILE or '未找到 .env'}")

    api_key = st.sidebar.text_input(
        "DeepSeek API Key",
        value=DEEPSEEK_API_KEY,
        type="password",
        help="默认从 RAG/.env 的 DEEPSEEK_API_KEY 读取",
    )
    if api_key and api_key != DEEPSEEK_API_KEY:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        # 密钥变化后需要重建 LLM 客户端
        engine = st.session_state.engine
        if engine is not None:
            engine._llm = None

    model = st.sidebar.selectbox(
        "DeepSeek 模型",
        options=["deepseek-chat", "deepseek-reasoner"],
        index=0 if DEEPSEEK_MODEL != "deepseek-reasoner" else 1,
    )
    st.session_state.llm_model = model

    st.session_state.top_k = st.sidebar.slider(
        "检索片段数 top_k", min_value=1, max_value=10, value=SIMILARITY_TOP_K
    )

    mode_labels = list(ANSWER_MODE_LABELS)
    default_mode = (
        ANSWER_MODE if ANSWER_MODE in ANSWER_MODE_LABELS.values() else "hybrid"
    )
    mode_label = st.sidebar.radio(
        "回答模式",
        options=mode_labels,
        index=mode_labels.index(ANSWER_MODE_NAMES[default_mode]),
        help=(
            "混合：优先用知识库资料，资料不足时用 DeepSeek 自身知识补充并明确标注；"
            "严格：只依据知识库，查不到就说未提及；"
            "通用：不看知识库，直接由 DeepSeek 用自己的知识回答。"
        ),
    )
    st.session_state.answer_mode = ANSWER_MODE_LABELS[mode_label]

    st.session_state.data_dir = st.sidebar.text_input(
        "知识库目录", value=DATA_DIR, help="默认 data/raw"
    )
    st.session_state.persist_dir = st.sidebar.text_input(
        "向量库目录", value=CHROMA_PERSIST_DIR
    )

    st.sidebar.divider()
    st.sidebar.caption(
        f"Embedding：{EMBED_MODEL}\n\n"
        "DeepSeek 不提供 embedding 接口，向量化使用本地模型。"
    )

    return st.sidebar.radio("选择功能", ["文档管理", "智能问答", "系统评估"])


# ----------------------------------------------------------------------
# 文档管理
# ----------------------------------------------------------------------
def document_management_page() -> None:
    st.header("📄 文档管理")

    data_dir = Path(st.session_state.data_dir)
    st.write(f"知识库目录：`{data_dir}`")

    if data_dir.exists():
        files = sorted(
            p
            for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        st.info(f"当前知识库共 {len(files)} 个文件（PDF / TXT / MD / DOCX）")
        if files:
            with st.expander("查看文件列表"):
                for path in files:
                    size_mb = path.stat().st_size / 1024 / 1024
                    st.write(f"- {path.name}（{size_mb:.2f} MB）")
    else:
        st.error(f"知识库目录不存在：{data_dir}")

    st.divider()
    st.subheader("上传补充文档")

    uploaded_files = st.file_uploader(
        "上传文档",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
        help="上传的文件会保存到「知识库目录」，然后可以一起建索引",
    )

    if uploaded_files:
        data_dir.mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            (data_dir / file.name).write_bytes(file.getbuffer())
        st.success(f"✓ 已保存 {len(uploaded_files)} 个文件到 {data_dir}")

    st.divider()
    col1, col2 = st.columns([1, 3])
    with col1:
        rebuild = st.checkbox("重建索引", value=True, help="勾选会先清空旧集合")
    with col2:
        build_clicked = st.button("🚀 构建索引", type="primary")

    engine = get_engine()
    existing = engine.count_nodes()
    if existing:
        st.caption(f"当前向量库已有 {existing} 个 chunk")

    if build_clicked:
        st.session_state.documents_loaded = False
        with st.spinner("正在解析文档、向量化并写入 Chroma（首次会加载本地 BGE 模型）..."):
            start = time.time()
            try:
                documents = engine.load_documents(str(data_dir))
                if not documents:
                    st.warning("⚠️ 知识库中没有可解析的文档")
                    return
                nodes = engine.build_index(
                    documents, rebuild=rebuild, show_progress=False
                )
                st.session_state.engine = engine
                st.session_state.documents_loaded = True
                st.success(
                    f"✅ 构建完成：{len(documents)} 个文本块 → {nodes} 个 chunk，"
                    f"耗时 {time.time() - start:.1f}s"
                )
                st.caption(f"向量库：{st.session_state.persist_dir} / 集合：{engine.collection_name}")
                if getattr(engine, "last_skipped", None):
                    st.warning(f"跳过的文件：{', '.join(engine.last_skipped)}")
            except Exception as exc:
                st.error(f"❌ 构建失败：{exc}")


# ----------------------------------------------------------------------
# 智能问答
# ----------------------------------------------------------------------
def qa_page() -> None:
    st.header("💬 智能问答")

    mode = st.session_state.answer_mode

    engine = load_engine_or_none()
    if engine is None and mode == "general":
        # 通用模式不检索知识库，没有索引也能问
        engine = sync_engine(st.session_state.engine or RAGEngine(
            persist_dir=st.session_state.persist_dir,
            collection_name=st.session_state.collection,
            similarity_top_k=st.session_state.top_k,
        ))
        st.session_state.engine = engine
    if engine is None:
        st.warning("⚠️ 还没有可用索引，请先在「文档管理」页面构建索引。")
        if st.session_state.load_error:
            st.caption(st.session_state.load_error)
        return

    ensure_llm(engine)

    st.caption(
        f"回答模式：{ANSWER_MODE_NAMES.get(mode, mode)} ｜ 模型：{st.session_state.llm_model} "
        f"｜ top_k：{st.session_state.top_k} ｜ 索引 chunk：{engine.count_nodes()}"
    )

    # 历史对话
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("mode"):
                st.caption(f"回答模式：{ANSWER_MODE_NAMES.get(turn['mode'], turn['mode'])}")
            if turn.get("sources"):
                with st.expander("📚 参考来源"):
                    for i, source in enumerate(turn["sources"], start=1):
                        page = source.get("page_number")
                        page_text = f" 第 {page} 页" if page else ""
                        st.markdown(
                            f"**[{i}] {source['file_name']}{page_text}**"
                            f"　相似度 {source['score']:.3f}"
                        )
                        st.write(source["content"])

    question = st.text_area(
        "输入你的问题",
        placeholder="例如：激光雷达在林业上有哪些应用？",
        height=100,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_clicked = st.button("提问", type="primary")
    with col2:
        if st.button("清空对话"):
            st.session_state.history = []
            st.rerun()

    if ask_clicked:
        if not question.strip():
            st.warning("请输入问题")
            return
        with st.spinner("正在检索资料并请求 DeepSeek..."):
            try:
                start = time.time()
                answer, sources = engine.query(question.strip(), mode=mode)
            except Exception as exc:
                st.error(f"❌ 出错：{exc}")
                return

        st.session_state.history.append(
            {
                "question": question.strip(),
                "answer": answer,
                "sources": sources,
                "mode": mode,
            }
        )
        st.caption(
            f"本次耗时 {time.time() - start:.1f}s ｜ 模式：{ANSWER_MODE_NAMES.get(mode, mode)}"
        )
        st.rerun()


# ----------------------------------------------------------------------
# 系统评估
# ----------------------------------------------------------------------
def evaluation_page() -> None:
    st.header("📊 系统评估")

    engine = load_engine_or_none()
    if engine is None:
        st.warning("⚠️ 请先构建 / 加载索引")
        return

    st.write(
        "上传黄金数据集（JSON 数组，每项含 `question`，可选 `answer`），"
        "由 DeepSeek 对「忠实度」和「相关性」打分。"
    )

    golden_file = st.file_uploader("上传黄金数据集", type=["json"])

    if golden_file:
        data_dir = Path("data/eval")
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = data_dir / "golden_dataset.json"
        dataset_path.write_bytes(golden_file.getbuffer())
        st.success(f"✓ 数据集已保存到 {dataset_path}")

        try:
            preview = json.loads(dataset_path.read_text(encoding="utf-8"))
            st.caption(f"数据集共 {len(preview)} 条")
        except Exception as exc:
            st.error(f"数据集格式错误：{exc}")
            return

        if st.button("开始评估", type="primary"):
            with st.spinner("正在逐条评估（每条需要 1-2 次 DeepSeek 调用）..."):
                try:
                    evaluator = RAGEvaluator(
                        build_llm(
                            model=st.session_state.llm_model,
                            temperature=0.0,
                            max_tokens=32,
                        )
                    )
                    results = evaluator.run_evaluation(engine, str(dataset_path))
                    st.session_state.eval_result = results
                except Exception as exc:
                    st.error(f"❌ 评估失败：{exc}")
                    return

    results = st.session_state.eval_result
    if not results:
        return

    st.subheader("评估结果")
    col1, col2 = st.columns(2)
    col1.metric("忠实度", f"{results['faithfulness']:.2%}", help="答案基于文档的程度")
    col2.metric("相关性", f"{results['relevancy']:.2%}", help="答案与问题的相关程度")
    st.info(f"✓ 评估了 {results['num_evaluated']} 个问答对")

    avg = (results["faithfulness"] + results["relevancy"]) / 2
    if avg > 0.8:
        st.success("🏆 优秀！系统表现良好")
    elif avg > 0.6:
        st.info("👍 良好，仍有优化空间")
    else:
        st.warning("⚠️ 需要改进")

    for i, item in enumerate(results.get("details", []), start=1):
        with st.expander(f"{i}. {item.get('question', '')[:40]}..."):
            st.write(item.get("answer", item.get("error", "")))
            if "faithfulness" in item:
                st.caption(
                    f"忠实度 {item['faithfulness']:.2f} ｜ 相关性 {item['relevancy']:.2f} "
                    f"｜ 来源：{', '.join(item.get('sources', []))}"
                )


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main() -> None:
    init_state()

    st.markdown('<p class="main-header">🌲 InteliKB-Lite</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">林业知识库智能问答系统 ｜ DeepSeek + 本地 BGE + Chroma</p>',
        unsafe_allow_html=True,
    )

    page = sidebar()

    if page == "文档管理":
        document_management_page()
    elif page == "智能问答":
        qa_page()
    else:
        evaluation_page()


if __name__ == "__main__":
    main()
