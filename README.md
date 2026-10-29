# InteliKB-Lite

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14-orange)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-4B6BFB)
![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-green)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

面向林业文献的**本地知识库 RAG 问答系统**：DeepSeek 生成答案、本地 BGE 中文向量模型检索、Chroma 持久化向量库。

支持**页码级引用溯源**、**三种回答模式**（严格 / 混合 / 通用）、**自动化评估**，并提供 Streamlit Web 界面与命令行工具。

---

## 1. 功能模块

| 模块 | 能力 |
| --- | --- |
| 📄 **文档管理** | 扫描知识库目录、上传 PDF/TXT/MD/DOCX、一键重建索引；逐页解析 → 中文清洗 → 分块 → 向量化 → 写 Chroma；返回文本块数 / chunk 数 / 耗时，提示解析失败与空页文件 |
| 💬 **智能问答** | 检索 top-k 片段交由 DeepSeek 生成答案；答案以 `【文件名 第X页】` 标注依据，同步展示来源片段与相似度；支持多轮历史、清空对话、切换模型与 top_k |
| 📊 **系统评估** | 上传 JSON 黄金问答集，以 DeepSeek 为裁判，对每条回答打「忠实度 / 相关性」0–1 分，输出均分、评级与逐条明细；评估固定 `strict` 模式 |

三种回答模式（详见第 6 节）：

| 模式 | 是否检索 | 资料不足时 | 典型用途 |
| --- | --- | --- | --- |
| `strict` 严格 | 是 | 明确回答「资料中未提及」并说明缺什么 | 论文引用、必须 100% 有出处 |
| `hybrid` 混合（默认） | 是 | 另起「模型补充（非知识库内容）」一节，用 DeepSeek 通用知识补全并声明未经校验 | 日常问答 |
| `general` 通用 | 否 | 直接由模型作答 | 概念、常识等知识库外问题 |

---

## 2. 技术栈

- **RAG 框架**：LlamaIndex（`VectorStoreIndex` / `SentenceSplitter` / `RetrieverQueryEngine`）
- **大模型**：DeepSeek `deepseek-chat` / `deepseek-reasoner`（OpenAI 兼容接口，自封装 `CustomLLM`）
- **向量模型**：`BAAI/bge-small-zh-v1.5`（sentence-transformers，512 维，CPU 可跑）
- **向量库**：Chroma（余弦相似度，本地持久化）
- **文档解析**：pypdf（逐页）、docx2txt
- **界面 / 配置**：Streamlit、python-dotenv

---

## 3. 数据流

```text
data/raw/*.pdf|txt|md|docx
        │  pypdf 逐页解析 + 中文清洗（全角空格/硬换行/空页过滤）
        ▼
   Document（metadata: file_name, page_number）
        │  SentenceSplitter  chunk_size=800, overlap=100
        ▼
   Node（chunk 开头注入【来源：文件名 第X页】）
        │  BGE 批量编码（L2 归一化）
        ▼
   Chroma（cosine, persist: ./chroma_db）
        │  查询向量 top-k 检索
        ▼
   DeepSeek 生成答案（提示词约束引用格式、三种模式策略）
        ▼
   答案 + 来源片段（文件名 / 页码 / 相似度）
```

---

## 4. 目录结构

```text
intelikb_lite/
├── app.py                  # Streamlit：文档管理 / 智能问答 / 系统评估
├── requirements.txt
├── .env.example            # 配置模板（复制为 .env 使用）
├── data/
│   ├── raw/                # ★ 知识库（PDF/TXT/MD/DOCX，仓库不含原始文献）
│   ├── processed/          # 预留：中间处理结果
│   └── eval/               # 黄金问答集示例
├── scripts/
│   ├── check_env.py        # 环境自检：密钥 / 接口 / 向量模型 / 向量库
│   ├── build_index.py      # 命令行建索引
│   └── ask.py              # 命令行问答（支持 --mode）
├── src/
│   ├── config.py           # 统一读取 .env（多路径查找、缺密钥可操作报错）
│   ├── deepseek_llm.py     # DeepSeek CustomLLM（流式 / 重试 / reasoner）
│   ├── embeddings.py       # 本地 BGE 向量模型（批量编码 + 进程内缓存）
│   ├── rag_engine.py       # RAG 引擎（解析 / 建索引 / 检索 / 三种模式）
│   └── evaluator.py        # 评估器（DeepSeek 打分）
└── chroma_db/              # 向量库（运行后生成，已在 .gitignore 中）
```

---

## 5. 快速开始

### 5.1 安装依赖

```bash
python -m pip install -r requirements.txt
```

> 本地验证环境：Python 3.11 + conda 虚拟环境（CPU）。
> 国内网络可加镜像：`-i https://mirrors.aliyun.com/pypi/simple/`。

### 5.2 配置密钥

复制模板并填入 DeepSeek Key：

```bash
cp .env.example .env
# 编辑 .env：DEEPSEEK_API_KEY=sk-xxxx
```

`.env` 查找顺序：环境变量 `INTELIKB_ENV_FILE` → 项目根 `.env` → 上级目录 `.env`。
全部可配置项见 [.env.example](.env.example) 与 [src/config.py](src/config.py)。

### 5.3 放入资料并建索引

```bash
# 把 PDF/TXT/MD/DOCX 放进 data/raw/
python scripts/build_index.py              # 重建索引（默认读取 data/raw）
python scripts/build_index.py --limit 3    # 只处理前 3 个文件（调试）
python scripts/build_index.py --no-rebuild # 追加到已有集合
```

### 5.4 提问

```bash
# 命令行（--mode: strict / hybrid / general）
python scripts/ask.py -q "激光雷达在林业上有哪些应用？" --mode hybrid

# 交互式
python scripts/ask.py

# Web 界面
python -m streamlit run app.py
```

打开 <http://localhost:8501>，在侧边栏选择回答模式后提问。

### 5.5 环境自检

```bash
python scripts/check_env.py
```

依次检查：`.env` → DeepSeek 接口连通性 → 本地向量模型 → 向量库状态。

---

## 6. 三种回答模式

知识库不覆盖的问题（例如「森林经理学的专业课程包含什么」）采用不同策略：

```text
strict  ：资料中未提及……（仅知识库）
hybrid  ：先给资料覆盖部分 + 页码引用
          ── 模型补充（非知识库内容）──
          用 DeepSeek 通用知识补全（声明未经知识库验证）
general ：直接由 DeepSeek 通用知识回答（不检索）
```

切换方式：

- Web 界面：侧边栏「回答模式」
- 命令行：`scripts/ask.py --mode hybrid "..."`
- 默认值：`.env` 中 `ANSWER_MODE=hybrid`

> `hybrid` / `general` 的模型补充来自模型自身知识，可能过时或不准确，正式引用请以知识库资料为准。
> 系统评估固定使用 `strict`，确保「忠实度」指标反映检索增强本身的效果。

---

## 7. 关键设计说明

**从 OpenAI 迁移到 DeepSeek**

| 组件 | 原来 | 现在 |
| --- | --- | --- |
| 生成模型 | `llama_index.llms.openai.OpenAI` + `gpt-3.5-turbo` | `src/deepseek_llm.py` 的 `DeepSeekLLM` + `deepseek-chat` |
| 向量模型 | `OpenAIEmbedding` + `text-embedding-3-small` | `src/embeddings.py` 的 `BGEEmbedding` + `BAAI/bge-small-zh-v1.5` |
| 评估打分 | OpenAI `gpt-3.5-turbo` | `DeepSeekLLM` 打分 |
| 密钥 | `OPENAI_API_KEY` | `DEEPSEEK_API_KEY` |

**为什么向量模型也换了？**
DeepSeek 目前只提供对话 / 推理接口（`/chat/completions`），**没有 embedding 接口**，`OpenAIEmbedding` 不可用。
改用本地 BGE 中文模型：完全离线、中文检索效果好、无向量 API 成本；如未来 DeepSeek 提供向量接口，只需替换 [src/embeddings.py](src/embeddings.py) 的实现。

**为什么不用 `llama_index.llms.openai.OpenAI` 直接指向 DeepSeek？**
新版 `llama-index-llms-openai` 会按 OpenAI 模型白名单校验模型名，传 `deepseek-chat` 会直接抛
`ValueError: Unknown model`。因此基于 `CustomLLM` 封装 DeepSeek 的 OpenAI 兼容接口，
支持流式输出、失败重试与 `deepseek-reasoner`。

**引用可溯源**
建索引时每个 chunk 开头写入 `【来源：文件名 第X页】`，提示词要求模型沿用该标注，
答案中的依据可直接对应到具体文献与页码；检索结果同时返回相似度，便于人工复核。

**工程化细节**
查询引擎按「模式」惰性创建，建索引流程不依赖 API Key；BGE 模型做进程内缓存并优先本地缓存加载，
避免 Streamlit 反复 rerun 与 HF Hub 不可达带来的启动阻塞；附 `check_env` 自检脚本降低排障成本。

---

## 8. 可调参数（`.env`）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 可换 `deepseek-reasoner`（更慢更强推理） |
| `DEEPSEEK_TEMPERATURE` | `0.3` | 越低越稳定，reasoner 自动忽略 |
| `DEEPSEEK_MAX_TOKENS` | `2048` | 单次回答最大 token |
| `EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 可换 `BAAI/bge-m3`（更强但更慢） |
| `EMBED_DEVICE` | `cpu` | 有显卡可改 `cuda` |
| `ANSWER_MODE` | `hybrid` | `strict` / `hybrid` / `general` |
| `SIMILARITY_TOP_K` | `5` | 检索片段数 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | 分块大小 / 重叠 |
| `DATA_DIR` | `data/raw` | 知识库目录 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | 向量库目录 |

---

## 9. 评估

黄金数据集为 JSON 数组（示例见 [data/eval/golden_dataset.json](data/eval/golden_dataset.json)）：

```json
[
  {
    "question": "基于森林清查资料估算森林植被碳储量时采用了什么方法？",
    "answer": "用生物量回归模型估算乔木林生物量，以树种含碳率作为转换系数……"
  }
]
```

在 Web 界面「系统评估」上传后运行，DeepSeek 对每条回答打「忠实度 / 相关性」0–1 分。
评估固定 `strict` 模式，因此忠实度反映的是检索增强本身的效果，而非模型自身知识。

---

## 10. 常见问题

* **报 `未找到 DEEPSEEK_API_KEY`**：确认 `.env` 中已配置，或运行 `python scripts/check_env.py`
  查看实际读取的配置文件路径。
* **报 `向量库中不存在集合`**：先运行 `python scripts/build_index.py`。
* **索引重复 / 数据变多**：默认重建会清空旧集合，`--no-rebuild` 才是追加。
* **回答总是「资料中未提及」**：检查 `data/raw` 是否有可解析文本（扫描版 PDF 需先 OCR），
  或增大 `SIMILARITY_TOP_K`；也可切换到 `hybrid` 模式。
* **扫描版 PDF 无内容**：pypdf 只能抽取文本层，需要先用 OCR（如 PaddleOCR）转换。
* **改完 `src/` 后页面报 `ImportError: cannot import name ...`**：Streamlit 的 Rerun 只重新执行
  `app.py`，已导入的 `src.*` 模块仍在内存中。必须结束进程重启
  （`Ctrl+C` 或 `fuser -k 8501/tcp`），必要时清理 `__pycache__` 后重新启动。

---

## 11. 数据说明

仓库**不包含** `data/raw` 下的原始期刊文献（版权与体积原因），请自行准备资料放入该目录；
仓库中的 `data/eval/golden_dataset.json` 为自行撰写的示例问答对，仅用于演示评估流程。
