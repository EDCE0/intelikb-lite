#!/usr/bin/env python
# 文件名：scripts/build_index.py
"""
构建知识库索引（命令行版）

用法：
    python scripts/build_index.py                          # 用 .env 里的 DATA_DIR
    python scripts/build_index.py --data-dir data/raw       # 指定知识库目录
    python scripts/build_index.py --limit 3                 # 只处理前 3 个文件（调试）
    python scripts/build_index.py --no-rebuild              # 追加到已有集合
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 让脚本可以直接运行（把项目根目录加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, DATA_DIR, describe  # noqa: E402
from src.rag_engine import RAGEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 InteliKB-Lite 向量索引")
    parser.add_argument("--data-dir", default=DATA_DIR, help="知识库目录")
    parser.add_argument("--persist-dir", default=CHROMA_PERSIST_DIR, help="向量库目录")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="向量集合名")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（0 表示全部）")
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="不清空旧集合，直接追加（默认重建，避免重复入库）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(describe())
    print("-" * 60)

    engine = RAGEngine(persist_dir=args.persist_dir, collection_name=args.collection)

    start = time.time()
    print(f"📂 正在加载文档：{args.data_dir}")
    documents = engine.load_documents(args.data_dir, limit=args.limit or None)
    if not documents:
        print("❌ 没有加载到任何文档，请检查知识库目录与文件格式")
        return 1

    files = {doc.metadata.get("file_name") for doc in documents}
    print(f"✅ 已加载 {len(files)} 个文件 / {len(documents)} 个文本块（按页/按文件）")

    print("🔢 正在分块并向量化（首次运行会加载本地 BGE 模型）...")
    nodes = engine.build_index(
        documents, rebuild=not args.no_rebuild, show_progress=True
    )

    elapsed = time.time() - start
    print("-" * 60)
    print(f"✅ 索引构建完成：{nodes} 个 chunk 已写入 Chroma")
    print(f"   集合数量：{engine.count_nodes()}   耗时：{elapsed:.1f}s")
    print(f"   向量库目录：{args.persist_dir}")
    if getattr(engine, "last_skipped", None):
        print(f"   ⚠️ 跳过的文件（空内容或解析失败）：{', '.join(engine.last_skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
