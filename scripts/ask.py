#!/usr/bin/env python
# 文件名：scripts/ask.py
"""
命令行问答（先用 build_index.py 建好索引）

用法：
    python scripts/ask.py -q "森林生物量怎么估测？"
    python scripts/ask.py                    # 进入交互式问答
    python scripts/ask.py -q "..." -k 8      # 调整检索片段数
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    ANSWER_MODE,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    SIMILARITY_TOP_K,
    VALID_ANSWER_MODES,
)
from src.rag_engine import RAGEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InteliKB-Lite 命令行问答")
    parser.add_argument("-q", "--question", default="", help="要提问的问题")
    parser.add_argument("-k", "--top-k", type=int, default=SIMILARITY_TOP_K, help="检索片段数")
    parser.add_argument("--persist-dir", default=CHROMA_PERSIST_DIR, help="向量库目录")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="向量集合名")
    parser.add_argument(
        "--mode",
        choices=list(VALID_ANSWER_MODES),
        default=ANSWER_MODE,
        help="回答模式：strict 只用资料 / hybrid 资料优先+模型补充 / general 纯模型知识",
    )
    return parser.parse_args()


def print_result(answer: str, sources, elapsed: float) -> None:
    print("\n📝 回答：")
    print(answer)
    print(f"\n📚 参考来源（耗时 {elapsed:.1f}s）：")
    for i, source in enumerate(sources, start=1):
        page = source.get("page_number")
        page_text = f" 第 {page} 页" if page else ""
        print(f"  [{i}] {source['file_name']}{page_text}  相似度 {source['score']:.3f}")
        print(f"      {source['content'][:120]}...")


def ask(engine: RAGEngine, question: str, mode: str = "hybrid") -> None:
    start = time.time()
    answer, sources = engine.query(question, mode=mode)
    if mode == "general":
        print("\n⚠️ 通用模式：以下内容来自 DeepSeek 自身知识，未经知识库验证。")
    print_result(answer, sources, time.time() - start)


def main() -> int:
    args = parse_args()

    print("🔌 正在加载索引与 DeepSeek 模型...")
    try:
        engine = RAGEngine.load_index(
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            similarity_top_k=args.top_k,
            answer_mode=args.mode,
        )
    except Exception as exc:
        if args.mode == "general":
            # 通用模式不检索，没有索引也能问
            print(f"⚠️ 未加载知识库索引（{exc}），通用模式将直接用模型知识回答")
            engine = RAGEngine(
                persist_dir=args.persist_dir,
                collection_name=args.collection,
                similarity_top_k=args.top_k,
                answer_mode=args.mode,
            )
        else:
            print(f"❌ 加载索引失败：{exc}")
            print("   请先运行：python scripts/build_index.py")
            return 1

    print(f"✅ 索引就绪，共 {engine.count_nodes()} 个 chunk")

    print(f"回答模式：{args.mode}")

    if args.question:
        ask(engine, args.question, mode=args.mode)
        return 0

    print("进入交互式问答，输入 exit / quit 退出。")
    while True:
        try:
            question = input("\n❓ 问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        try:
            ask(engine, question, mode=args.mode)
        except Exception as exc:
            print(f"❌ 查询失败：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
