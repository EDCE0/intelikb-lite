#!/usr/bin/env python
# 文件名：scripts/check_env.py
"""
环境自检：确认 .env、DeepSeek 接口、本地向量模型、向量库是否正常

用法：
    python scripts/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    DEEPSEEK_API_BASE,
    DEEPSEEK_MODEL,
    EMBED_MODEL,
    describe,
    require_api_key,
)


def main() -> int:
    print("=" * 60)
    print(describe())
    print("=" * 60)

    failed = False

    # 1) API Key
    try:
        require_api_key()
        print("✅ DEEPSEEK_API_KEY 已配置")
    except RuntimeError as exc:
        print(f"❌ {exc}")
        failed = True

    # 2) DeepSeek 对话接口
    try:
        from src.deepseek_llm import DeepSeekLLM

        llm = DeepSeekLLM(temperature=0.0, max_tokens=32)
        reply = llm.complete("只回复四个字：连接成功")
        print(f"✅ DeepSeek 接口可用（{DEEPSEEK_MODEL} @ {DEEPSEEK_API_BASE}）")
        print(f"   模型回复：{str(reply).strip()[:50]}")
    except Exception as exc:
        print(f"❌ DeepSeek 接口不可用：{exc}")
        failed = True

    # 3) 本地向量模型
    try:
        from src.embeddings import BGEEmbedding

        embedding = BGEEmbedding()
        vector = embedding.get_text_embedding("森林资源监测")
        print(f"✅ 本地向量模型可用（{EMBED_MODEL}，维度 {len(vector)}）")
    except Exception as exc:
        print(f"❌ 向量模型加载失败：{exc}")
        failed = True

    # 4) 向量库
    try:
        from src.rag_engine import RAGEngine

        if RAGEngine.index_exists():
            count = RAGEngine().count_nodes()
            print(f"✅ 向量库已有索引，共 {count} 个 chunk")
        else:
            print("ℹ️  向量库为空，请运行：python scripts/build_index.py")
    except Exception as exc:
        print(f"⚠️  向量库检查失败：{exc}")

    print("=" * 60)
    print("自检通过 ✅" if not failed else "自检未通过 ❌ 请按上面的提示修复")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
