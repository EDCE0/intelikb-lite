# 文件名：src/evaluator.py
"""
RAG 评估器

用 DeepSeek 给「忠实度」和「相关性」打分（0-1），替代原来写死的 gpt-3.5-turbo。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import DEEPSEEK_MODEL
from .deepseek_llm import DeepSeekLLM, build_llm

# 只要求模型输出一个数字，降低解析失败率
SCORE_FORMAT_HINT = "只输出一个 0 到 1 之间的小数，不要输出任何解释或单位。"


class RAGEvaluator:
    """RAG 评估器（DeepSeek 作为裁判模型）"""

    def __init__(self, llm: Optional[DeepSeekLLM] = None) -> None:
        self.llm = llm or build_llm(
            model=DEEPSEEK_MODEL, temperature=0.0, max_tokens=32
        )

    # ------------------------------------------------------------------
    # 打分
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_score(text: str, default: float = 0.5) -> float:
        """从模型输出里解析 0-1 分数，兼容 0.8 / 8 / 80% 等写法"""
        if not text:
            return default
        match = re.search(r"\d+(?:\.\d+)?", text)
        if not match:
            return default
        score = float(match.group())
        if "%" in text and score > 1:
            score /= 100.0
        elif 1 < score <= 10:
            score /= 10.0
        elif 10 < score <= 100:
            score /= 100.0
        return max(0.0, min(1.0, score))

    def _ask_score(self, prompt: str) -> float:
        try:
            response = self.llm.complete(prompt)
            return self._parse_score(response.text)
        except Exception as exc:
            print(f"⚠️  评估调用失败，按默认分 0.5 计：{exc}")
            return 0.5

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """忠实度：答案是否只依据参考资料"""
        context_text = "\n\n".join(
            f"文档{i + 1}: {c}" for i, c in enumerate(contexts)
        ) or "（无参考资料）"

        prompt = (
            "请评估答案是否完全基于参考文档，不包含文档之外或编造的信息。\n"
            "1 表示答案完全有文档依据，0 表示答案与文档无关或大量编造。\n"
            f"{SCORE_FORMAT_HINT}\n\n"
            f"参考文档：\n{context_text}\n\n"
            f"答案：\n{answer}\n\n"
            "评分："
        )
        return self._ask_score(prompt)

    def evaluate_relevancy(self, question: str, answer: str) -> float:
        """相关性：答案是否正面回答了问题"""
        prompt = (
            "请评估答案是否直接、完整地回答了问题。\n"
            "1 表示完全回答了问题，0 表示答非所问或没有提供有效信息。\n"
            f"{SCORE_FORMAT_HINT}\n\n"
            f"问题：{question}\n"
            f"答案：{answer}\n\n"
            "评分："
        )
        return self._ask_score(prompt)

    # ------------------------------------------------------------------
    # 数据集与批量评估
    # ------------------------------------------------------------------
    def load_golden_dataset(self, path: str) -> List[Dict[str, Any]]:
        """加载黄金数据集（JSON 列表，每项至少含 question）"""
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        if not isinstance(dataset, list) or not dataset:
            raise ValueError("黄金数据集必须是非空 JSON 数组")
        return dataset

    def evaluate_one(
        self, rag_engine, question: str, mode: str = "strict"
    ) -> Tuple[str, List[Dict[str, Any]], float, float]:
        """单条问答：检索 -> 生成 -> 打分

        评估固定使用 strict 模式：只有严格依据知识库作答，
        「忠实度」才真正反映检索增强的效果；hybrid 模式会引入模型补充内容。
        """
        answer, sources = rag_engine.query(question, mode=mode)
        contexts = [s["content"] for s in sources]
        faith = self.evaluate_faithfulness(answer, contexts)
        rel = self.evaluate_relevancy(question, answer)
        return answer, sources, faith, rel

    def run_evaluation(self, rag_engine, golden_dataset_path: str) -> Dict[str, Any]:
        """
        运行评估

        Args:
            rag_engine: RAG 引擎
            golden_dataset_path: 黄金数据集路径

        Returns:
            评估结果（平均忠实度、平均相关性、明细）
        """
        dataset = self.load_golden_dataset(golden_dataset_path)

        faithfulness_scores: List[float] = []
        relevancy_scores: List[float] = []
        details: List[Dict[str, Any]] = []

        for item in dataset:
            question = item["question"]
            try:
                answer, sources, faith, rel = self.evaluate_one(rag_engine, question)
            except Exception as exc:
                details.append({"question": question, "error": str(exc)})
                continue

            faithfulness_scores.append(faith)
            relevancy_scores.append(rel)
            details.append(
                {
                    "question": question,
                    "answer": answer,
                    "ground_truth": item.get("answer", ""),
                    "faithfulness": faith,
                    "relevancy": rel,
                    "sources": [s["file_name"] for s in sources],
                }
            )

        if not faithfulness_scores:
            raise RuntimeError("所有题目都评估失败，请检查 API Key / 向量库状态")

        return {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
            "relevancy": sum(relevancy_scores) / len(relevancy_scores),
            "num_evaluated": len(faithfulness_scores),
            "details": details,
        }
