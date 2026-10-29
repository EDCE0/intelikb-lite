# 文件名：src/deepseek_llm.py
"""
DeepSeek 大模型适配器（OpenAI 兼容接口）

DeepSeek 官方接口与 OpenAI Chat Completions 协议兼容，但 llama-index 的
OpenAI 类会校验模型名（deepseek-chat 不在白名单里会直接报错），因此这里用
CustomLLM 自己封装一层，直接调用 openai SDK 的 /chat/completions。

支持：
  * deepseek-chat / deepseek-reasoner
  * 同步 / 异步（异步沿用基类默认实现）
  * 流式输出
  * 失败重试（由 openai SDK 负责）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import CustomLLM
from pydantic import Field, PrivateAttr

from .config import (
    DEEPSEEK_API_BASE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_CONTEXT_WINDOW,
    DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_TIMEOUT,
    require_api_key,
)

# 注意：这里只放「角色设定」，不放「只准用资料」这类约束。
# 具体约束由 rag_engine 的提示词控制（strict / hybrid / general 三种模式不同），
# 否则通用模式会被系统提示词限制住。
DEFAULT_SYSTEM_PROMPT = (
    "你是 InteliKB-Lite 知识库助手，擅长林业、遥感、生态等领域的资料整理与问答。\n"
    "请使用简洁、准确的中文回答，条理清晰，必要时分点说明；\n"
    "具体作答规则（是否只能依据资料、是否允许通用知识补充）以用户消息中的要求为准。"
)


class DeepSeekLLM(CustomLLM):
    """DeepSeek 对话模型（默认 deepseek-chat）"""

    model_name: str = Field(default=DEEPSEEK_MODEL, description="DeepSeek 模型名")
    api_key: str = Field(default=DEEPSEEK_API_KEY, description="DeepSeek API Key")
    api_base: str = Field(default=DEEPSEEK_API_BASE, description="DeepSeek API 地址")
    temperature: float = Field(default=DEEPSEEK_TEMPERATURE, description="采样温度")
    max_tokens: int = Field(default=DEEPSEEK_MAX_TOKENS, description="最大输出 token")
    context_window: int = Field(
        default=DEEPSEEK_CONTEXT_WINDOW, description="上下文窗口大小"
    )
    timeout: float = Field(default=DEEPSEEK_TIMEOUT, description="请求超时（秒）")
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT, description="系统提示词"
    )

    _client: Any = PrivateAttr(default=None)

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    def __init__(self, **data: Any) -> None:
        data.setdefault("api_key", DEEPSEEK_API_KEY)
        super().__init__(**data)
        # 构造时就校验密钥，避免出问题时只看到模糊的 401
        self.api_key = require_api_key(self.api_key)

    @classmethod
    def class_name(cls) -> str:
        return "deepseek_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
            # DeepSeek 支持 function calling，但本项目用不到，避免误导
            is_function_calling_model=False,
            model_name=self.model_name,
            system_role=MessageRole.SYSTEM,
        )

    def _get_client(self):
        """惰性创建 openai 客户端，指向 DeepSeek 接口"""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=DEEPSEEK_MAX_RETRIES,
            )
        return self._client

    # ------------------------------------------------------------------
    # 请求参数
    # ------------------------------------------------------------------
    def _is_reasoner(self) -> bool:
        return "reasoner" in self.model_name.lower()

    def _request_kwargs(self, stream: bool, **kwargs: Any) -> Dict[str, Any]:
        """组装 chat.completions 参数；reasoner 模型不接受 temperature"""
        params: Dict[str, Any] = {
            "model": self.model_name,
            "stream": stream,
            "max_tokens": kwargs.get("max_tokens") or self.max_tokens,
        }
        if not self._is_reasoner():
            params["temperature"] = kwargs.get("temperature", self.temperature)
        # 透传用户显式指定的其他参数（top_p、stop 等）
        for key in ("top_p", "stop", "frequency_penalty", "presence_penalty", "seed"):
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        return params

    def _to_openai_messages(
        self, messages: Sequence[ChatMessage]
    ) -> List[Dict[str, str]]:
        """llama-index ChatMessage -> OpenAI messages"""
        result: List[Dict[str, str]] = []
        has_system = False
        for message in messages:
            role = message.role.value
            if role not in ("system", "user", "assistant"):
                role = "user"
            if role == "system":
                has_system = True
            content = message.content
            result.append(
                {
                    "role": role,
                    "content": content if isinstance(content, str) else str(content),
                }
            )
        if not has_system and self.system_prompt:
            result.insert(0, {"role": "system", "content": self.system_prompt})
        return result

    @staticmethod
    def _extract_text(choice: Any) -> str:
        message = choice.message
        content = getattr(message, "content", None) or ""
        if not content.strip():
            # deepseek-reasoner 在思考阶段可能只有 reasoning_content
            content = getattr(message, "reasoning_content", None) or content
        return content

    # ------------------------------------------------------------------
    # 同步接口
    # ------------------------------------------------------------------
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        response = self._get_client().chat.completions.create(
            messages=self._to_openai_messages(messages),
            **self._request_kwargs(stream=False, **kwargs),
        )
        text = self._extract_text(response.choices[0])
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
            raw=response.model_dump(),
        )

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        stream = self._get_client().chat.completions.create(
            messages=self._to_openai_messages(messages),
            **self._request_kwargs(stream=True, **kwargs),
        )
        content = ""
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            piece = getattr(delta, "content", None) if delta else None
            if not piece:
                continue
            content += piece
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                delta=piece,
                raw=chunk.model_dump(),
            )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        messages: List[ChatMessage] = []
        if self.system_prompt:
            messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
            )
        messages.append(ChatMessage(role=MessageRole.USER, content=prompt))
        response = self.chat(messages, **kwargs)
        return CompletionResponse(text=response.message.content or "", raw=response.raw)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        messages: List[ChatMessage] = []
        if self.system_prompt:
            messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
            )
        messages.append(ChatMessage(role=MessageRole.USER, content=prompt))
        text = ""
        for chat_response in self.stream_chat(messages, **kwargs):
            delta = chat_response.delta or ""
            text += delta
            yield CompletionResponse(text=text, delta=delta, raw=chat_response.raw)


def build_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs: Any,
) -> DeepSeekLLM:
    """按需构造 DeepSeek LLM（供脚本 / Streamlit 复用）"""
    params: Dict[str, Any] = {}
    if model:
        params["model_name"] = model
    if api_key:
        params["api_key"] = api_key
    if temperature is not None:
        params["temperature"] = temperature
    params.update(kwargs)
    return DeepSeekLLM(**params)
