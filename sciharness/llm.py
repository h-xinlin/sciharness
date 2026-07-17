"""
LLM 调用的统一入口。所有对 DeepSeek 的请求都过这里，
方便统一记录 token 消耗（对应简历里"降低约30% Token消耗"这类指标的数据来源）。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TokenMeter:
    """累加一次任务/一次实验里消耗的总 token，用于跑 baseline vs 优化版对比时算真实节省比例"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, resp: LLMResponse):
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        self.calls += 1

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class DeepSeekClient:
    """
    真正联网调用 DeepSeek 的实现。需要 `pip install openai` 并设置 DEEPSEEK_API_KEY。
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI  # 延迟导入，避免没装包时整个模块都import失败

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )


class MockLLMClient:
    """
    离线可跑的假客户端：用固定/规则化的回复替代真实API。
    用途：
    1. 没有网络或没有API key时，照样能跑通整个pipeline，验证代码逻辑没有bug
    2. 单元测试
    真正出成绩、出报告数字的时候，把 DeepSeekClient 换上去即可，Agent 代码完全不用改。
    """

    def __init__(self, script: List[str] | None = None):
        # script: 预设的回复队列，按调用顺序依次返回；用完了就返回一个兜底回复
        self._script = list(script) if script else []
        self._default = '{"thought": "mock", "action": "finish", "action_input": "42"}'

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> LLMResponse:
        text = self._script.pop(0) if self._script else self._default
        # 粗略估算token数（真实场景由API返回，这里仅用于离线联调）
        approx_prompt_tokens = sum(len(m["content"]) for m in messages) // 2
        approx_completion_tokens = len(text) // 2
        return LLMResponse(
            text=text,
            prompt_tokens=approx_prompt_tokens,
            completion_tokens=approx_completion_tokens,
        )
