"""
LLM-as-Judge 判分器：用LLM本身来判断候选答案和标准答案是否语义等价，
弥补规则判分（关键词子串匹配）认不出同义表达的问题——
这正是昨天EOS那几道题被规则判分误判的原因。
"""

from __future__ import annotations
import json
import re
from .llm import TokenMeter

JUDGE_PROMPT_TEMPLATE = """你是一个严格但公正的判分员，负责判断"候选答案"在语义上是否等价于"标准答案"。

问题：{question}
标准答案：{reference}
候选答案：{candidate}

判分规则：
- 只要候选答案表达的意思和标准答案一致，即使措辞、语序、详略程度不同，也判正确
- 如果候选答案缺少关键信息、方向反了、或者数值明显不对，判错误
- 数值类答案允许合理的四舍五入/单位换算误差

只输出一个JSON：{{"correct": true/false, "reason": "一句话说明理由"}}
"""


def llm_judge_grade(llm_client, question: str, candidate: str, reference: str, meter: TokenMeter | None = None):
    """返回 (is_correct: bool, reason: str)"""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, reference=reference, candidate=candidate)
    resp = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    if meter is not None:
        meter.add(resp)
    try:
        match = re.search(r"\{.*\}", resp.text, re.DOTALL)
        parsed = json.loads(match.group(0))
        return bool(parsed.get("correct", False)), parsed.get("reason", "")
    except Exception:
        return False, f"[judge解析失败] 原始输出: {resp.text[:100]}"
