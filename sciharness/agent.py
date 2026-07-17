"""
Agent Loop 核心实现：Planning -> Tool Call -> Observation -> Reflection

设计上刻意保持简单直接（没有用现成的LangChain agent），
是为了让"每一步在做什么、为什么失败"都是可追踪、可解释的——
这正是评测/失效分析类岗位（Shopee JD里的Bad Case深度分析）真正关心的东西，
而不是简单调一个封装好的框架黑盒。
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .llm import LLMResponse, TokenMeter
from .memory import ShortTermMemory, LongTermMemory
from .tools import BaseTool

SYSTEM_PROMPT_TEMPLATE = """你是一个科学推理助手，需要一步步解决用户的问题。

可用工具：
{tool_descriptions}

每一步你必须只输出一个 JSON 对象，不要有任何多余文字，格式如下：
{{"thought": "你的推理过程", "action": "工具名或finish", "action_input": "工具输入或最终答案"}}

规则：
- 如果还需要计算或查资料，action填工具名，action_input填工具的输入
- 如果已经能给出最终答案，action填"finish"，action_input填最终答案（只写答案本身，不要解释）
- 每一步都要基于之前的观察结果（observation）推进，不要重复已经做过的事
"""

REFLECTION_PROMPT_TEMPLATE = """请检查下面这个候选答案是否可靠。

问题：{question}
推理过程摘要：
{trace}
候选答案：{answer}

只输出一个JSON：{{"satisfied": true/false, "reason": "简短原因"}}
如果发现推理链断裂、工具结果没被正确使用、或答案明显不合理，satisfied填false。
"""


@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: str
    observation: Optional[str] = None


@dataclass
class AgentResult:
    question: str
    final_answer: str
    steps: List[AgentStep] = field(default_factory=list)
    token_meter: TokenMeter = field(default_factory=TokenMeter)
    failure_mode: Optional[str] = None  # tool_hallucination / context_loss / broken_chain / None
    reflection_retries: int = 0


def _extract_json(text: str) -> Dict[str, Any]:
    """LLM有时会在JSON前后加废话，这里做容错提取"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"无法从模型输出中解析出JSON: {text!r}")
    return json.loads(match.group(0))


class Agent:
    def __init__(
        self,
        llm_client,
        tools: Dict[str, BaseTool],
        max_steps: int = 6,
        max_reflection_retry: int = 1,
        long_term_memory: Optional[LongTermMemory] = None,
    ):
        self.llm = llm_client
        self.tools = tools
        self.max_steps = max_steps
        self.max_reflection_retry = max_reflection_retry
        self.long_term_memory = long_term_memory

    def _tool_descriptions(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    def run(self, question: str) -> AgentResult:
        short_term = ShortTermMemory()
        meter = TokenMeter()
        steps: List[AgentStep] = []

        recalled = self.long_term_memory.recall(question) if self.long_term_memory else None
        if recalled:
            short_term.add("memory", f"历史经验参考: {recalled}")

        reflection_retries = 0
        final_answer = ""
        failure_mode = None

        for step_idx in range(self.max_steps):
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                tool_descriptions=self._tool_descriptions()
            )
            user_prompt = (
                f"问题：{question}\n\n"
                f"已有进展：\n{short_term.render_for_prompt() or '（尚未开始）'}"
            )
            resp = self.llm.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            meter.add(resp)

            try:
                parsed = _extract_json(resp.text)
                thought = parsed.get("thought", "")
                action = parsed.get("action", "")
                action_input = str(parsed.get("action_input", ""))
            except (ValueError, json.JSONDecodeError):
                failure_mode = "broken_chain"
                final_answer = "[解析失败，未能得到有效动作]"
                break

            step = AgentStep(thought=thought, action=action, action_input=action_input)

            if action == "finish":
                # 进入反思环节，而不是直接采信
                trace_text = "\n".join(
                    f"{i+1}. thought={s.thought} action={s.action} obs={s.observation}"
                    for i, s in enumerate(steps)
                )
                should_retry, reason = self._reflect(
                    question, trace_text, action_input, meter
                )
                if should_retry and reflection_retries < self.max_reflection_retry:
                    reflection_retries += 1
                    short_term.add(
                        "reflection",
                        f"上一个候选答案 '{action_input}' 被认为不可靠，原因: {reason}，请重新推理。",
                    )
                    steps.append(step)
                    continue
                final_answer = action_input
                steps.append(step)
                break

            if action not in self.tools:
                failure_mode = "tool_hallucination"
                step.observation = f"[错误] 工具 '{action}' 不存在，可用工具: {list(self.tools.keys())}"
                steps.append(step)
                short_term.add("observation", step.observation)
                continue

            observation = self.tools[action].run(action_input)
            step.observation = observation
            steps.append(step)
            short_term.add("thought", thought)
            short_term.add("action", f"{action}({action_input})")
            short_term.add("observation", observation)
        else:
            failure_mode = failure_mode or "broken_chain"
            final_answer = final_answer or "[达到最大步数仍未得出结论]"

        if self.long_term_memory and final_answer and not failure_mode:
            self.long_term_memory.remember(
                question[:50], f"最终采用的方法: {steps[-1].action if steps else 'N/A'}"
            )

        return AgentResult(
            question=question,
            final_answer=final_answer,
            steps=steps,
            token_meter=meter,
            failure_mode=failure_mode,
            reflection_retries=reflection_retries,
        )

    def _reflect(self, question, trace_text, candidate_answer, meter: TokenMeter):
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            question=question, trace=trace_text or "（无中间步骤）", answer=candidate_answer
        )
        resp = self.llm.chat([{"role": "user", "content": prompt}])
        meter.add(resp)
        try:
            parsed = _extract_json(resp.text)
            return not parsed.get("satisfied", True), parsed.get("reason", "")
        except (ValueError, json.JSONDecodeError):
            return False, ""


def run_baseline(llm_client, question: str) -> AgentResult:
    """
    对照组：不给工具、不反思，直接让模型一次性回答。
    用于和完整Agent对比，衡量Agent Loop到底带来多少提升——
    这个数字（成功率/幻觉率差异）才是简历上百分比的真实来源。
    """
    meter = TokenMeter()
    resp = llm_client.chat(
        [{"role": "user", "content": f"请直接回答，只给答案不要解释：{question}"}]
    )
    meter.add(resp)
    return AgentResult(question=question, final_answer=resp.text.strip(), token_meter=meter)
