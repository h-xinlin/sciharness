"""
工具定义。每个 Tool 都有统一接口：name / description / run(input) -> str

新增工具时只需要：
1. 继承 BaseTool
2. 实现 run()
3. 在 TOOL_REGISTRY 里注册
"""

from __future__ import annotations
import math
import re
from typing import Dict


class BaseTool:
    name: str = "base_tool"
    description: str = "工具基类，不要直接使用"

    def run(self, tool_input: str) -> str:
        raise NotImplementedError


class CalculatorTool(BaseTool):
    """
    安全的数值计算工具。只允许数字、基本运算符和 math 模块里的白名单函数，
    不使用裸 eval，避免任意代码执行。
    """
    name = "calculator"
    description = (
        "用于数值计算，输入一个数学表达式（可以用 sqrt/sin/cos/log/pi 等），"
        "例如 'sqrt(2)*3.14' 或 '(1074+273.15)**0.5'。"
    )

    _ALLOWED_NAMES = {
        k: v for k, v in vars(math).items() if not k.startswith("__")
    }

    def run(self, tool_input: str) -> str:
        expr = tool_input.strip()
        if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\,\s\w]+", expr):
            return f"[calculator error] 表达式包含不允许的字符: {expr}"
        try:
            # 只暴露白名单函数，builtins 置空
            result = eval(expr, {"__builtins__": {}}, self._ALLOWED_NAMES)
            return str(result)
        except Exception as e:
            return f"[calculator error] {e}"


class KnowledgeBaseTool(BaseTool):
    """
    检索工具的壳子，真正的检索逻辑委托给 rag.SimpleRetriever。
    这样 Agent 侧只需要知道"有一个叫 knowledge_search 的工具"，
    不需要关心底层是 TF-IDF 还是向量检索。
    """
    name = "knowledge_search"
    description = (
        "在本地知识库（热力学 / 量子力学 / 物态方程笔记）中检索相关内容，"
        "输入检索关键词或问题本身。"
    )

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, tool_input: str) -> str:
        hits = self.retriever.search(tool_input)
        if not hits:
            return "[knowledge_search] 未检索到相关内容"
        return "\n---\n".join(
            f"(来源: {doc_id}, 相似度 {score:.2f})\n{text}"
            for doc_id, text, score in hits
        )


def build_tool_registry(retriever) -> Dict[str, BaseTool]:
    calc = CalculatorTool()
    kb = KnowledgeBaseTool(retriever)
    return {calc.name: calc, kb.name: kb}
