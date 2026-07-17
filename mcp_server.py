"""
SciHarness的MCP Server：把calculator和knowledge_search包装成标准MCP协议的工具。

跟sciharness/tools.py里那套自定义JSON协议的关键区别：
- 这里用官方 `mcp` SDK的 FastMCP 类定义工具，工具的名字/描述/参数schema
  都是标准化声明的，任何支持MCP的客户端（不只是我们自己的Agent）都能直接
  发现并调用这些工具，不需要为SciHarness这个项目单独写一套对接代码
- 通信走的是MCP标准的stdio传输（Server作为一个独立子进程运行，Client通过
  标准输入输出用JSON-RPC协议跟它对话），不是Python函数直接调用

运行方式：一般不直接运行这个文件，而是由MCP Client把它当子进程启动
（见 mcp_client_demo.py）
"""

import math
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from sciharness.rag import SimpleRetriever

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SciHarness Tools")

_ALLOWED_NAMES = {k: v for k, v in vars(math).items() if not k.startswith("__")}

_retriever = SimpleRetriever(
    os.path.join(os.path.dirname(__file__), "benchmark", "knowledge_base"),
    top_k=3,
)


@mcp.tool()
def calculator(expression: str) -> str:
    """用于数值计算，输入一个数学表达式（可以用sqrt/sin/cos/log/pi等），
    例如 'sqrt(2)*3.14' 或 '(1074+273.15)**0.5'。"""
    expr = expression.strip()
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\,\s\w]+", expr):
        return f"[calculator error] 表达式包含不允许的字符: {expr}"
    try:
        result = eval(expr, {"__builtins__": {}}, _ALLOWED_NAMES)
        return str(result)
    except Exception as e:
        return f"[calculator error] {e}"


@mcp.tool()
def knowledge_search(query: str) -> str:
    """在本地知识库（热力学/量子力学/物态方程笔记）中检索相关内容，
    输入检索关键词或问题本身。"""
    hits = _retriever.search(query)
    if not hits:
        return "[knowledge_search] 未检索到相关内容"
    return "\n---\n".join(
        f"(来源: {doc_id}, 相似度 {score:.2f})\n{text}"
        for doc_id, text, score in hits
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
