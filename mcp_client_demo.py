"""
MCP Client demo：连接 mcp_server.py，走标准MCP协议做三件事：
1. 初始化连接（MCP协议握手）
2. 发现Server上有哪些工具（list_tools，不需要提前知道工具长什么样，
   这就是"标准化"的意义——换一个MCP Server，这段代码不用改）
3. 调用其中的工具，验证真实可用

运行：python3 mcp_client_demo.py
"""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="python3",
        args=[os.path.join(os.path.dirname(__file__), "mcp_server.py")],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. 协议握手
            await session.initialize()
            print("[1] MCP连接初始化成功\n")

            # 2. 发现Server上的工具（不需要提前硬编码工具列表）
            tools_result = await session.list_tools()
            print("[2] 发现的工具列表：")
            for tool in tools_result.tools:
                print(f"    - {tool.name}: {tool.description}")
            print()

            # 3. 调用 calculator 工具
            calc_result = await session.call_tool(
                "calculator", arguments={"expression": "5.67e-8 * 1000**4"}
            )
            print("[3] 调用 calculator('5.67e-8 * 1000**4') 结果：")
            print("   ", calc_result.content[0].text)
            print()

            # 4. 调用 knowledge_search 工具
            kb_result = await session.call_tool(
                "knowledge_search", arguments={"query": "斯特藩-玻尔兹曼定律"}
            )
            print("[4] 调用 knowledge_search('斯特藩-玻尔兹曼定律') 结果：")
            print("   ", kb_result.content[0].text[:200], "...")


if __name__ == "__main__":
    asyncio.run(main())
