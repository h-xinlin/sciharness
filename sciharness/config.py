"""
全局配置。API Key 通过环境变量注入，不要硬编码到代码里。

使用方式（本地运行前）：
    export DEEPSEEK_API_KEY="sk-xxxx"
"""

import os

# DeepSeek 使用 OpenAI 兼容接口
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Agent Loop 参数
MAX_STEPS = 6          # 单次任务最多允许多少轮 Plan-Act-Observe 循环
MAX_REFLECTION_RETRY = 1  # 反思后允许重试几次

# 短期记忆窗口（保留最近多少轮对话）
SHORT_TERM_WINDOW = 8

# RAG 检索参数
RAG_TOP_K = 3
