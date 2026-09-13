"""
消融实验：分别去掉 calculator 和 knowledge_search，
用规则判分（不跑judge，省时间省token）对比三种工具配置下的正确率，
定位Agent相对baseline的提升到底来自谁。
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sciharness.agent import Agent
from sciharness.llm import DeepSeekClient
from sciharness.rag import SimpleRetriever
from sciharness.tools import CalculatorTool, KnowledgeBaseTool
from sciharness import config
from run_eval import grade  # 复用已有的规则判分逻辑，不重复造轮子

base_dir = os.path.dirname(__file__)
with open(os.path.join(base_dir, "questions.json"), encoding="utf-8") as f:
    questions = json.load(f)

retriever = SimpleRetriever(os.path.join(base_dir, "knowledge_base"), top_k=config.RAG_TOP_K)
llm_client = DeepSeekClient(config.API_KEY, config.DEEPSEEK_BASE_URL, config.DEEPSEEK_MODEL)

calc = CalculatorTool()
kb = KnowledgeBaseTool(retriever)

configs = {
    "full (calc+kb)": {calc.name: calc, kb.name: kb},
    "no_calculator (kb only)": {kb.name: kb},
    "no_retrieval (calc only)": {calc.name: calc},
}

for label, tools in configs.items():
    agent = Agent(llm_client=llm_client, tools=tools, max_steps=config.MAX_STEPS,
                  max_reflection_retry=config.MAX_REFLECTION_RETRY, long_term_memory=None)
    correct = 0
    for q in questions:
        result = agent.run(q["question"])
        if grade(result.final_answer, q["reference_answer"]):
            correct += 1
    print(f"{label}: {correct}/{len(questions)} = {correct/len(questions):.1%}")
