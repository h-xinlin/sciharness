import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sciharness.agent import Agent
from sciharness.llm import DeepSeekClient
from sciharness.rag import SimpleRetriever
from sciharness.tools import CalculatorTool, KnowledgeBaseTool
from sciharness import config
from run_eval import grade

base_dir = os.path.dirname(__file__)
with open(os.path.join(base_dir, "questions.json"), encoding="utf-8") as f:
    questions = json.load(f)

retriever = SimpleRetriever(os.path.join(base_dir, "knowledge_base"), top_k=config.RAG_TOP_K)
llm_client = DeepSeekClient(config.API_KEY, config.DEEPSEEK_BASE_URL, config.DEEPSEEK_MODEL)
calc = CalculatorTool()
kb = KnowledgeBaseTool(retriever)

configs = {
    "full": {calc.name: calc, kb.name: kb},
    "no_retrieval": {calc.name: calc},
}

results = {}
for label, tools in configs.items():
    agent = Agent(llm_client=llm_client, tools=tools, max_steps=config.MAX_STEPS,
                  max_reflection_retry=config.MAX_REFLECTION_RETRY, long_term_memory=None)
    per_q = {}
    for q in questions:
        r = agent.run(q["question"])
        per_q[q["id"]] = (grade(r.final_answer, q["reference_answer"]), r.failure_mode)
    results[label] = per_q

for qid in results["full"]:
    full_ok, full_fail = results["full"][qid]
    nr_ok, nr_fail = results["no_retrieval"][qid]
    if full_ok != nr_ok:
        print(f"[{qid}] full={full_ok}(fail={full_fail})  no_retrieval={nr_ok}(fail={nr_fail})")
