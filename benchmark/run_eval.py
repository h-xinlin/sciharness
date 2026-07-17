"""
评测入口脚本。

用法：
    # 先跑通逻辑（不用API key，离线mock模式，验证pipeline没bug）
    python run_eval.py --mock

    # 真正出成绩（需要先 export DEEPSEEK_API_KEY=sk-xxxx）
    python run_eval.py

    # 加上LLM-as-Judge判分，并和规则判分做一致性对比
    python run_eval.py --judge

输出：
    results/results.json          每题的详细trace和判分
    终端打印总体成功率 + 分学科成功率 + baseline vs agent的token对比
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sciharness.agent import Agent, run_baseline
from sciharness.llm import DeepSeekClient, MockLLMClient, TokenMeter
from sciharness.memory import LongTermMemory
from sciharness.rag import SimpleRetriever
from sciharness.tools import build_tool_registry
from sciharness.judge import llm_judge_grade
from sciharness import config


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\.\-]", "", s)
    return s


def grade(candidate: str, reference: str) -> bool:
    """
    简单规则判分：数字类答案做近似数值比较，文本类答案做归一化子串匹配。
    """
    cand_norm = normalize(candidate)
    ref_norm = normalize(reference)

    num_pattern = r"-?\d+\.?\d*e?-?\d*"
    cand_nums = re.findall(num_pattern, candidate.replace(",", ""))
    ref_nums = re.findall(num_pattern, reference.replace(",", ""))
    if ref_nums:
        try:
            ref_val = float(ref_nums[0])
            for cn in cand_nums:
                if abs(float(cn) - ref_val) <= max(abs(ref_val) * 0.05, 0.5):
                    return True
        except ValueError:
            pass

    return ref_norm in cand_norm or cand_norm in ref_norm


def build_llm_client(use_mock: bool):
    if use_mock or not config.API_KEY:
        if not use_mock:
            print("[警告] 未检测到 DEEPSEEK_API_KEY，自动切换到 mock 模式，"
                  "结果仅用于验证pipeline，不代表真实效果。", file=sys.stderr)
        return MockLLMClient()
    return DeepSeekClient(config.API_KEY, config.DEEPSEEK_BASE_URL, config.DEEPSEEK_MODEL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="强制使用离线mock LLM，不联网")
    parser.add_argument("--limit", type=int, default=None, help="只跑前N题，方便调试")
    parser.add_argument("--judge", action="store_true", help="额外用LLM-as-Judge判分，并和规则判分做一致性对比")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    retriever = SimpleRetriever(os.path.join(base_dir, "knowledge_base"), top_k=config.RAG_TOP_K)
    tools = build_tool_registry(retriever)
    llm_client = build_llm_client(args.mock)
    long_term = LongTermMemory(path=os.path.join(base_dir, "results", "long_term_memory.json"))

    agent = Agent(
        llm_client=llm_client,
        tools=tools,
        max_steps=config.MAX_STEPS,
        max_reflection_retry=config.MAX_REFLECTION_RETRY,
        long_term_memory=long_term,
    )

    records = []
    domain_stats = defaultdict(lambda: {"total": 0, "baseline_correct": 0, "agent_correct": 0})
    failure_counts = defaultdict(int)
    baseline_tokens_total = 0
    agent_tokens_total = 0
    judge_meter = TokenMeter()
    agreement_count = 0
    disagreement_examples = []

    for q in questions:
        print(f"[跑题] {q['id']}: {q['question'][:30]}...")

        baseline_result = run_baseline(llm_client, q["question"])
        agent_result = agent.run(q["question"])

        baseline_rule_ok = grade(baseline_result.final_answer, q["reference_answer"])
        agent_rule_ok = grade(agent_result.final_answer, q["reference_answer"])

        baseline_tokens_total += baseline_result.token_meter.total
        agent_tokens_total += agent_result.token_meter.total

        judge_agent_ok, judge_reason = None, None
        if args.judge:
            judge_agent_ok, judge_reason = llm_judge_grade(
                llm_client, q["question"], agent_result.final_answer,
                q["reference_answer"], meter=judge_meter,
            )
            if judge_agent_ok == agent_rule_ok:
                agreement_count += 1
            else:
                disagreement_examples.append({
                    "id": q["id"],
                    "agent_answer": agent_result.final_answer,
                    "reference": q["reference_answer"],
                    "rule_judged": agent_rule_ok,
                    "llm_judged": judge_agent_ok,
                    "judge_reason": judge_reason,
                })

        agent_final_ok = judge_agent_ok if args.judge else agent_rule_ok

        domain = q["domain"]
        domain_stats[domain]["total"] += 1
        domain_stats[domain]["baseline_correct"] += int(baseline_rule_ok)
        domain_stats[domain]["agent_correct"] += int(agent_final_ok)

        if agent_result.failure_mode:
            failure_counts[agent_result.failure_mode] += 1

        records.append({
            "id": q["id"],
            "domain": domain,
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "baseline_answer": baseline_result.final_answer,
            "baseline_correct": baseline_rule_ok,
            "baseline_tokens": baseline_result.token_meter.total,
            "agent_answer": agent_result.final_answer,
            "agent_correct_by_rule": agent_rule_ok,
            "agent_correct_by_judge": judge_agent_ok,
            "agent_correct_final": agent_final_ok,
            "judge_reason": judge_reason,
            "agent_tokens": agent_result.token_meter.total,
            "agent_steps": len(agent_result.steps),
            "agent_reflection_retries": agent_result.reflection_retries,
            "failure_mode": agent_result.failure_mode,
        })

    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    with open(os.path.join(base_dir, "results", "results.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    total = len(questions)
    baseline_correct = sum(r["baseline_correct"] for r in records)
    agent_correct = sum(r["agent_correct_final"] for r in records)

    print("\n" + "=" * 50)
    print(f"总题数: {total}")
    print(f"判分方式: {'LLM-as-Judge' if args.judge else '规则判分'}")
    print(f"Baseline 成功率: {baseline_correct}/{total} = {baseline_correct/total:.1%}")
    print(f"Agent    成功率: {agent_correct}/{total} = {agent_correct/total:.1%}")
    print(f"Baseline 总token: {baseline_tokens_total}")
    print(f"Agent    总token: {agent_tokens_total}")
    if args.judge:
        print(f"Judge判分消耗token: {judge_meter.total}")
    print("\n分学科成功率:")
    for domain, stat in domain_stats.items():
        t = stat["total"]
        print(f"  {domain}: baseline {stat['baseline_correct']}/{t}  |  agent {stat['agent_correct']}/{t}")
    print("\n失效模式统计:")
    for mode, cnt in failure_counts.items():
        print(f"  {mode}: {cnt}次")
    if args.judge:
        print(f"\n规则判分 vs LLM判分 一致率: {agreement_count}/{total} = {agreement_count/total:.1%}")
        if disagreement_examples:
            print(f"不一致的{len(disagreement_examples)}个案例：")
            for ex in disagreement_examples:
                print(f"  [{ex['id']}] 规则判{ex['rule_judged']} vs LLM判{ex['llm_judged']} | 理由: {ex['judge_reason']}")
    print("=" * 50)
    print(f"\n详细结果已写入 results/results.json")


if __name__ == "__main__":
    main()
