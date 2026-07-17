# SciHarness

科学推理 Agent 评测框架（MVP阶段）。用一个可离线跑通、可复现的最小实现，
量化 Agent Loop（Planning → Tool Call → Observation → Reflection）相对于
单次直接问答（baseline）在科学推理任务上的提升，并对失效模式做归因。


## 当前进度（诚实说明，2026-07-17更新）

这是一个正在迭代的项目，当前 MVP 已经跑通并用真实DeepSeek API验证过：

- ✅ 完整 Agent Loop：Planning → Tool Call → Observation → Reflection
- ✅ 两个工具：安全计算器 + 基于 TF-IDF 的本地知识库检索
- ✅ 分层记忆：短期对话窗口 + 基于文件的长期经验存储
- ✅ 评测题库：22题，覆盖热力学 / 量子力学 / 物态方程(EOS) 三个学科
- ✅ LLM-as-Judge 判分：与规则判分做一致性对比，发现规则判分对语义等价答案的
  漏判率约27%（22题中6题），验证了LLM-as-Judge判分的必要性
- ✅ 真实评测结果：Baseline成功率68.2% → Agent成功率95.5%（LLM-as-Judge判分），
  相对提升约40%；分学科看，EOS学科提升最明显（2/7→6/7）
- ✅ MCP Server/Client：用官方mcp SDK把calculator和knowledge_search包装成标准MCP协议工具
  （mcp_server.py + mcp_client_demo.py），Client端通过标准协议做工具发现（list_tools）和
  调用（call_tool），不依赖硬编码的工具列表

发现的问题（如实记录，不回避）：

- ⚠️ Token消耗：Agent因多轮LLM调用，总token消耗约为Baseline的27倍，
  不是token降低，这与直觉相反但符合预期——多步推理必然带来更多调用
- ⚠️ 发现1例因大模型输出随机性（temperature=0.2）导致的多步推导未收敛案例，
  同一问题重复运行3次，步数分别为3/4/6步，说明关键决策步骤的temperature设置
  还有优化空间，是下一步要做的事

还在做、尚未完成的部分：

- ⏳ 题库规模：目标扩到100+题
- ⏳ Planning/Reflection步骤的temperature调优，减少因随机性导致的失效
- ⏳ 检索：目前是TF-IDF，还没换成embedding检索

## 架构

```
sciharness/
  agent.py     Agent Loop 核心：plan -> act -> observe -> reflect
  llm.py       LLM调用封装（DeepSeekClient真实调用 / MockLLMClient离线mock）
  tools.py     工具定义：CalculatorTool, KnowledgeBaseTool
  memory.py    ShortTermMemory（对话窗口）+ LongTermMemory（跨任务经验）
  rag.py       SimpleRetriever：TF-IDF检索 + Context Compression（截断+top-k）
  config.py    全局配置

benchmark/
  questions.json        评测题库
  knowledge_base/       RAG检索用的本地知识库
  run_eval.py           评测入口：跑baseline vs agent对比
  results/results.json  评测结果（运行后生成）
```

## 快速开始

```bash
pip install -r requirements.txt

# 方式一：离线验证pipeline（不需要API key，用mock LLM）
python benchmark/run_eval.py --mock --limit 5

# 方式二：真实评测（需要DeepSeek API key）
export DEEPSEEK_API_KEY="sk-xxxx"
python benchmark/run_eval.py
```

运行后会在终端打印总体成功率、分学科成功率、baseline与agent的token消耗对比，
详细逐题结果写入 `benchmark/results/results.json`。

## 设计取舍说明

- **为什么不用LangChain封装的Agent**：评测/失效分析场景需要每一步的thought/action/observation
  都可追踪、可解释，自己写Loop虽然多花时间，但排查"为什么这道题失败了"时更直接。
- **为什么先用TF-IDF而不是embedding检索**：先用最简单能跑通、可解释、零外部依赖的方案把
  pipeline打通，接口设计上predict/search是独立的，换成embedding检索不需要改Agent代码。
- **为什么判分先用规则而不是LLM-as-Judge**：规则判分完全离线可复现，作为第一版baseline；
  LLM-as-Judge本身需要先验证一致性和偏见控制，是下一步要做的事，不能跳过验证直接上线。


## Token优化方向（已知问题，还没实现）

Agent相对Baseline的token消耗高出约27倍，这是多步推理的必然代价，但下面这几个具体优化点目前都还没做：

1. **减少不必要的步数**：system prompt可以更明确引导模型"能一步到位就别拆多步"，减少无意义的中间确认
2. **合并Planning和Reflection**：目前是两次独立的LLM调用，理论上可以合并成一次prompt完成"给答案+自检"
3. **更激进的上下文压缩**：短期记忆目前把完整thought文本堆进prompt，可以只保留关键结论，而不是完整推理过程
4. **分层用模型**：Reflection这种简单判断任务可以用更便宜的小模型甚至规则处理，只有Planning用大模型
5. **有选择地跳过Reflection**：简单知识型问题可以直接采信答案，只有复杂题目或模型置信度低时才触发反思

## Roadmap

1. 题库扩到100+题
2. RAG换成embedding检索，对比TF-IDF的召回质量差异
3. 实现Subagent分工（比如一个专门做数值计算的子agent + 一个专门做文献检索的子agent）
4. 补充pytest单元测试，接入CI
