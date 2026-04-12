# 01 · Context Engineering 简介

*前置要求：[../../02_Prompt_Engineering/README.md](../../02_Prompt_Engineering/README.md)。*
*在 CE 流水线中的位置：作为 `Load -> Budget -> Compress -> Assemble -> Observe` 全流程之前的导论。*

---

## 1. 什么是 Context Engineering

Anthropic 在 2025 年 9 月 29 日发布的工程文章中，将 Context Engineering 视为 Prompt Engineering 的自然延伸。它强调的不是只写好提示词，而是在 LLM 推理过程中，持续整理并维护一组最优 token，包括那些并不直接写在 prompt 里的信息，例如工具、记忆、外部数据和消息历史。

用更工程化的话说：

- **Prompt Engineering** 关心的是：指令应该怎么写。
- **Context Engineering** 关心的是：模型此刻应该看到什么信息，这些信息按什么顺序进入窗口，以什么优先级存在，又该以什么形式呈现。

本模块采用这个更宽的定义。真实的生产级 LLM 系统，很少只依赖一段 prompt。它实际处理的是一个被动态组装出来的上下文，其中通常包括：

- 系统指令
- 当前任务规则
- 对话历史
- 检索到的记忆
- RAG 结果
- 工具定义
- 工具输出
- 运行时状态

Context Engineering 研究的，就是这些组成部分如何在有限上下文窗口中竞争、协作和演化。

---

## 2. 为什么它重要

当系统从单轮问答走向多轮对话、长链路任务和工具代理时，核心工程问题会发生变化。

问题不再只是：

- 如何写出更好的 system prompt

而是：

- 如何避免上下文窗口被低信号 token 填满
- 如何把当前步骤最相关的信息保留下来
- 如何控制延迟和 token 成本
- 如何避免 “Lost in the Middle” 这类位置效应问题
- 如何在多轮交互中维持状态一致性

更大的上下文窗口确实提供了更多空间，但并没有消除问题。token 变多，意味着信息机会变多，也意味着噪声、注意力稀释和成本同步增长。因此，上下文不应被当作“无限文本缓冲区”，而应被视作**有限且昂贵的工作内存**。

---

## 3. 典型使用场景

当任务依赖以下任一因素时，你通常就需要认真做 Context Engineering：

- 多轮连续性
- 外部检索
- 长文档处理
- 工具调用
- 持久化的用户或任务状态
- 多模态输入

如果你想系统理解：为什么问答、客服、文档分析、编码、Agent、个性化和多模态任务需要不同的 CE 策略，请直接看 [02_Context_Strategies_by_Scenario](./02_Context_Strategies_by_Scenario_cn.md)。

---

## 4. 什么时候不需要很重的 CE

如果你的任务具备以下特点，通常不需要一整套复杂的 Context Engineering 层：

- 单轮且很短
- 高度模板化
- 指令固定，外部上下文很少
- 只是做简单抽取、分类或格式化

在这些场景里，严谨的 Prompt Engineering 往往已经足够。只有当运行时状态、外部检索、长历史或工具交互开始不断积累时，Context Engineering 的价值才会明显上升。

---

## 5. CE 主要解决什么问题

本模块把 Context Engineering 视为对五类高频生产问题的回应：

### 5.1 Selection：选什么

潜在相关信息总是比窗口能容纳的更多。

核心问题：

- 这一轮调用里，究竟什么应该进上下文？

### 5.2 Ordering：怎么排

信息摆放位置会影响模型行为。

核心问题：

- 哪些内容应该放在开头、中间和结尾？

### 5.3 Budgeting：怎么分预算

输入和输出共享有限 token 预算。

核心问题：

- 指令、记忆、RAG、历史、工具结果和输出预留各应该占多少？

### 5.4 Compression：超预算时怎么办

长会话或长任务迟早会超预算。

核心问题：

- 哪些内容应该总结、截断、结构化，或者直接丢弃？

### 5.5 Observability：怎么证明策略有效

没有观测，就只能靠感觉调 CE。

核心问题：

- 如何衡量利用率、压缩质量、成本和上下文策略效果？

---

## 6. CE 与相邻主题的边界

Context Engineering 和周边模块有交叉，但它不是它们的简单别名。

| 主题 | 主要问题 | 对应模块 |
| :--- | :--- | :--- |
| Prompt Engineering | 指令应该怎么写？ | [../../02_Prompt_Engineering](../../02_Prompt_Engineering) |
| Context Engineering | 推理这一步到底该让模型看到什么？ | 本模块 |
| Memory | 信息如何跨会话持久化？ | [../../04_Memory](../../04_Memory) |
| RAG | 外部信息应该如何被检索回来？ | [../../05_RAG](../../05_RAG) |
| Agent | 工具、计划和行动应该如何编排？ | [../../06_Agent](../../06_Agent) |

一个很有用的速记规则是：

- **RAG 负责取回信息**
- **Memory 负责长期保留信息**
- **Prompting 负责写清指令**
- **Agents 负责执行动作**
- **Context Engineering 负责决定什么真正进入模型当前的工作集**

---

## 7. 一个最小心智模型

对于本仓库，一个足够实用的默认模型是：

1. **Load**：从 prompt、memory、retrieval、history、tools 中收集候选上下文。
2. **Budget**：计算可用窗口并预留输出空间。
3. **Compress / Degrade**：在预算不足时压缩或降级低优先级内容。
4. **Assemble**：按照有利于模型注意力的顺序组装最终上下文。
5. **Observe**：记录 token 使用、成本、裁剪决策和结果质量。

如果只记住一句话，那就是：

> Context Engineering 本质上是在约束条件下做运行时信息分配。

### 7.1 一个务实的实现原则

在真实系统里，大多数团队并不会一开始就做出一个完美的意图识别器、完整的路由图，或者一整套归档激活机制。

更现实的工程路径通常是：

1. 先采用一个简单的默认上下文策略
2. 在评估或生产中观察这个默认策略究竟在哪里持续失效
3. 只有在这些失败反复出现时，才引入检索、归档召回、前置路由或结构化记忆

这很重要，因为“决策层”本身也是一个系统，它同样有成本、延迟和自己的失败模式。如果引入得太早，系统可能还没解决原问题，就先背上了额外复杂度。

因此，更符合实际的顺序通常是：

- 先有默认方案
- 再观察失败
- 最后定点升级

换句话说：

> 不要一开始就追求最大复杂度，而应先用最简单可行的策略，再在现实问题逼迫下逐步升级。

---

## 8. 本模块会教你什么

本模块按“定义 -> 机制 -> 生产模式”的顺序展开：

- [02_Context_Strategies_by_Scenario](./02_Context_Strategies_by_Scenario_cn.md)：不同任务下 CE 策略如何变化
- [03_Context_Window_Mechanics](./03_Context_Window_Mechanics_cn.md)：把上下文窗口理解为受约束的工作内存
- [04_Context_Composition](./04_Context_Composition_cn.md)：上下文分层、排序与优先级
- [05_Token_Budget_and_Cost](./05_Token_Budget_and_Cost_cn.md)：token 分配与压缩策略
- [06_Long_Context_Techniques](./06_Long_Context_Techniques_cn.md)：chunking、map-reduce、摘要树
- [07_Dynamic_Context_Management](./07_Dynamic_Context_Management_cn.md)：多轮演化与 schema-driven state tracking
- [08_Advanced_Context_Paradigms](./08_Advanced_Context_Paradigms_cn.md)：更进阶的压缩与编排范式
- [09_CE_Evaluation](./09_CE_Evaluation_cn.md)：如何验证 CE 策略真的有效

接下来的 Practical 层，会把这些概念落成共享组件和案例实现。

---

## 9. 推荐下一步

读完本导论后，建议继续看 [02_Context_Strategies_by_Scenario](./02_Context_Strategies_by_Scenario_cn.md)，再看 [03_Context_Window_Mechanics](./03_Context_Window_Mechanics_cn.md)。

那篇文档会建立后续所有 CE 决策的物理约束基础：有限预算、KV cache 成本、prefix caching、位置效应，以及生产级四步流水线。

---

## 参考来源

- Anthropic Engineering. [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Published September 29, 2025.
