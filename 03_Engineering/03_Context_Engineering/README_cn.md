# 03 · 上下文工程

*前置要求：[02_Prompt_Engineering](../02_Prompt_Engineering)。
在开始管理“什么内容应该进入上下文窗口”之前，应先理解如何写出有效的提示词。*

---

上下文工程（Context Engineering）是一门研究**上下文窗口里应该放什么、按什么顺序放、以及各部分应占据什么优先级**的工程实践。提示词工程关注的是如何措辞，而上下文工程关注的是整个上下文的**组成方式与生命周期**。

在生产环境中，上下文往往由多个来源动态拼装而成，例如系统提示词、检索到的记忆、RAG 结果、对话历史、工具输出等，它们都在争夺有限的 token 预算。把这件事处理好，往往决定了系统是稳定可靠，还是会出现幻觉、截断和成本失控。

---

## 模块结构

```text
03_Context_Engineering/
|-- 01_Theory/          概念与思维模型
|-- 02_Practical/       可运行实现
`-- 03_Best_Practice/   生产模式与决策框架
```

---

## 01 理论

| 文件 | 主题 | 前置要求 |
| :--- | :--- | :--- |
| [01_Introduction](01_Theory/01_Introduction_cn.md) | **[导论]** CE 的定义、为什么重要、典型场景、范围边界、最小心智模型 | Prompt Engineering |
| [02_Context_Strategies_by_Scenario](01_Theory/02_Context_Strategies_by_Scenario_cn.md) | 面向问答、客服、文档分析、编码、代理、个性化与多模态场景的 CE 策略 | Introduction |
| [03_Context_Window_Mechanics](01_Theory/03_Context_Window_Mechanics_cn.md) | **[基础]** “一个核心、两个轴”的思维模型 + KV 缓存成本、前缀缓存、中间遗失（Lost in the Middle） | Prompt Engineering 01 |
| [04_Context_Composition](01_Theory/04_Context_Composition_cn.md) | 7 层上下文解剖、三明治模式、优先级层级、多模态成本 | Theory 03 |
| [05_Token_Budget_and_Cost](01_Theory/05_Token_Budget_and_Cost_cn.md) | 预算模型、基于优先级的分配、压缩策略、成本优化 | Theory 04 |
| [06_Long_Context_Techniques](01_Theory/06_Long_Context_Techniques_cn.md) | NIAH 测试、分块策略、map-reduce、摘要树、位置感知放置 | Theory 04 |
| [07_Dynamic_Context_Management](01_Theory/07_Dynamic_Context_Management_cn.md) | 将上下文视为状态机、Schema-Driven State Tracking、混合 schema、分层模型路由、记忆整合 | Theory 05 + 06 |
| [08_Advanced_Context_Paradigms](01_Theory/08_Advanced_Context_Paradigms_cn.md) | 信息论压缩、结构/模态感知预算、多代理上下文编排、进阶评估 | Theory 05 + 07 |
| [09_CE_Evaluation](01_Theory/09_CE_Evaluation_cn.md) | 评估指标、NIAH 变体、上下文相关性评分、基准方法论 | Theory 06 + 08 |

---

## 02 实践

### 共享库

供所有案例研究复用的基础组件。

| 文件 | 实现内容 | 前置要求 |
| :--- | :--- | :--- |
| [shared/composer.py](02_Practical/shared/composer.py) | `ContextLayer`、`ContextComposer`、基于优先级的裁剪、三明治模式 | Theory 04, 05 |
| [shared/budget_controller.py](02_Practical/shared/budget_controller.py) | `TokenBudgetController`、按层分配、压缩触发器 | Theory 05 |
| [shared/compressor.py](02_Practical/shared/compressor.py) | 5 种压缩策略：截断、滑动窗口、抽取式、抽象式、实体压缩 | Theory 05 |
| [shared/observability.py](02_Practical/shared/observability.py) | `ContextObserver`、成本归因、上下文 diff 日志、OpenTelemetry spans | Theory 03-05 |

### 案例研究

将共享库组合起来，用于解决真实问题的端到端场景。

| 目录 | 展示内容 | 前置要求 |
| :--- | :--- | :--- |
| [customer_support/](02_Practical/customer_support/) | 多轮上下文管理、模式驱动状态跟踪、压缩触发器 | Theory 07 + shared/ |
| [document_analysis/](02_Practical/document_analysis/) | 分块策略、map-reduce、层级摘要、位置感知组装 | Theory 06 + shared/ |

---

## 03 最佳实践

| 文件 | 主题 |
| :--- | :--- |
| [01_Context_Architecture_Patterns](03_Best_Practice/01_Context_Architecture_Patterns_cn.md) | 4 种架构模式、决策树、静态优先规则（Static-First Rule）、反模式、生产检查清单 |
| [02_Context_Quality_and_Evaluation](03_Best_Practice/02_Context_Quality_and_Evaluation_cn.md) | NIAH、上下文相关性评分、利用率、ROUGE-L、A/B 测试 |
| [03_Production_Context_Optimization](03_Best_Practice/03_Production_Context_Optimization_cn.md) | 异步组装、前缀缓存 ROI、分层模型、优雅降级、可观测性 |
| [04_Anthropic_CE_Practices](03_Best_Practice/04_Anthropic_CE_Practices_cn.md) | 最小高信号 token 集、Context Rot、Compaction、JIT 加载 |
| [05_OpenAI_CE_Practices](03_Best_Practice/05_OpenAI_CE_Practices_cn.md) | 四阶段记忆流水线、Compaction API、RunContextWrapper |
| [06_Google_CE_Practices](03_Best_Practice/06_Google_CE_Practices_cn.md) | State as Context Bus、`include_contents="none"`、delta 跟踪 |
| [07_Cursor_CE_Practices](03_Best_Practice/07_Cursor_CE_Practices_cn.md) | 分层检索、embedding recall + reranking、推测式编辑 |
| [08_Frameworks_CE_Practices](03_Best_Practice/08_Frameworks_CE_Practices_cn.md) | LangGraph reducers、CrewAI 实体记忆、MemGPT 自管理记忆 |
| [09_CE_Cross_Comparison](03_Best_Practice/09_CE_Cross_Comparison_cn.md) | 各家方案在理念、组成与性能数据上的对比 |
| [10_CE_Framework_Design](03_Best_Practice/10_CE_Framework_Design_cn.md) | 面向高度可配置、厂商无关 CE 框架的设计文档 |

---

## 推荐学习路径

建议先读 [01_Introduction](01_Theory/01_Introduction_cn.md)，再进入后续理论部分。

1. [01_Introduction](01_Theory/01_Introduction_cn.md)
2. [02_Context_Strategies_by_Scenario](01_Theory/02_Context_Strategies_by_Scenario_cn.md)
3. [03_Context_Window_Mechanics](01_Theory/03_Context_Window_Mechanics_cn.md)
4. [04_Context_Composition](01_Theory/04_Context_Composition_cn.md)
5. [05_Token_Budget_and_Cost](01_Theory/05_Token_Budget_and_Cost_cn.md)
6. 共享库：
   [shared/composer.py](02_Practical/shared/composer.py)、
   [shared/budget_controller.py](02_Practical/shared/budget_controller.py)、
   [shared/compressor.py](02_Practical/shared/compressor.py)、
   [shared/observability.py](02_Practical/shared/observability.py)
7. [06_Long_Context_Techniques](01_Theory/06_Long_Context_Techniques_cn.md) -> [document_analysis/](02_Practical/document_analysis/)
8. [07_Dynamic_Context_Management](01_Theory/07_Dynamic_Context_Management_cn.md) -> [customer_support/](02_Practical/customer_support/)
9. [08_Advanced_Context_Paradigms](01_Theory/08_Advanced_Context_Paradigms_cn.md)
10. 最佳实践 01 -> 02 -> 03
11. [09_CE_Evaluation](01_Theory/09_CE_Evaluation_cn.md)

---

## 范围边界

| 本模块涵盖 | 其他模块涵盖 |
| :--- | :--- |
| 什么内容进入上下文窗口 | 如何编写指令 -> [02_Prompt_Engineering](../02_Prompt_Engineering) |
| Token 预算与压缩 | 跨会话持久化 -> [04_Memory](../04_Memory) |
| 长上下文处理（窗口内） | 外部检索 -> [05_RAG](../05_RAG) |
| Agent 的上下文管理 | Agent 架构与工具使用 -> [06_Agent](../06_Agent) |
