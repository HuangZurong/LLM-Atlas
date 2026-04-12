# 02 · 按场景选择上下文策略

*前置要求：[04_Context_Composition_cn.md](04_Context_Composition_cn.md) | [05_Token_Budget_and_Cost_cn.md](05_Token_Budget_and_Cost_cn.md) | [07_Dynamic_Context_Management_cn.md](07_Dynamic_Context_Management_cn.md)。*

---

不存在一种对所有任务都最优的上下文策略。上下文工程始终取决于具体场景：
- 模型到底要完成什么任务
- 这个任务需要多强的上下文连续性
- 是否需要外部知识
- 工作流是单轮、多轮，还是代理式执行

真正该问的不是“什么 prompt 结构最好”，而是：

**在这个场景里，哪些上下文层应该成为主角，哪些应当弱化，它们又应该按什么顺序出现？**

---

## 1. 核心原则

不同场景优化的是不同瓶颈：

| 场景类型 | 主要瓶颈 | CE 关注点 |
| :--- | :--- | :--- |
| 单轮事实问答 | 检索精度 | RAG 质量、极简历史 |
| 多轮客服对话 | 状态连续性 | 最近历史 + 记忆 |
| 长文档分析 | 窗口上限 | 分块、分层、路由 |
| 编码助手 | 局部相关性 | 当前文件、依赖、近期修改 |
| 工具型 Agent | 上下文膨胀 | 草稿板控制、工具结果压缩 |
| 个性化助手 | 长期相关性 | 选择性记忆注入 |
| 多模态工作流 | 预算不对称 | 模态感知预算 |

这意味着 CE 不是一个统一模板，而是一组随任务而变的策略家族。

这些场景化策略背后的实现哲学，已经在 [01_Introduction](./01_Introduction_cn.md) 中说明：先采用最简单可行的默认策略，只有当默认策略在真实评估或生产中反复失效时，才逐步加入额外控制层。

---

## 2. 场景一：单轮事实问答

### 目标

以尽可能低的成本，尽可能准确地回答一个明确问题。

### 推荐的上下文形状

```text
[System Prompt]
[Top-1 或 Top-3 RAG Chunks]
[User Query]
```

### 应重点强调什么

- 强检索精度
- 极少历史
- 极少工具输出
- 把 query 放在末尾

### 应避免什么

- “以防万一”地注入长对话历史
- 除非直接相关，否则不加载长期用户记忆
- 放入过多弱相关的 RAG chunks

### 例子

用户问：“退款政策里的时限是多少？”

更合理的 CE 做法是：
- 一小段 system prompt
- 1 到 2 段最相关的政策原文
- 当前 query

通常不需要：
- 用户前 10 轮聊天记录
- 无关画像信息
- 原始工具日志

### 常见失败模式

prompt 变长了，但效果反而更差，因为无关上下文稀释了真正的答案片段。

---

## 3. 场景二：多轮客服对话

### 目标

在保持上下文连续性的同时，持续跟踪用户状态变化。

### 推荐的上下文形状

```text
[System Prompt]
[User / Ticket Memory]
[Recent Verbatim History]
[Relevant Tool Results]
[Current Query]
```

### 应重点强调什么

- 最近 2 到 4 轮逐字保留
- 稳定的用户或工单事实写成 memory
- 注入当前操作状态的工具结果
- 更早历史压缩为摘要

### 应避免什么

- 让原始对话历史无限增长
- 反复重复已经解决的问题
- 把所有历史都视为同等重要

### 例子

对话过程：
- “我的包裹还没到。”
- “订单号是 AC-7823。”
- “我现在不是想退款，是想改地址。”

更合理的 CE 做法是：
- 最近几轮原话保留
- 把订单号和当前意图写入 memory
- 只注入最新的物流状态

### 常见失败模式

模型沿用陈旧意图回答当前问题，因为 earlier turns 从未被提炼成状态。

---

## 4. 场景三：长文档分析

### 目标

处理一份接近或超过有效上下文窗口长度的长文档。

### 推荐的上下文形状

```text
[System Prompt]
[Section Summaries or Routing Layer]
[Most Relevant Detail Chunks]
[User Query]
```

### 应重点强调什么

- 语义分块
- 分层摘要
- 位置感知放置
- 对超长输入采用 map-reduce 或 refine

### 应避免什么

- 默认把 100 页文档整篇塞进一次调用
- 把所有章节视为同等相关
- 让最关键 chunk 掉进长提示词的中部

### 例子

用户问：“找出这份 120 页合同里的违约责任条款，并总结风险点。”

更合理的 CE 做法是：
- 先对章节做摘要
- 先路由到相关章节
- 再把违约责任条款放到最靠近 query 的位置

### 常见失败模式

相关条款明明在上下文里，却被埋在中部，结果模型没有真正抓住。

---

## 5. 场景四：编码助手

### 目标

帮助模型围绕代码库进行推理，同时保持上下文局部、聚焦、可执行。

### 推荐的上下文形状

```text
[System Prompt / Coding Rules]
[Current File + Cursor Region]
[Dependency or Call-Site Context]
[Recent Edits / Test Failures]
[User Task]
```

### 应重点强调什么

- 当前文件和局部区域优先级最高
- 按依赖关系检索相关代码
- 最近 diff 与测试失败信息
- 少而精的代码上下文

### 应避免什么

- 直接塞整个代码库
- 注入同仓库中无关文件
- 不压缩地塞入大段终端输出

### 例子

用户问：“把这个接口改成分页，并修掉 tests。”

更合理的 CE 做法是：
- 当前 handler
- service 层实现
- 对应测试
- 最近失败日志

### 常见失败模式

模型看到了太多代码，反而失去了对当前编辑目标的聚焦。

---

## 6. 场景五：工具型 Agent

### 目标

在多步执行中防止工具结果和中间推理把上下文淹没。

### 推荐的上下文形状

```text
[System Prompt]
[Task Definition]
[Tool Schemas]
[Condensed Scratchpad State]
[Latest Relevant Tool Outputs]
[Current Step]
```

### 应重点强调什么

- 草稿板摘要化
- 对工具返回结果设置硬上限
- 长任务中的 checkpoint
- 每步之后都修剪工具结果

### 应避免什么

- 无限追加每一次工具结果
- 把原始 JSON 或多页输出直接塞入上下文
- 把已经完成的推理链和当前工作状态混在一起

### 例子

研究型 agent 工作流：
- 搜网页
- 读报告
- 对比厂商
- 写结论

更合理的 CE 做法是：
- 只保留当前仍活跃的子问题状态
- 已解决步骤的工具结果做压缩
- 保留关键发现，而不是保留完整痕迹

### 常见失败模式

运行 20 到 30 步之后，系统性能迅速恶化，因为上下文膨胀本身成了主要故障源。

---

## 7. 场景六：个性化助手

### 目标

使用长期用户偏好回答问题，但又不把无关个人信息每轮都塞进去。

### 推荐的上下文形状

```text
[System Prompt]
[Relevant User Profile Memory]
[Recent History (if needed)]
[Current Query]
```

### 应重点强调什么

- 选择性注入用户记忆
- 使用稳定偏好，而不是原始历史转录
- 根据当前任务决定是否加载这段记忆

### 应避免什么

- 每次请求都注入完整用户画像
- 不区分时效和置信度地混入旧偏好
- 把个性化误当成 retrieval 的替代品

### 例子

相关偏好包括：
- 偏好简洁回答
- 居住在上海
- 对贝类过敏

这些可能会影响：
- 行程规划
- 餐厅推荐
- 商品推荐

但通常不会影响：
- 解释 Transformer attention
- 调试 Python 报错

### 常见失败模式

prompt 被无关的个性化信息撑大了，但这些信息与当前任务并无关系。

---

## 8. 场景七：多模态工作流

### 目标

在同一预算内平衡文本、图片、音频和结构化输入。

### 推荐的上下文形状

```text
[System Prompt]
[Critical Images / Audio Metadata]
[Compressed Text Context]
[Current Query]
```

### 应重点强调什么

- 分辨率控制
- 模态感知预算分配
- 在删文本之前先降低图像或音频保真度
- 用清晰的文本指令固定任务目标

### 应避免什么

- 所有图片都以最高精度发送
- 忽视不同模态的 token 成本
- 让多模态历史悄悄挤掉真正关键的文本说明

### 例子

用户上传 8 张截图，请你分析 UI 问题。

更合理的 CE 做法是：
- 只保留最有信息量的 2 到 3 张高精度图
- 其余图降级
- 附上一段简洁的问题描述

### 常见失败模式

高分辨率图片吃掉了大部分预算，结果真正说明任务目标的文字反而被压缩掉了。

---

## 9. 一个实用的选择矩阵

| 场景 | 最重的层 | 最轻的层 | 关键排序动作 |
| :--- | :--- | :--- | :--- |
| 事实问答 | RAG、Query | History、Memory | 把最强证据贴近 query |
| 客服对话 | Recent History、Memory、Query | Broad RAG | 最近几轮逐字保留 |
| 文档分析 | Summaries、Detail Chunks、Query | Full History | 先路由，再把关键 chunk 放近末尾 |
| 编码助手 | Current File、Dependencies、Task | 长对话历史 | 局部代码贴近任务 |
| 工具型 Agent | Scratchpad State、Latest Tool Outputs | 旧工具日志 | 激进压缩旧步骤 |
| 个性化助手 | User Profile Memory、Query | 无关历史 | 只加载任务相关偏好 |
| 多模态 | Critical Media + Text Query | 满精度全量媒体 | 把预算花在最有信息量的模态上 |

---

## 10. 元规则

最好的上下文策略，不是那个“塞进最多信息”的策略。

而是那个能够：
- 只放入最低限度的必要信息
- 把最重要的内容放到模型真正会关注的位置
- 只在确实需要时保留连续性
- 对其余内容做压缩或隔离

因此，真正的 CE 循环始终是：

**任务类型 -> 上下文选择 -> 预算分配 -> 位置感知组装 -> 评估**

---

## 关键参考文献

1. **Anthropic. (2025). Effective Context Engineering for AI Agents.** Anthropic Engineering Blog.
2. **Anthropic. (2025). Long Context Prompting Tips.** Anthropic Documentation.
3. **OpenAI. (2025). GPT-4.1 Prompting Guide.** OpenAI Cookbook.
4. **Google. (2025). Gemini Long Context Guide.** Google AI for Developers.
5. **LangChain. (2025). Context Engineering for Agents.** LangChain Blog.
