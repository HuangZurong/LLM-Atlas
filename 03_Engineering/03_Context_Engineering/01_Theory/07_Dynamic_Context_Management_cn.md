# 07 · 动态上下文管理

*前置要求：[05_Token_Budget_and_Cost_cn.md](05_Token_Budget_and_Cost_cn.md) | [06_Long_Context_Techniques_cn.md](06_Long_Context_Techniques_cn.md)。*
*在 CE 流水线中的位置：第 3 步（Compress & Degrade）和第 4 步（Assemble & Observe）*

---

动态上下文管理（Dynamic Context Management）是运行时编排层，负责在一次对话或代理会话的生命周期内组装、更新并维护上下文窗口。

## 1. 将上下文视为状态机

把上下文窗口看作一个会随时间演化的**受管状态（managed state）**，而不是一个静态 prompt：

```
Session Start
     │
     ▼
[Initialize]  加载 system prompt、用户画像、session 配置
     │
     ▼
[Assemble]    每一轮：检索 memory + RAG + 格式化历史
     │
     ▼
[Execute]     使用组装好的上下文调用 LLM
     │
     ▼
[Update]      存储新一轮内容、抽取实体、更新记忆
     │
     ▼
[Compress?]   若上下文 > 阈值：进行摘要 / 修剪
     │
     └──────────────────────────────────────────────────────→ Next Turn
```

## 2. 逐轮演化的上下文

### 单轮（无状态）

每个请求彼此独立。上下文每次都从零开始重新组装：

```python
context = build_context(
    system_prompt=SYSTEM_PROMPT,
    rag_results=retrieve(query),
    user_query=query,
)
response = llm.call(context)
```

### 多轮（有状态）

上下文会跨轮次不断累积。关键挑战在于控制其增长：

```
Turn 1:  [System][Query1][Response1]                          ~1K
Turn 2:  [System][Query1][Response1][Query2][Response2]       ~2K
Turn N:  [System][Q1][R1]...[QN-1][RN-1][QN]                 ~NK
```

到某个时刻，累积的历史会超过预算。上下文管理器必须介入。

## 3. 上下文压缩触发器

定义明确的阈值，用于触发压缩：

```python
COMPRESSION_THRESHOLDS = {
    "soft_limit": 0.70,   # 开始压缩旧历史
    "hard_limit": 0.85,   # 激进压缩：摘要化所有内容
    "emergency":  0.95,   # 立即丢弃非必要内容
}

def should_compress(current_tokens: int, max_tokens: int) -> str:
    ratio = current_tokens / max_tokens
    if ratio >= COMPRESSION_THRESHOLDS["emergency"]:
        return "emergency"
    elif ratio >= COMPRESSION_THRESHOLDS["hard_limit"]:
        return "hard"
    elif ratio >= COMPRESSION_THRESHOLDS["soft_limit"]:
        return "soft"
    return "none"
```

## 4. 不同会话类型的上下文窗口策略

| 会话类型 | 历史策略 | 记忆策略 | 典型预算 |
| :--- | :--- | :--- | :--- |
| **单次 Q&A** | 无 | 无 | 4–8K |
| **短聊天** | 完整逐字保留 | 无 | 16–32K |
| **长对话** | 最近内容逐字保留 + 旧内容摘要化 | 实体抽取 | 32–64K |
| **文档分析** | 极少历史 | 无 | 32–128K |
| **代理任务** | Scratchpad + checkpoints | 任务状态 | 16–64K |
| **多会话** | 仅摘要 | Vector DB | 每个会话 8–16K |

## 5. 代理上下文管理

代理之所以特殊，是因为它们会执行多步计划并调用工具，这会带来独特的上下文挑战。

### 草稿板模式（Scratchpad Pattern）

维护一个结构化 scratchpad，把推理与结果分开：

```
[System Prompt]
[Task Definition]
[Tool Schemas]
─────────────────────────────────────────
SCRATCHPAD:
Thought: I need to find the user's order history first.
Action: search_orders(user_id="u123")
Result: [Order #1001, #1002, #1003]

Thought: Now I need to check the status of #1001.
Action: get_order_status(order_id="1001")
Result: {"status": "shipped", "eta": "2025-03-15"}
─────────────────────────────────────────
[Current Step]
```

### 草稿板摘要化

当 scratchpad 过大时，要压缩已完成的推理链：

```
BEFORE（详细）:
  Thought: I need X. Action: tool_a(). Result: {...full JSON...}
  Thought: Based on X, I need Y. Action: tool_b(). Result: {...full JSON...}
  Thought: X and Y together mean Z. Action: tool_c(). Result: {...full JSON...}

AFTER（压缩）:
  COMPLETED: Retrieved X via tool_a, Y via tool_b, Z via tool_c.
  KEY FACTS: [fact1, fact2, fact3]
  CURRENT STATE: Ready to synthesize final answer.
```

### 上下文检查点

对于特别长的代理运行（>30 步），应定期做 checkpoint：

1. 把已完成步骤摘要为一个结构化状态对象。
2. 将该状态保存到外部存储（Redis、DB）。
3. 重置上下文窗口。
4. 只注入 checkpoint 摘要和当前任务。

```python
checkpoint = {
    "task": original_task,
    "completed_steps": summarize_scratchpad(scratchpad),
    "key_findings": extract_key_facts(tool_results),
    "current_subtask": current_step,
    "remaining_steps": plan[current_index:],
}
```

## 6. 模式驱动的状态跟踪

从“文本累积”进化到“结构化状态管理”，代表了上下文工程中的一次范式转移。与其把对话历史逐字追加，不如维护一个通过交互不断演化的**固定大小状态机**。

### 6.1 范式转移

| 传统方式 | Schema-Driven 方式 |
| :--- | :--- |
| 向 prompt 中追加历史 | 跟踪结构化状态对象 |
| O(N) 的上下文增长 | O(1) 的上下文复杂度 |
| 完整转录 | 压缩后的事实 |
| 存在 Lost in the Middle 风险 | 高信号、低噪声 |

**核心洞见**：自然语言是低效的状态存储格式。一个提取过事实的 JSON 对象，token 效率可能比等价的对话转录高 10–100 倍。

### 6.2 混合模式架构

生产系统会使用一种平衡代码稳定性和 LLM 灵活性的**混合 schema（hybrid schema）**：

```
┌─────────────────────────────────────────────────────────────┐
│                    User Session Schema                       │
├─────────────────────────────────────────────────────────────┤
│  core_state: CoreState          ← 强类型，代码拥有          │
│    ├── intent: str              ← 业务逻辑依赖此字段        │
│    ├── destination: str         ← 用于 API 调用            │
│    ├── budget: float | None     ← 必须严格匹配 schema      │
│    └── status: enum             ← 状态机控制               │
├─────────────────────────────────────────────────────────────┤
│  dynamic_profile: dict          ← 弱类型，LLM 拥有          │
│    ├── dietary_preference: str  ← 运行时发现               │
│    ├── allergies: list[str]     ← 用户主动提供的信息       │
│    └── pet_info: dict           ← 嵌套、临时结构           │
├─────────────────────────────────────────────────────────────┤
│  recent_context: str            ← 情绪/语气摘要            │
└─────────────────────────────────────────────────────────────┘
```

**设计原则**：
- **Core State**：由工程侧定义的固定 schema。直接映射到 API 调用、数据库查询和业务逻辑。LLM 只能*修改*已有字段，不能新增字段。
- **Dynamic Profile**：LLM 可扩展的自由字典。记录对话中发现的用户偏好、约束和上下文。
- **Recent Context**：对最近 2–3 轮做简短摘要，以保留情绪连续性（例如“用户把目的地从日本改成了泰国，看起来很兴奋”）。

### 6.3 分层模型路由

Schema 更新非常适合交给**更小、更便宜的模型**：

```
User Query
    │
    ├──► [Small Model Pipeline] (GPT-4o-mini, Claude Haiku)
    │    Input: [Current Schema] + [User Query]
    │    Output: Updated Schema (JSON diff)
    │    Cost: ~$0.0001 per update
    │
    ├──► [Code Layer]
    │    Parse core_state → Trigger APIs if fields complete
    │    Validate dynamic_profile → Merge, dedupe
    │
    └──► [Large Model Pipeline] (GPT-4o, Claude Sonnet)
         Input: [Minimal Schema] + [API Results] + [Query]
         Output: Final response
         Cost: ~$0.01 per response
```

**Schema 就是一层压缩层**：大模型看到的不是原始对话历史，而是蒸馏后的状态。

### 6.4 处理键膨胀（Key Proliferation）

当 LLM 可以自由向 `dynamic_profile` 添加键时，迟早会出现语义重复：

```json
// 问题：同一概念，不同键名
{
  "food_taboos": ["seafood"],
  "cannot_eat": ["shellfish"],
  "dietary_restrictions": ["no shrimp"]
}
```

**解决方案：Memory Consolidation**

运行后台整合流水线（由 CRON 触发，或每 N 轮触发一次）：

```python
CONSOLIDATION_PROMPT = """
You are a memory consolidation engine. Given a dynamic profile with potentially redundant keys:

1. Identify keys with overlapping semantics
2. Merge them into a single canonical key
3. Preserve all values (union)
4. Output a cleaned, deduplicated profile

Original: {dynamic_profile}
Consolidated:
"""
```

**解决方案：Tree-Structured Memory**

不要使用扁平字典，而是强制采用带预定义父节点的层级结构：

```python
DYNAMIC_PROFILE_SCHEMA = {
    "dietary_profile": {},      # 所有饮食相关偏好
    "travel_preferences": {},   # 所有出行相关偏好
    "personal_info": {},        # 所有个人信息
}
# LLM 可以加键，但只能加在这些父节点之下
```

这模仿了认知 schema：“花粉过敏”和“海鲜过敏”都会被归入 `dietary_profile.allergies`。

### 6.5 实现模式

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from enum import Enum

class SessionStatus(str, Enum):
    EXPLORING = "exploring"
    COLLECTING = "collecting"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"

class CoreState(BaseModel):
    """业务逻辑使用的强类型状态。LLM 不能添加字段。"""
    intent: Optional[str] = Field(default=None, description="User's primary goal")
    destination: Optional[str] = Field(default=None, description="Target location")
    departure_date: Optional[str] = Field(default=None)
    budget: Optional[float] = Field(default=None)
    status: SessionStatus = Field(default=SessionStatus.EXPLORING)

class SessionSchema(BaseModel):
    """采用混合类型的完整会话状态。"""
    session_id: str
    core_state: CoreState                           # 代码拥有
    dynamic_profile: Dict[str, Any] = Field(default_factory=dict)  # LLM 拥有
    recent_summary: str = ""

    def is_ready_for_api(self) -> bool:
        """检查 core state 是否已具备调用外部 API 所需的信息。"""
        return all([
            self.core_state.destination,
            self.core_state.departure_date,
            self.core_state.budget is not None,
        ])

# 小模型的状态更新 prompt
UPDATE_STATE_PROMPT = """
You are a state management engine. Update the session schema based on the user's latest message.

Rules:
1. core_state: Only modify existing fields. Never add new keys.
2. dynamic_profile: You may add new keys to record user preferences (e.g., allergies, preferences).
3. If you detect duplicate semantics in dynamic_profile, merge them.
4. Update recent_summary with a 1-sentence emotional/contextual note.

Current Schema: {current_schema}
User Message: {user_message}

Output the updated schema as JSON:
"""
```

### 6.6 研究前沿

| 研究方向 | 关键论文 / 项目 | 核心思想 |
| :--- | :--- | :--- |
| **自演化记忆（Self-Evolving Memory）** | Evo-Memory (UIUC + Google DeepMind, 2025) | 测试时学习：代理在空闲时运行 `ReMem (Action-Think-Memory Refine)` 流水线 |
| **MemSkill** | MemSkill (2026) | 把记忆作为可学习技能；代理维护自己的技能库以演化 schema |
| **MemTree** | MemTree (2024) | 动态树结构模仿认知 schema；自动把相关概念归到父节点下 |
| **Ontology-Driven Memory** | Knowledge Graph Memory | 固定本体骨架 + 动态实例节点；适合多代理协作 |
| **LangMem** | LangChain LangMem | 提供 Profiles（强类型）+ Collections（弱类型）的生产框架 |

**关键趋势**：记忆正在变成“主动的” — 它不再只是存储，而是代理可以查询、重组和精炼的推理底座。

## 7. 多代理系统中的上下文隔离

当多个代理共享信息时，上下文边界必须是显式的：

```
Orchestrator Context:
  [System: Orchestrator role]
  [Task decomposition]
  [Sub-agent results: SUMMARY ONLY]   ← 不是完整的 sub-agent 上下文

Sub-agent Context:
  [System: Specialist role]
  [Assigned sub-task]
  [Relevant tools + data]
  [No access to other sub-agents' contexts]
```

**关键原则**：子代理的上下文彼此隔离。编排者只接收结构化摘要，而不接收原始子代理转录。这样可以防止上下文污染，并保持每个代理的窗口聚焦。

## 8. 上下文可观测性

为上下文管理打点，以便调试和优化：

```python
# 为每次 LLM 调用记录上下文构成
context_log = {
    "timestamp": ...,
    "session_id": ...,
    "total_tokens": ...,
    "layer_breakdown": {
        "system": 500,
        "memory": 1800,
        "rag": 3200,
        "history": 2900,
        "query": 180,
        "output_reserve": 4420,
    },
    "compression_applied": "soft",
    "cache_hit": True,
}
```

持续跟踪这些指标，可以发现上下文膨胀、低效检索和可压缩机会。

---

## 关键参考文献

1. **Anthropic. (2025). Effective Context Engineering for AI Agents.** Anthropic Engineering Blog.
2. **Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models.** *ICLR 2023*.
3. **Wang, L., et al. (2024). A Survey on Large Language Model based Autonomous Agents.** *Frontiers of Computer Science*.
4. **Evo-Memory. (2025). Self-Evolving Memory for LLM Agents.** UIUC + Google DeepMind.
5. **MemTree. (2024). Dynamic Tree-Structured Memory for Conversational Agents.**
6. **LangChain. (2025). LangMem: Semantic Memory for AI Agents.** LangChain Documentation.
