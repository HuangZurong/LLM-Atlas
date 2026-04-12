# 06 · Google 上下文工程实践

*整理自 Google ADK 源码、Gemini API 文档、Vertex AI 文档、Google I/O 2025 发布内容以及 ADK 示例代理。*

---

## 1. 核心理念

### 1.1 以状态作为上下文总线

Google ADK 的核心设计原则是：**结构化状态传递优于对话历史传递**。与其依赖原始消息线程在代理之间搬运信息，ADK 更倾向于把共享的 `session.state` 字典作为主要的上下文传输机制。

### 1.2 把长上下文本身当作特性

与 Anthropic 和 OpenAI 强调尽量压缩上下文不同，Google 更强调 Gemini 的 1M+ token 上下文窗口是一种差异化优势。他们的实践往往建议把整份文档或整个代码库直接送入上下文，而不是强依赖分块式 RAG，但前提是必须配合谨慎的摆放策略。

---

## 2. ADK 的上下文架构

### 2.1 上下文对象层级

ADK 实现了一个分层上下文系统，共有四种不同对象，每种对象的访问能力不同：

```
ReadonlyContext（基础）
  └── CallbackContext（增加可变 state、artifacts、credentials）
        └── ToolContext（增加 function_call_id、auth、search_memory）

InvocationContext（单次调用的主上下文）
  └── 对 session、artifact、memory、credential 等服务的全部引用
```

| Context 对象 | 访问级别 | 核心能力 |
|---------------|-------------|------------------|
| `ReadonlyContext` | 只读 | `user_content`、`invocation_id`、`agent_name`、`state`（不可变 `MappingProxyType`）、`session` |
| `CallbackContext` | 可读写 | 可变 state（通过 delta tracking）、artifact 的 load/save/list、credential 管理 |
| `ToolContext` | 可读写 + tools | `function_call_id`、auth request/response、tool confirmation、`search_memory()` |
| `InvocationContext` | 完整 | 所有服务、agent states、branching 信息、cost tracking、resumability config |

### 2.2 三前缀状态约定

`State` 类实现了一个带 delta 跟踪的 dict，并通过三个关键前缀定义可见域：

```python
class State:
    APP_PREFIX = "app:"    # 整个应用的所有 session 共享
    USER_PREFIX = "user:"  # 某个用户的所有 session 共享
    TEMP_PREFIX = "temp:"  # 临时数据，不会跨 invocation 持久化
```

状态变化通过 `_delta` 字典跟踪。当你写入 `ctx.state['key'] = value` 时，实时值和 delta 都会同步更新。这个 delta 会在 invocation 结束时提交到存储。

**来自 ADK 样例的用法**：

```python
# FOMC Research agent — 显式的 state storage tool
def store_state_tool(state: dict, tool_context: ToolContext) -> dict:
    tool_context.state.update(state)
    return {"status": "ok"}

# Deep Search agent — 在迭代中累积 sources
callback_context.state["url_to_short_id"] = url_to_short_id
callback_context.state["sources"] = sources

# Customer Service agent — 预先把 customer profile 加载到 session state
# （在对话开始前加载，agent 从 state 中读取）
```

---

## 3. `output_key` 模式：代理间上下文传递

这是 Google 在多代理系统中最核心的上下文传递机制。

### 3.1 工作方式

当某个 agent 设置了 `output_key="some_key"` 后，它的最终文本响应会被自动写入 `session.state["some_key"]`。下游代理随后就可以在指令中通过 `{some_key}` 模板语法引用该值。

来自 LlmAgent 源码：

```python
if self.output_key and event.is_final_response() and event.content and event.content.parts:
    result = ''.join(part.text for part in event.content.parts if part.text and not part.thought)
    if self.output_schema:
        result = self.output_schema.model_validate_json(result).model_dump(exclude_none=True)
    event.actions.state_delta[self.output_key] = result
```

### 3.2 真实例子

**Story Teller agent** — 协同写作状态机：

```python
prompt_enhancer:   output_key = "enhanced_prompt"
creative_writer:   output_key = "creative_chapter_candidate"
focused_writer:    output_key = "focused_chapter_candidate"
critique_agent:    output_key = "current_story"
editor_agent:      output_key = "final_story"
```

**Deep Search agent** — 通过 state 串联研究过程：

```python
section_planner:          output_key = "report_sections"
section_researcher:       output_key = "section_research_findings"
research_evaluator:       output_key = "research_evaluation"
enhanced_search_executor: output_key = "section_research_findings"  # Overwrites!
report_composer:          output_key = "final_cited_report"
```

**Parallel Task Decomposition** — 扇出 / 扇入：

```
message_enhancer → output_key = "enhanced_message"
  ↓（通过 state 流入三个并行分支）
email_drafter    → output_key = "drafted_email"
slack_drafter    → output_key = "drafted_slack_message"
event_extractor  → output_key = "event_details"
  ↓（所有结果都可被 summary_agent 读取）
summary_agent 读取全部 output_keys
```

---

## 4. 上下文过滤与窗口化

### 4.1 上下文过滤插件（ContextFilterPlugin）

`ContextFilterPlugin` 提供两种管理上下文窗口大小的机制：

```python
class ContextFilterPlugin(BasePlugin):
    def __init__(self, num_invocations_to_keep=None, custom_filter=None):
```

- `num_invocations_to_keep`：仅保留最近 N 次 invocation（用户消息 + 模型回复对），裁掉更老历史
- `custom_filter`：一个 `(List[Event]) -> List[Event]` 回调，用于实现任意过滤逻辑

它会作为 `before_model_callback` 运行，在请求到达模型前修改 `llm_request.contents`。

### 4.2 `include_contents`：核选项

LlmAgent 上的 `include_contents` 参数：

```python
include_contents: Literal['default', 'none'] = 'default'
# 'none': 模型完全不接收历史内容，只依赖
#         当前 instruction 和 input
```

**这是任何框架里最干净的上下文隔离模式之一。** Deep Search agent 在 report_composer 上就采用了它：

```python
report_composer = LlmAgent(
    include_contents="none",  # 不传任何对话历史
    instruction="""
    Research Plan: {research_plan}
    Research Findings: {section_research_findings}
    Citation Sources: {sources}
    Report Structure: {report_sections}
    """,
)
```

也就是说，对话历史被彻底剥离，只通过 state 模板变量注入结构化数据。

### 4.3 通过 Branching 实现上下文隔离

`InvocationContext.branch` 字段可在子代理之间实现上下文隔离：

```python
branch: Optional[str] = None
# 格式：agent_1.agent_2.agent_3
# 用于多个子代理不应看到彼此对话历史的场景
```

系统可通过 `_get_events(current_branch=True)` 按 branch 过滤事件，从而避免并行子代理相互污染上下文窗口。

---

## 5. 上下文缓存（Gemini 特有）

### 5.1 配置

```python
class ContextCacheConfig(BaseModel):
    cache_intervals: int = 10    # 最多复用同一缓存的 invocation 次数
    ttl_seconds: int = 1800      # 30 分钟 TTL
    min_tokens: int = 0          # 触发缓存的最小 token 数
```

### 5.2 缓存策略

1. **第一次请求**：生成 fingerprint（system instruction + tools + 前 N 个 contents 的哈希），但不立即创建缓存
2. **第二次请求**：如果 fingerprint 一致，就通过 `genai_client.aio.caches.create()` 创建 Gemini cached content 对象
3. **后续请求**：验证缓存是否有效（未过期、未超过 interval 限制、fingerprint 仍匹配）。若有效，则从请求中剥离 system instruction / tools / cached contents，并设置 `cached_content = cache_name`
4. **缓存边界**：通过寻找最后一批连续 user contents 来确定，把其之前的内容全部作为缓存前缀

对于 system instructions 和 tool definitions 很大、但又相对稳定的代理来说，这是一项非常可观的成本优化。

---

## 6. 记忆服务：长期上下文

ADK 提供三种 memory service 实现：

### 6.1 InMemoryMemoryService

用关键字匹配做原型。将 session events 存在内存里，通过词重叠进行搜索。

### 6.2 VertexAiRagMemoryService

基于 Vertex AI RAG corpus：
- 将 session events 作为 JSON lines 上传到 RAG corpus
- 用 `rag.retrieval_query()` 检索，并可配置 `similarity_top_k` 与 `vector_distance_threshold`

### 6.3 VertexAiMemoryBankService

基于 Vertex AI Memory Bank（托管服务）：
- 使用 `memories.generate()` 从 session events 中生成结构化 memories
- 使用 `memories.retrieve()` 做相似度搜索
- **它抽取的是对话中的“事实”**，而不是简单保存原始 events

### 6.4 工具层级的记忆访问

工具可这样查询 memory：

```python
results = tool_context.search_memory(query)
# 返回带 content、author 和 timestamp 的 MemoryEntry 对象
```

---

## 7. 事实锚定（Grounding）与上下文增强

### 7.1 从事实锚定到引文的流水线

来自 Deep Search agent 的完整 grounding 模式：

1. `google_search` 工具通过 `grounding_metadata` 提供网页 grounding
2. `grounding_chunks` 包含网页来源 URI、标题和域名
3. `grounding_supports` 包含带置信度的文本片段，并映射到 chunks
4. `collect_research_sources_callback` 把这些内容整合进 state 中一个结构化 citation 数据库
5. `citation_replacement_callback` 在最终报告生成后，把 `<cite source="src-N"/>` 替换为 markdown links

这是一个完全通过上下文工程回调搭建出来的 grounding-to-citation 流水线。

---

## 8. 多代理上下文编排

### 8.1 四种编排原语

每种原语都有不同的上下文流动语义：

| 原语 | 上下文流动 | 使用场景 |
|-----------|-------------|----------|
| **SequentialAgent** | 代理按顺序运行；每个代理都能看到前面所有代理累积的 state | 需要逐步累积上下文的流水线 |
| **ParallelAgent** | 代理并发运行；每个代理得到当前 state 的快照；输出再被合并 | 可并行完成的独立任务 |
| **LoopAgent** | 最多运行 `max_iterations` 轮；state 在循环中累积 | 生成-评估-改进式循环 |
| **LLM-based transfer** | LlmAgent 根据对话动态转交给子代理或同级代理 | 需要按用户意图动态路由 |

### 8.2 转移控制权

通过两个参数控制：
- `disallow_transfer_to_parent`：阻止代理向上级回传控制权
- `disallow_transfer_to_peers`：阻止代理转给同级代理

---

## 9. Gemini 长上下文最佳实践

### 9.1 摆放策略

| 策略 | 说明 |
|----------|-------------|
| 开头和结尾都放指令 | 缓解 “lost in the middle” 效应 |
| 原生多模态上下文 | 直接输入 PDF、图片、音频、视频，而不是先提取文本 |
| 整文 in-context | 对检索类任务，Gemini 往往可以直接在上下文中处理整份文档，而非 chunked RAG，并常得到更高准确率 |
| Many-shot prompting | 利用长上下文容纳数十到数百个示例 |
| 整个代码库进上下文 | 一次性喂入完整代码库，而非逐文件喂入 |

### 9.2 用上下文缓存控制成本

对于 1M+ token 的上下文，缓存变得至关重要。使用 `ContextCacheConfig`（第 5 节）来避免重复发送巨大的稳定前缀。

---

## 10. Google I/O 2025 及近期发展

| 发布内容 | 说明 |
|-------------|-------------|
| Gemini 2.5 Pro | 1M token 上下文，并增强了 “needle in a haystack” 表现 |
| Agent-to-Agent（A2A）协议 | 跨框架的代理通信标准 |
| ADK 开源发布 | 官方代理框架，包含本文所述的上下文模式 |
| Vertex AI Agent Engine | 带内置 session 和 memory 管理的托管部署平台 |
| Memory Bank | 面向长期代理记忆的托管服务（VertexAiMemoryBankService） |

---

## 11. 关键上下文工程模式总结

| 模式 | 说明 | ADK 机制 |
|---------|-------------|---------------|
| **State as context bus** | 用 `output_key` 和 `state[]` 在代理间传递结构化数据 | `output_key`、`{template}` 语法 |
| **Context isolation** | 剥离历史，只通过 state templates 注入必需数据 | `include_contents="none"` |
| **Branching** | 按 branch 过滤 events，防止上下文污染 | `InvocationContext.branch` |
| **Context windowing** | 用 invocation 限制或自定义过滤器管理上下文大小 | `ContextFilterPlugin` |
| **Context caching** | 在多次 invocation 之间缓存稳定部分 | `ContextCacheConfig` |
| **Memory services** | 跨 session 持久存在的长期上下文 | RAG 或 Memory Bank |
| **Callback-based transformation** | 在模型调用前后变换上下文 | `before_model_callback`、`after_agent_callback` |
| **Grounding pipeline** | Google Search grounding + 结构化来源跟踪 | Grounding metadata + callbacks |
| **Delta-tracked state** | 所有 state 变更都以 delta 形式跟踪，提高持久化效率 | `State._delta` |
| **Three-scope prefixes** | 控制 state 可见性与持久化范围 | `app:`、`user:`、`temp:` |

---

## 参考资料

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Google ADK Samples Repository](https://github.com/google/adk-samples)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Vertex AI Agent Engine](https://cloud.google.com/vertex-ai/docs/agents)
- [Gemini Long Context Guide](https://ai.google.dev/gemini-api/docs/long-context)
- [Google I/O 2025 Developer Keynote](https://io.google/2025/)
- ADK source code: `google.adk.agents`, `google.adk.sessions.state`, `google.adk.plugins.context_filter_plugin`, `google.adk.models.gemini_context_cache_manager`
