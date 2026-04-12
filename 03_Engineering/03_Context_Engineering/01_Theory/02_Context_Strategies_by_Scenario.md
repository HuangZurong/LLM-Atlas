# 02 · Context Strategies by Scenario

*Prerequisite: [04_Context_Composition.md](04_Context_Composition.md) | [05_Token_Budget_and_Cost.md](05_Token_Budget_and_Cost.md) | [07_Dynamic_Context_Management.md](07_Dynamic_Context_Management.md).*

---

There is no single "best" context strategy. Context engineering is always conditional on the task:
- what the model is being asked to do
- how much continuity the task requires
- whether external knowledge is needed
- whether the workflow is single-shot, multi-turn, or agentic

The practical question is not "What is the best prompt structure?" but:

**Given this scenario, which context layers should dominate, which should be minimized, and how should they be ordered?**

---

## 1. The Core Principle

Different scenarios optimize for different bottlenecks:

| Scenario Type | Primary Bottleneck | CE Focus |
| :--- | :--- | :--- |
| Single-turn factual Q&A | Retrieval precision | RAG quality, minimal history |
| Multi-turn support chat | State continuity | Recent history + memory |
| Long document analysis | Window limits | Chunking, hierarchy, routing |
| Coding assistant | Local relevance | Current file, dependencies, recent edits |
| Tool-using agent | Context growth | Scratchpad control, tool result compression |
| Personalized assistant | Long-term relevance | Selective memory injection |
| Multimodal workflow | Budget asymmetry | Modality-aware budgeting |

This means CE is not one template. It is a family of task-specific strategies.

The implementation philosophy behind these scenario-specific strategies is introduced in [01_Introduction](./01_Introduction.md): start with the simplest strategy that works, then add control layers only when repeated failures justify the added complexity.

---

## 2. Scenario 1: Single-Turn Factual Q&A

### Goal

Answer one narrow question as accurately and cheaply as possible.

### Recommended Context Shape

```text
[System Prompt]
[Top-1 or Top-3 RAG Chunks]
[User Query]
```

### What to Emphasize

- Strong retrieval precision
- Minimal conversation history
- Minimal tool output
- Query placed at the end

### What to Avoid

- Injecting long chat history "just in case"
- Loading persistent user memory unless it directly affects the answer
- Overloading the prompt with multiple weakly relevant RAG chunks

### Example

User asks: "What is the refund window in the policy?"

Best CE approach:
- short system prompt
- one or two policy passages
- current query

Not needed:
- the user's last 10 chat turns
- unrelated profile information
- raw tool logs

### Failure Mode

The prompt becomes larger but worse because irrelevant context dilutes the retrieved answer span.

---

## 3. Scenario 2: Multi-Turn Customer Support

### Goal

Maintain continuity while tracking evolving user state.

### Recommended Context Shape

```text
[System Prompt]
[User / Ticket Memory]
[Recent Verbatim History]
[Relevant Tool Results]
[Current Query]
```

### What to Emphasize

- Last 2-4 turns verbatim
- Stable user or ticket facts as memory
- Current operational status from tools
- Compression of older history

### What to Avoid

- Letting raw conversation history grow without summarization
- Repeating the same resolved issue across turns
- Treating all past dialogue as equally important

### Example

Conversation progression:
- "My package is late."
- "Order ID is AC-7823."
- "Actually, I want to change the address, not refund it."

Best CE approach:
- keep recent turns verbatim
- maintain memory for order ID + current intent
- inject only the latest shipping status

### Failure Mode

The model answers the current turn using stale intent because earlier turns were never distilled into state.

---

## 4. Scenario 3: Long Document Analysis

### Goal

Extract, compare, or reason over material that approaches or exceeds the effective context window.

### Recommended Context Shape

```text
[System Prompt]
[Section Summaries or Routing Layer]
[Most Relevant Detail Chunks]
[User Query]
```

### What to Emphasize

- Semantic chunking
- Hierarchical summaries
- Position-aware placement
- Map-reduce or refine workflows for very long inputs

### What to Avoid

- Dumping an entire 100-page document into one call by default
- Treating all sections as equally relevant
- Placing the key chunk in the middle of a long block

### Example

User asks: "Find the liability clauses in this 120-page contract and summarize the risk."

Best CE approach:
- summarize sections first
- route to the relevant sections
- place the liability clauses closest to the query

### Failure Mode

The relevant clauses are technically present, but buried in the middle of a long prompt and missed by the model.

---

## 5. Scenario 4: Coding Assistant

### Goal

Help the model reason over a codebase while keeping context local and actionable.

### Recommended Context Shape

```text
[System Prompt / Coding Rules]
[Current File + Cursor Region]
[Dependency or Call-Site Context]
[Recent Edits / Test Failures]
[User Task]
```

### What to Emphasize

- Current file and local region first
- Dependency-aware retrieval
- Recent diffs and failing tests
- Minimal but high-signal code context

### What to Avoid

- Dumping the entire codebase
- Including unrelated files from the same repository
- Feeding raw terminal output without compression

### Example

User asks: "Make this endpoint paginated and fix the tests."

Best CE approach:
- current handler
- service layer implementation
- related tests
- recent failure output

### Failure Mode

The model sees too much code and loses the local edit target.

---

## 6. Scenario 5: Tool-Using Agent

### Goal

Execute a multi-step workflow without drowning in accumulated tool outputs and intermediate reasoning.

### Recommended Context Shape

```text
[System Prompt]
[Task Definition]
[Tool Schemas]
[Condensed Scratchpad State]
[Latest Relevant Tool Outputs]
[Current Step]
```

### What to Emphasize

- Scratchpad summarization
- Hard caps on tool returns
- Checkpointing on long runs
- Tool result pruning after each step

### What to Avoid

- Appending every tool result forever
- Injecting raw JSON or multi-page outputs
- Confusing completed reasoning with current working state

### Example

Research agent workflow:
- search web
- open reports
- compare vendors
- draft summary

Best CE approach:
- retain only active subproblem state
- compress resolved tool results
- preserve key findings, not the full trace

### Failure Mode

The agent degrades after 20-30 steps because context growth becomes the dominant failure mode.

---

## 7. Scenario 6: Personalized Assistant

### Goal

Respond using durable user preferences without loading irrelevant personal context every turn.

### Recommended Context Shape

```text
[System Prompt]
[Relevant User Profile Memory]
[Recent History (if needed)]
[Current Query]
```

### What to Emphasize

- Selective memory injection
- Stable preferences over raw transcripts
- Task-conditioned memory loading

### What to Avoid

- Injecting the full user profile into every request
- Mixing old preferences with no recency or confidence control
- Treating personalization as a replacement for retrieval

### Example

Relevant memory:
- prefers concise answers
- lives in Shanghai
- allergic to shellfish

These may matter for:
- travel planning
- restaurant suggestions
- product recommendations

They usually do not matter for:
- explaining transformer attention
- debugging a Python stack trace

### Failure Mode

The prompt becomes bloated with personalization that has no bearing on the current task.

---

## 8. Scenario 7: Multimodal Workflow

### Goal

Balance text, image, audio, and structured inputs within the same limited budget.

### Recommended Context Shape

```text
[System Prompt]
[Critical Images / Audio Metadata]
[Compressed Text Context]
[Current Query]
```

### What to Emphasize

- Resolution control
- Modality-aware budgeting
- Image or audio downscaling before dropping text
- Positioning textual instructions clearly

### What to Avoid

- Sending every image at maximum detail
- Ignoring modality-specific token costs
- Letting multimodal history silently crowd out query-critical text

### Example

User uploads 8 screenshots and asks for UI diagnosis.

Best CE approach:
- keep the 2-3 most informative images at higher detail
- downgrade the rest
- include a concise issue description

### Failure Mode

High-resolution images consume the budget and squeeze out the instructions that explain what the model should analyze.

---

## 9. A Practical Selection Matrix

| Scenario | Heaviest Layers | Lightest Layers | Key Ordering Move |
| :--- | :--- | :--- | :--- |
| Factual Q&A | RAG, Query | History, Memory | Put top evidence next to query |
| Support chat | Recent History, Memory, Query | Broad RAG | Keep last turns verbatim |
| Document analysis | Summaries, Detail Chunks, Query | Full history | Route first, then place relevant chunk near end |
| Coding assistant | Current file, dependencies, task | Long conversation | Keep local code closest to task |
| Tool agent | Scratchpad state, latest tool outputs | Old tool logs | Compress old steps aggressively |
| Personalized assistant | User profile memory, query | Irrelevant history | Load only task-relevant profile facts |
| Multimodal | Critical media + text query | Full-detail media everywhere | Spend budget on the most informative modality |

---

## 10. The Meta-Rule

The best context strategy is not the one that includes the most information.

It is the one that:
- includes the minimum sufficient information
- places the most important pieces where the model will actually attend to them
- preserves continuity only where continuity matters
- compresses or isolates everything else

That is why the real CE loop is always:

**Task type -> context selection -> budget allocation -> position-aware assembly -> evaluation**

---

## Key References

1. **Anthropic. (2025). Effective Context Engineering for AI Agents.** Anthropic Engineering Blog.
2. **Anthropic. (2025). Long Context Prompting Tips.** Anthropic Documentation.
3. **OpenAI. (2025). GPT-4.1 Prompting Guide.** OpenAI Cookbook.
4. **Google. (2025). Gemini Long Context Guide.** Google AI for Developers.
5. **LangChain. (2025). Context Engineering for Agents.** LangChain Blog.
