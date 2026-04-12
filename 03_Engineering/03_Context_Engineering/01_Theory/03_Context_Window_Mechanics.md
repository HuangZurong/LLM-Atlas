# 03 · Context Window Mechanics

*Prerequisite: [../../02_Prompt_Engineering/01_Theory/01_Foundations_and_Anatomy.md](../../02_Prompt_Engineering/01_Theory/01_Foundations_and_Anatomy.md).*

---

## 1. The Context Engineering Mental Model

Before diving into mechanics, understand the overarching framework:

### 1.1 One Core and Two Axes

**One Core**: Treat the Context Window as **extremely expensive, highly constrained, and unreliable system RAM**. Context Engineering decides *what* to load, in *what order*, and at *what priority* when multiple sources compete for this limited space.

**Two Axes**:
- **Spatial Axis**: How to arrange information within a single request (Prefix Caching, Lost in the Middle, Sandwich Pattern).
- **Temporal Axis**: How context evolves across multi-turn conversations and agent loops (covered in `07_Dynamic_Context_Management.md`).

### 1.2 The Three Stages of Evolution

**Three Stages** of evolution:
1. Static Assembly → 2. Dynamic Budgeting → 3. Agentic Orchestration.

*This document covers the Spatial Axis and the physical constraints that drive all budgeting decisions.*

These stages are not abstract labels. They describe a typical maturity path for real context engineering systems.

#### 1.2.1 Stage 1: Static Assembly

This is the earliest and most common pattern. The developer manually stitches together the context:
- fixed system prompt
- fixed few-shot examples
- a small amount of history
- the current user query

It works well for single-turn Q&A, FAQ bots, and simple classification tasks.  
Its strength is simplicity; its weakness is that every request tends to share the same rigid structure, with little flexibility and almost no budgeting discipline.

#### 1.2.2 Stage 2: Dynamic Budgeting

At this stage, the system starts treating context as a resource to allocate:
- count tokens first
- assign budgets across layers
- trim by priority when over budget
- reserve output space explicitly

These systems start making explicit distinctions between:
- what must be preserved
- what can be compressed
- what can be dropped

This is the critical shift from "put everything into the prompt" to "allocate scarce resources under constraints."

#### 1.2.3 Stage 3: Agentic Orchestration

Once the system enters multi-step tool use, long-running execution, or multi-agent coordination, context stops being a static prompt and becomes a live runtime state. At that point you must manage:
- accumulated tool results
- growing scratchpads
- elongated reasoning traces
- checkpoints, compaction, and context resets
- isolation boundaries between agents

At this stage, the core problem is no longer "write a good prompt." It becomes "maintain a context state that remains runnable, compressible, and observable over time."

### 1.3 The Four-Step Pipeline

**Four-Step Pipeline** for production systems:
1. **Load** → 2. **Budget & Sort** → 3. **Compress & Degrade** → 4. **Assemble & Observe**.

This four-step pipeline is the core workflow behind production-grade context management.

#### 1.3.1 Step 1: Load

First, collect candidate context sources. At this point, you have not yet decided what will make it into the final prompt. Candidate sources may include:
- system prompt
- few-shot examples
- user profile or long-term memory
- RAG results
- recent conversation history
- tool outputs
- the current query

The essence of this stage is: **build a candidate pool, not the final input.**

#### 1.3.2 Step 2: Budget & Sort

Once the candidate pool is gathered, the system starts counting tokens and ranking content by priority. It must answer:
- what can never be removed
- what can be shortened
- what should be dropped first if over budget
- how much space must be reserved for output

This step solves the allocation problem. Many systems fail not because they lack compression, but because they never sorted by priority in the first place.

#### 1.3.3 Step 3: Compress & Degrade

If the candidate pool exceeds budget, the system cannot just truncate blindly. It must degrade by layer, for example:
- summarize old history
- shrink RAG from top-5 to top-2
- convert raw tool output into concise summaries
- drop low-priority context entirely

Here, "degrade" does not mean "delete at random." It means **retreat progressively based on importance and information density.**

#### 1.3.4 Step 4: Assemble & Observe

Only at this final stage does the system construct the prompt that will actually be sent to the model. At the same time, it should record observability signals such as:
- token usage by layer
- whether cache was hit
- whether compression was triggered
- total latency and total cost

The point of this stage is: **assemble the prompt while keeping the system debuggable, explainable, and optimizable.**

---

## 2. What the Context Window Actually Is

The context window is the **total token budget** for a single LLM call — it includes both input (prompt) and output (completion). Everything the model "knows" during a call must fit inside it.

```
┌─────────────────────────────────────────────────────┐
│                  Context Window (e.g. 128K tokens)  │
│                                                     │
│  ← Input Tokens ──────────────────→ ← Output →     │
│  [System][Memory][RAG][History][Query]  [Response]  │
└─────────────────────────────────────────────────────┘
```

These slots can be interpreted as follows:
- **System**: The system prompt. It defines the role, constraints, rules, and output requirements. It is usually the highest-priority layer.
- **Memory**: Durable facts about the user or task that remain relevant across turns, often loaded from external storage.
- **RAG**: External material retrieved specifically for the current question, such as documentation chunks, policy excerpts, or code snippets.
- **History**: The recent conversation trail that preserves continuity and tells the model what has already been said.
- **Query**: The current user request or instruction that directly triggers this call.
- **Response**: The model's output. It shares the same total token budget as the input, so space must be reserved for it.

### 2.1 `System`: the governing instruction layer

`System` is usually the highest-priority layer in the prompt. It establishes the model's role, boundaries, rules, and expected output behavior.

Typical contents include:
- role definition: `You are a customer support assistant`
- behavioral rules: `If information is missing, do not invent facts`
- output requirements: `Return JSON only`
- safety constraints: `Do not reveal internal policy`

If the full context is treated like runtime memory, then `System` is the execution contract that governs how all later layers should be interpreted.

### 2.2 `Memory`: durable background facts

In context engineering, `Memory` does not mean the model's parameters, nor does it mean all past dialogue. More precisely, it means: **facts explicitly injected into the current context window that remain relevant to the user or task over time**.

It usually comes from external stores rather than the current turn itself, for example:
- user profile: `the user is allergic to seafood`, `the user prefers a formal tone`
- task facts: `this ticket has already been escalated to L2`, `the previous refund failed because the card had expired`
- historical summary: `the first 20 turns already established destination=Tokyo, budget=8000, departure=October 1`

In practice, the distinction is:
- **History**: the raw dialogue trail — what happened
- **Memory**: distilled stable facts — what still matters
- **RAG**: retrieved external evidence — what this answer must consult

So `Memory` is best understood as "persistent background facts," while `RAG` is "material fetched specifically for this task."

### 2.3 `RAG`: retrieved material for the current task

`RAG` refers to the retrieved content that is actually inserted into the context window for the current request. Its defining property is not durability, but **temporary relevance to the current question**.

Common sources include:
- knowledge base chunks
- product documentation excerpts
- policy paragraphs
- relevant files or functions from a codebase

For example, if the user asks, "Can I still request a refund after 30 days?", the system may retrieve two relevant policy passages and inject them as the `RAG` layer for this call.

So `RAG` answers the question "what must I look up right now?", not "what facts remain true about this user or task over time?"

### 2.4 `History`: the recent conversational trail

`History` is the recent sequence of dialogue turns leading up to the current call. Its purpose is not to add external knowledge, but to preserve continuity:
- what was just discussed
- what has already been confirmed
- what the current question is referring back to

For example:
- if the user said in the previous turn, "Not Beijing, Shanghai,"
- the next turn must preserve the fact that the destination has already been corrected

In real systems, `History` usually does not grow without bound. Instead:
- the most recent turns are kept verbatim
- older turns are summarized
- even older context is moved into `Memory` or external storage

### 2.5 `Query`: the direct trigger for this call

`Query` is the actual task, question, or instruction the user is issuing right now.

Examples:
- `Summarize this article`
- `Where is order AC-7823 now?`
- `Fix this bug based on the code below`

If `System` defines how to behave, and `Memory` / `RAG` provide the background needed to act, then `Query` determines what this specific turn is trying to accomplish.

### 2.6 `Response`: output shares the same budget

`Response` is the answer, plan, code, or action result generated by the model. It is not outside the context window — it **shares the same total budget as the input**.

That creates an important engineering constraint:
- the fuller the input becomes
- the less space remains for output
- the more likely the model is to be truncated by length limits

So `Response` is not something to think about only at the end. It must be accounted for from the beginning as part of the total budget.

Unlike human memory, the context window is:
- **Flat**: No inherent hierarchy or importance weighting.
- **Ephemeral**: Discarded after the call ends.
- **Expensive**: Every token has a latency, memory, and billing cost.

## 3. The KV Cache: Why Context is Expensive

Every token in the context generates a **Key-Value (KV) pair** in each Transformer attention layer. These pairs are stored in GPU VRAM.

### Cost Model

```text
KV Cache Memory = 2 × num_layers × hidden_dim × num_tokens × precision_bytes
```

For a 70B model (e.g., Llama 3 70B: 80 layers, 8192 hidden dim, FP16):
- ~2.5 MB VRAM per 1,000 tokens
- A 128K context window consumes **~320 MB per request**.
- If serving 100 concurrent users at full context, KV cache alone requires **~32 GB of VRAM**.

This hardware constraint directly translates to cost limits on API calls.

### Triple Cost Per Token

| Cost Dimension | Impact |
| :--- | :--- |
| **Latency** | More tokens → slower Time-to-First-Token (TTFT) |
| **Memory** | More KV entries → fewer concurrent requests per GPU |
| **Billing** | Input tokens are billed (typically at 50% of output rate) |

### PagedAttention

vLLM's PagedAttention manages KV cache like an OS manages virtual memory — allocating non-contiguous physical memory blocks to logical sequences. This enables higher batch sizes and better GPU utilization without changing model behavior.

## 4. Prefix Caching (KV Cache Reuse)

When multiple requests share the same prefix (system prompt + few-shot examples), the KV cache for that prefix can be **reused** across requests.

```
Request 1: [System Prompt][Examples][User Query A]
                ↑ computed and cached ↑
Request 2: [System Prompt][Examples][User Query B]
                ↑ reused — zero recomputation ↑
```

### Provider Support (as of early 2025 — verify against official docs)

| Provider | Mechanism | Discount |
| :--- | :--- | :--- |
| **OpenAI** | Automatic Prefix Caching | ~50% on cached input tokens |
| **Anthropic** | `cache_control` parameter | ~90% on cached tokens |
| **DeepSeek** | Automatic (>32 token prefix) | ~90% on cached tokens |
| **vLLM (self-hosted)** | `--enable-prefix-caching` flag | Free (you own the GPU) |

### 4.1 The Prefix Caching Anti-Pattern: Dynamic Injections

A common pitfall that destroys caching efficiency is injecting dynamic variables (timestamps, UUIDs, or session IDs) into otherwise static blocks.

```text
❌ BAD: Breaks cache every second
System: "You are a helpful assistant. Current time: 2025-03-20 10:00:01"
User:   "What is my task?"

✅ GOOD: Preserves cache
System: "You are a helpful assistant."
User:   "[Time: 2025-03-20 10:00:01] What is my task?"
```

**Engineering Rule**: Isolate dynamic variables to the very end of the prompt (the user message layer). Never pollute the static `[System Prompt]` or `[Few-shot Examples]` with per-request data.

```
✅ [System Prompt][Few-shot Examples][Retrieved Docs][User Query]
   ←────── stable, cacheable ──────→ ←── dynamic ──→

❌ [User Query][System Prompt][Few-shot Examples][Retrieved Docs]
   ← dynamic → ←────────── cache miss every time ──────────────→
```

## 5. Attention Patterns & "Lost in the Middle"

Research (Liu et al., 2023) shows LLMs attend most strongly to:
1. The **beginning** of the context (primacy bias).
2. The **end** of the context (recency bias).
3. Information in the **middle** is frequently ignored or hallucinated.

```
Attention
Strength
  ▲
  │█                                                    █
  │██                                                  ██
  │███                                              ████
  │████                                          ██████
  │█████████                            ████████████████
  └──────────────────────────────────────────────────────→
  Start                                                End
                    Context Position
```

### Mitigations

- **Sandwich Pattern**: Place critical instructions at both the START and END of the prompt.
- **Recency Placement**: Put the most relevant RAG chunk at the END, not buried in the middle.
- **NIAH Testing**: Use Needle-in-a-Haystack tests to validate your model's retrieval accuracy at different context positions before deploying.

### 5.1 Does this happen at all context lengths?

In principle, **position bias exists broadly**, but it is not equally severe at every length.

More precisely:
- in shorter contexts, the effect may already exist, but often remains mild
- in longer contexts, it is much more likely to become a real engineering failure mode
- the longer the context, the deeper the middle, and the more complex the task, the more visible the U-shaped pattern tends to become

So this should not be understood as a phenomenon that "suddenly appears only after some threshold." A better interpretation is:
**models are naturally biased toward the beginning and end, and longer contexts simply amplify that bias.**

As a rough intuition:
- **around 1K context**: the middle region is shallow, so the bias may exist without seriously affecting many simple tasks
- **around 10K context**: if key facts are buried in the middle and surrounded by noise, the U-shape often becomes obvious
- **at 100K scale**: the middle frequently becomes the most dangerous information burial zone, and placement strategy becomes essential

So the engineering question is not "Does Lost in the Middle exist at all?" but rather:
- how severe is it at your target context length?
- can your model still retrieve reliably from the middle?
- are your critical facts being placed in the worst possible region?

That is why long-context systems should run NIAH-style position sensitivity tests before deployment, rather than relying only on the model's advertised context window.

## 6. Effective Window vs. Nominal Window

Just because a model *supports* 128K tokens doesn't mean it can *reason* over 128K tokens effectively.

| Window Type | Definition |
| :--- | :--- |
| **Nominal Window** | The maximum tokens the model accepts without error |
| **Effective Window** | The range over which the model reliably retrieves and reasons |

Most models show measurable performance degradation well before hitting their nominal limit. The effective window is typically 60–80% of the nominal window for complex reasoning tasks.

**Practical implication**: Don't fill the context to 95% capacity. The model needs headroom to "breathe".

### 6.1 The Output Safety Margin

A critical oversight in naive implementations is filling the context to the brim with inputs, leaving no room for the generated response.
If the context budget is 128K and your input is 127.9K, the model will output 100 tokens and abruptly halt (`finish_reason="length"`).

**Engineering Rule**: Always define a strict `Output Reserve` (e.g., 4,000 tokens) that cannot be encroached upon by the input context composer.

## 7. Model Context Windows at a Glance (early 2025)

A reference for planning context budgets. Verify against official documentation — these change frequently.

| Model | Nominal Window | Notes |
| :--- | :--- | :--- |
| **Claude 3.5 Sonnet** | 200K tokens | Prefix caching via `cache_control` |
| **Claude 3.5 Haiku** | 200K tokens | Prefix caching via `cache_control` |
| **GPT-4o** | 128K tokens | Automatic prefix caching |
| **GPT-4o mini** | 128K tokens | Automatic prefix caching |
| **Gemini 1.5 Pro** | 1M tokens | Implicit caching |
| **Gemini 2.0 Flash** | 1M tokens | Implicit caching |
| **DeepSeek-V3** | 128K tokens | Automatic prefix caching (>32T) |
| **Llama 3.3 70B** | 128K tokens | Prefix caching via vLLM flag |
| **Qwen2.5 72B** | 128K tokens | Prefix caching via vLLM flag |

**Key insight**: a larger nominal window does not automatically mean better long-context reasoning. Always run NIAH tests at your target depth before relying on the full window.

---

## Key References

1. **Liu, N. F., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts.** *TACL, 12*, 157–173.
2. **Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention.** *SOSP 2023*, 611–626.
3. **Bertsch, A., et al. (2024). Needle In A Haystack: Evaluating Long-Context Language Models.** *arXiv:2407.05831*.
