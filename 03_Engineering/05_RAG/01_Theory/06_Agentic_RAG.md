# Agentic RAG

*Prerequisite: [02_Advanced_RAG.md](02_Advanced_RAG.md), [../../05_Agent/01_Theory/01_Theory_Overview.md](../../05_Agent/01_Theory/01_Theory_Overview.md).*
*See Also: [../../05_Agent/01_Theory/02_Agent_Architecture.md](../../05_Agent/01_Theory/02_Agent_Architecture.md) (ReAct, tool-calling), [../../05_Agent/01_Theory/04_Multi_Agent_Systems.md](../../05_Agent/01_Theory/04_Multi_Agent_Systems.md) (multi-agent patterns).*

---

Standard RAG is a **single-shot pipeline**: query → retrieve → generate. **Agentic RAG** replaces this rigid pipeline with an **autonomous agent** that can plan, reason, use tools, and iterate — treating retrieval as one of many available actions rather than a fixed step.

## 1. The Problem with Pipeline RAG

| Limitation | Example |
|---|---|
| **No query planning** | "Compare Apple and Microsoft's 2024 revenue" requires two separate retrievals, but pipeline RAG does one |
| **No retrieval validation** | Retrieved documents may be irrelevant; pipeline RAG feeds them to the LLM anyway |
| **No multi-source orchestration** | The answer may require vector DB + SQL DB + web search; pipeline RAG uses one source |
| **No iterative refinement** | If the first retrieval fails, pipeline RAG has no retry mechanism |
| **No tool use** | Some questions need calculation, code execution, or API calls — not just document retrieval |

Agentic RAG addresses all of these by wrapping the RAG pipeline in an agent loop.

## 2. Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agentic RAG                              │
│                                                                 │
│   User Query                                                    │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────────────────────────────────────┐                 │
│   │            Agent (ReAct Loop)              │                 │
│   │                                            │                 │
│   │  THINK: What do I need to answer this?     │                 │
│   │    │                                       │                 │
│   │    ▼                                       │                 │
│   │  ACT: Choose a tool                        │                 │
│   │    ├── vector_search(query)                │                 │
│   │    ├── sql_query(query)                    │                 │
│   │    ├── web_search(query)                   │                 │
│   │    ├── calculator(expression)              │                 │
│   │    ├── code_executor(code)                 │                 │
│   │    └── sub_agent(task)                     │                 │
│   │    │                                       │                 │
│   │    ▼                                       │                 │
│   │  OBSERVE: Evaluate the result              │                 │
│   │    ├── Sufficient? → final_answer()        │                 │
│   │    └── Insufficient? → THINK again         │                 │
│   └───────────────────────────────────────────┘                 │
│       │                                                         │
│       ▼                                                         │
│   Final Answer (grounded in retrieved evidence)                 │
└─────────────────────────────────────────────────────────────────┘
```

The key difference: **retrieval is a tool call, not a pipeline stage**. The agent decides **when**, **how many times**, and **from which source** to retrieve.

## 3. Design Patterns

### 3.1 Single-Agent RAG (Router Agent)

The simplest form: one agent with multiple retrieval tools.

```
Agent
├── Tool: vector_search    (semantic retrieval)
├── Tool: keyword_search   (BM25 / full-text)
├── Tool: sql_query         (structured data)
├── Tool: graph_query       (knowledge graph)
└── Tool: web_search        (live internet)
```

- The agent reads the query and **routes** to the appropriate tool (cf. Routing in Advanced RAG Section 3).
- Unlike static routing, the agent can **chain** multiple tools in sequence: retrieve from vector DB → validate → if insufficient, search the web.
- **Best For**: Multi-source retrieval with moderate complexity.

### 3.2 Adaptive Retrieval Agent

The agent dynamically decides its retrieval strategy based on query complexity:

```
Query: "What is 2+2?"
  → Agent: No retrieval needed. Direct answer.

Query: "What was Apple's Q3 2024 revenue?"
  → Agent: Single retrieval from financial DB.

Query: "Compare the AI strategies of Apple, Google, and Microsoft in 2024"
  → Agent: Decompose into 3 sub-queries → Retrieve for each → Synthesize.
```

This mirrors **Self-RAG** (Advanced RAG Section 7.1) but with explicit tool-calling rather than special tokens. The agent's reasoning trace is interpretable and debuggable.

### 3.3 Multi-Agent RAG

For complex workflows, decompose the RAG pipeline into **specialized agents**:

```
Orchestrator Agent
│
├── Research Agent
│   ├── Tool: vector_search
│   ├── Tool: web_search
│   └── Tool: academic_search
│
├── Data Agent
│   ├── Tool: sql_query
│   ├── Tool: calculator
│   └── Tool: code_executor
│
├── Fact-Check Agent
│   ├── Tool: vector_search (different corpus)
│   └── Tool: web_search
│
└── Writer Agent
    └── Synthesize findings into final answer
```

- The **Orchestrator** plans the overall approach and delegates sub-tasks.
- Each specialist agent runs its own ReAct loop with domain-specific tools.
- The **Fact-Check Agent** validates claims before the final answer — a structural solution to hallucination.
- **Best For**: Enterprise Q&A systems with high accuracy requirements and diverse data sources.

### 3.4 Corrective Agent (CRAG as Agent)

CRAG's logic (Advanced RAG Section 6.4) implemented as an agent:

```
Agent receives query
  │
  ├── ACT: vector_search(query) → documents
  │
  ├── THINK: Evaluate relevance of each document
  │   ├── Document A: Relevant ✓
  │   ├── Document B: Ambiguous → extract useful parts
  │   └── Document C: Irrelevant ✗
  │
  ├── THINK: Insufficient coverage. Need more.
  │
  ├── ACT: web_search(refined_query) → supplementary results
  │
  ├── THINK: Now I have enough context.
  │
  └── ACT: final_answer(synthesized response)
```

The agent's advantage over static CRAG: it can **reason about why** a document is irrelevant and craft a better follow-up query, rather than applying fixed rules.

## 4. Agentic RAG vs Advanced RAG

| Dimension | Advanced RAG (Pipeline) | Agentic RAG |
|---|---|---|
| **Control flow** | Fixed: query → translate → route → retrieve → rerank → generate | Dynamic: agent decides at each step |
| **Retrieval calls** | Usually 1 (or fixed N for multi-query) | 0 to N, as needed |
| **Error recovery** | CRAG rules, RRR loop | Agent reasons about failures and adapts |
| **Multi-source** | Static routing (logical/semantic) | Dynamic tool selection per step |
| **Tool use** | Retrieval only | Retrieval + calculation + code + APIs |
| **Cost** | Predictable (fixed pipeline) | Variable (more LLM calls for reasoning) |
| **Latency** | Lower (fewer LLM calls) | Higher (multi-turn reasoning) |
| **Debuggability** | Trace through fixed stages | Full reasoning trace in agent memory |

**When to use which**:
- **Advanced RAG**: High-volume, latency-sensitive, well-defined query patterns.
- **Agentic RAG**: Complex queries, diverse data sources, high accuracy requirements, or when the query type is unpredictable.

## 5. Failure Modes & Guardrails

Agentic RAG introduces agent-specific risks that pipeline RAG does not have.

### 5.1 Failure Modes

| Failure | Description | Symptom |
|---|---|---|
| **Infinite loop** | Agent keeps retrieving without converging on an answer | Step count hits max_steps; repeated similar tool calls |
| **Over-retrieval** | Agent retrieves far more documents than needed, flooding the context | Token budget exhausted; latency spike; answer quality degrades from noise |
| **Tool misuse** | Agent calls the wrong tool (e.g., web_search for a question answerable from the internal KB) | Irrelevant results; data leakage to external services |
| **Context overflow** | Accumulated observations across steps exceed the LLM's context window | Truncated context; agent "forgets" early observations |
| **Hallucinated tool calls** | Agent invents tool names or parameters that don't exist | Runtime errors; crashes |
| **Premature termination** | Agent calls final_answer before gathering sufficient evidence | Shallow or incorrect answers |

### 5.2 Guardrails

| Guardrail | Mitigates |
|---|---|
| **max_steps limit** (e.g., 5–10) | Infinite loop, over-retrieval |
| **Token budget per step**: summarize observations before appending to memory | Context overflow |
| **Tool schema validation**: reject calls that don't match defined tool signatures | Hallucinated tool calls |
| **Retrieval deduplication**: skip re-retrieving documents already in memory | Over-retrieval, infinite loop |
| **Mandatory source check**: require at least one retrieval before final_answer | Premature termination |
| **Tool access control**: restrict which tools are available per query type (e.g., no web_search for internal-only queries) | Tool misuse, data leakage |
| **Step-level monitoring**: log each THINK/ACT/OBSERVE for post-hoc analysis | All failures (detection) |

## 6. Evaluation

Agentic RAG is multi-step and non-deterministic, making evaluation harder than pipeline RAG.

### 6.1 End-to-End Metrics

| Metric | What it measures |
|---|---|
| **Answer correctness** | Is the final answer factually correct? (same as pipeline RAG) |
| **Faithfulness** | Is the answer grounded in retrieved documents, not hallucinated? |
| **Answer completeness** | Does the answer cover all aspects of the query? |

### 6.2 Agent-Specific Metrics

| Metric | What it measures |
|---|---|
| **Step efficiency** | How many steps did the agent take? Fewer is better (given same answer quality) |
| **Tool selection accuracy** | Did the agent choose the right tool at each step? Requires annotated traces |
| **Retrieval sufficiency** | Did the agent retrieve enough relevant documents before answering? |
| **Recovery rate** | When the first retrieval failed, did the agent successfully recover? |
| **Cost per query** | Total LLM tokens + API calls. Directly tied to step count |

### 6.3 Evaluation Methods

- **Trace-level evaluation**: Annotate the full reasoning trace (THINK → ACT → OBSERVE per step). Judge each decision point. Expensive but comprehensive.
- **Outcome-only evaluation**: Only evaluate the final answer (correctness, faithfulness). Cheaper but doesn't diagnose *why* the agent fails.
- **LLM-as-Judge on traces**: Feed the agent's full trace to a judge LLM and ask: "Were the tool choices appropriate? Was evidence sufficient before answering?"
- **Regression testing**: Maintain a golden set of (query, expected_trace_pattern, expected_answer). Run after each change to detect regressions in agent behavior.

## 7. Framework Mappings

Agentic RAG is a pattern, not a framework. It can be implemented in any agent framework:

| Framework | Agentic RAG Support | Key Feature |
|---|---|---|
| **LangGraph** | Native. Build RAG as a stateful graph with retrieval, evaluation, and re-retrieval nodes. | Conditional edges enable CRAG-style branching. Checkpointing allows replay. |
| **LlamaIndex** | Built-in `QueryEngine` + `AgentRunner`. `SubQuestionQueryEngine` decomposes queries across data sources. | Deep RAG integration. Index-as-tool is first-class. |
| **smolagents** (HuggingFace) | `CodeAgent` or `ToolCallingAgent` with retrieval tools. | Lightweight. Code-as-action enables multi-step retrieval logic in one step. |
| **CrewAI** | Multi-agent RAG via role-based agents (Researcher, Analyst, Writer). | Declarative role assignment. Good for multi-agent RAG (Pattern 3.3). |
| **Google ADK** | Agent with tool declarations. Supports sub-agents for orchestration. | Native Google Search integration. |
| **OpenAI Agents SDK** | Function-calling agents with retrieval tools. `FileSearch` tool built-in. | Tight integration with OpenAI models and vector store. |

See [../../05_Agent/04_Frameworks/](../../05_Agent/04_Frameworks/) for detailed framework documentation.

## 8. Implementation Considerations

### 8.1 Retrieval as a Tool

The key abstraction: wrap each retrieval method as a tool with a clear docstring.

```python
@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Search the internal knowledge base using semantic similarity.
    Use this for questions about company policies, product docs, and internal processes.

    Args:
        query: the search query
        top_k: number of results to return
    """
    results = vector_db.similarity_search(query, k=top_k)
    return format_results(results)
```

The tool **docstring** is critical — it is how the agent decides when to use this tool vs. others.

### 8.2 Grounding & Citation

Agentic RAG must maintain **provenance**:

- Each retrieved chunk carries a `source_id` and `page_number`.
- The agent's final answer includes inline citations: "According to [Document A, p.12], the revenue was..."
- This is easier in agentic RAG because the agent's memory contains the full reasoning trace, making it straightforward to track which sources informed which claims.

### 8.3 Cost Control

Agentic RAG can be expensive (multiple LLM calls per query). Mitigation strategies:

- **Max steps limit**: Cap the agent's reasoning loop (e.g., `max_steps=5`).
- **Tiered models**: Use a cheap model (GPT-4o-mini) for routing/evaluation, expensive model (GPT-4o) only for final generation.
- **Cache**: Cache frequent query → retrieval results to avoid redundant searches.
- **Adaptive complexity**: Use a classifier to detect simple queries and route them to a fast pipeline RAG, reserving the agent for complex queries.

## 9. The Spectrum of RAG Autonomy

RAG implementations exist on a spectrum from fully manual to fully autonomous:

```
Naive RAG ──► Advanced RAG ──► Agentic RAG ──► Fully Autonomous
(fixed pipe)   (optimized pipe)  (agent loop)    (self-improving)
                                                       │
                                                  Learns from
                                                  feedback to
                                                  improve tools
                                                  and retrieval
```

Most production systems today sit between Advanced RAG and Agentic RAG, using agent-like patterns (CRAG, Self-RAG) within a broadly pipeline architecture. Fully autonomous systems that learn and self-improve are an active research frontier.
