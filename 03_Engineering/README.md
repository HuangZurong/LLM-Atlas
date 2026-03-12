# Track 3: The Engineer (System Development)

This track covers the engineering required to build robust LLM applications. It focuses on prompt engineering, context management, retrieval-augmented generation (RAG), agents, and production deployment.

## 🗺️ Curriculum Map

### [01. LLMs](./01_LLMs)

- Model landscape, tokenization, API mechanics, cost optimization.

### [02. Prompt Engineering](./02_Prompt_Engineering)

- Instruction engineering, reasoning patterns (CoT, ReAct), structured output, DSPy optimization.

### [03. Context Engineering](./03_Context_Engineering)

- Context window mechanics, context composition and priority, token budget management, long context techniques, dynamic context management for agents.

### [04. Memory](./04_Memory)

- Cross-session memory systems, vector stores, entity extraction, memory lifecycle.

### [05. RAG](./05_RAG)

- Retrieval Augmented Generation pipelines, advanced RAG, GraphRAG, agentic RAG.

### [06. Agent](./06_Agent)

- Tool-use, ReAct frameworks, multi-agent collaboration, MCP protocol.

### [07. Deployment](./07_Deployment)

- vLLM, PagedAttention, quantization, continuous batching, cloud deployment.

### [08. Security](./08_Security)

- Guardrails, prompt injection detection, PII redaction, safety benchmarks.

### [09. LLMOps](./09_LLMOps)

- Monitoring, evaluation automation, observability, CI/CD for LLM applications.

---

## Module Progression

```
01_LLMs → 02_Prompt_Engineering → 03_Context_Engineering → 04_Memory → 05_RAG → 06_Agent
                                                                                      ↓
                                                              09_LLMOps ← 08_Security ← 07_Deployment
```

Each module follows a three-layer structure:
- `01_Theory/` — Concepts and mental models
- `02_Practical/` — Working code implementations
- `03_Best_Practice/` — Production patterns and decision frameworks

---

_Note: Deployment and LLMOps modules were migrated from the Scientist track to reflect their production focus. Context Engineering was added as a dedicated module (previously scattered across 01_LLMs and 03_Memory)._
