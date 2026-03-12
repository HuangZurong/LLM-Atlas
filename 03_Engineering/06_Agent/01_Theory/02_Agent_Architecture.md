# Agent Architecture

*Prerequisite: [01_Theory_Overview.md](01_Theory_Overview.md).*

---

The agent "brain" is composed of four key modules: **Profile**, **Memory**, **Planning**, and **Action**. Understanding this architecture is the foundation for building any agentic system.

## The Agent "Brain"

### 1. Profile Module

The identity layer that shapes the Agent's behavior before any reasoning begins.

- **Role**: What the agent is (e.g., "Security Auditor", "Data Analyst", "Customer Support Rep").
- **Goals**: What the agent aims to achieve (e.g., "Find vulnerabilities", "Generate weekly reports").
- **Constraints**: Boundaries the agent must respect (e.g., "Never modify production data", "Always cite sources").

This significantly impacts the LLM's probability distribution for reasoning — a "Security Auditor" profile will look for vulnerabilities more aggressively than a "General Assistant".

### 2. Memory System

- **Short-term Memory (Working Memory)**: Focuses on in-context learning and recent conversation history. It is constrained by the LLM's context window.
- **Long-term Memory**: Leverages external storage (e.g., Vector Databases like Milvus, Pinecone) for petabyte-scale retrieval through embedding similarity.
- **Hybrid Memory**: The cutting edge of memory design, combining unstructured vectors with structured Knowledge Graphs (KG).
- **Procedural Memory**: Stores validated execution plans and strategies as reusable templates — the agent's "muscle memory" for recurring tasks (e.g., a proven deployment workflow).
- **Cognitive Operations**:
  - **Summarization**: Compressing long histories into key-value findings to preserve context without exhausting tokens.
  - **GraphRAG**: Traversing relationships in a graph to provide context that simple similarity search might miss (e.g., "How is person A related to building C?").

### 3. Planning Module

Planning is the "Pre-frontal Cortex" of the agent, responsible for task decomposition.

- **Prompt-level** (enhancing single-step reasoning quality):
  - **Chain-of-Thought (CoT)**: Forcing the model to output its intermediate logic (`Let's think step by step`).
  - **Tree-of-Thought (ToT)**: Generating a tree of potential next steps and using a search algorithm (like BFS or DFS) to evaluate which "branch" is most likely to succeed.
- **Workflow-level** (human pre-defines the execution flow, LLM executes within each node, see [03_Workflow_Patterns.md](03_Workflow_Patterns.md)):
  - **Prompt Chaining**: Sequential pipeline where each step's output feeds the next.
  - **Routing**: Conditional branching to different handlers based on input classification.
  - **Parallelization**: Running multiple sub-tasks concurrently and aggregating results.
  - **Orchestrator-Workers**: A manager LLM dynamically delegates sub-tasks to worker models and synthesizes results.
  - **Evaluator-Optimizer**: A generate-evaluate-revise loop that iterates until quality criteria are met.
- **Agent-level** (LLM autonomously governs the multi-step loop, see [01_Theory_Overview.md L2](01_Theory_Overview.md)):
  - **ReAct (Reasoning + Acting)**: The gold standard for agents, interleaving thoughts with actions to stay grounded in environment feedback.
  - **Plan-and-Solve**: The agent first generates a multi-step execution plan and then executes it step-by-step.
  - **Reflection**: The agent reviews its own output, identifies flaws, and iterates to improve quality.

### 4. Action Space (The Executive Module)

Actions are how the agent impacts the physical or digital world.

- **Tool Use (Function Calling)**: Transforming an intent into a structured API call or code execution.
- **Environment Context**: Giving the agent "senses" (e.g., the ability to read a file or probe a network state) before it takes action.
- **Communication Protocols**: Standards like **MCP** (see [05_MCP_Protocol.md](05_MCP_Protocol.md)) allow agents to share actions across disparate systems.
