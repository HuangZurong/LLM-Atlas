# Agent Architecture

*Prerequisite: [01_Theory_Overview.md](01_Theory_Overview.md).*

---

The agent "brain" is composed of four key modules: **Profile**, **Memory**, **Planning**, and **Action**. Understanding this architecture is the foundation for building any agentic system.

## The Agent "Brain"

An agent is generally composed of four key modules: Profile, Memory, Planning, and Action.

### 1. Unified Memory System

- **Short-term Memory (Working Memory)**: Focuses on in-context learning and recent conversation history. It is constrained by the LLM's context window.
- **Long-term Memory**: Leverages external storage (e.g., Vector Databases like Milvus, Pinecone) for petabyte-scale retrieval through embedding similarity.
- **Hybrid Memory**: The cutting edge of memory design, combining unstructured vectors with structured Knowledge Graphs (KG).
- **Cognitive Operations**:
  - **Reflection/Self-Critique**: An iterative process where the agent generates a "Critique" of its own past actions and stores this meta-knowledge to avoid future errors.
  - **Summarization**: Compressing long histories into key-value findings to preserve context without exhausting tokens.
  - **GraphRAG**: Traversing relationships in a graph to provide context that simple similarity search might miss (e.g., "How is person A related to building C?").

### 2. Planning Module

Planning is the "Pre-frontal Cortex" of the agent, responsible for task decomposition.

- **Step-by-step reasoning (CoT)**: Forcing the model to output its intermediate logic (`Let's think step by step`).
- **Tree-of-Thought (ToT)**: Generating a tree of potential next steps and using a search algorithm (like BFS or DFS) to evaluate which "branch" is most likely to succeed.
- **Plan-and-Solve**: The agent first generates a multi-step execution plan and then executes it step-by-step, rather than deciding each step on the fly.
- **ReAct (Reasoning + Acting)**: The gold standard for agents, interleaving thoughts with actions to stay grounded in environment feedback.

### 3. Action Space (The Executive Module)

Actions are how the agent impacts the physical or digital world.

- **Tool Use (Function Calling)**: Transforming an intent into a structured API call or code execution.
- **Environment Context**: Giving the agent "senses" (e.g., the ability to read a file or probe a network state) before it takes action.
- **Communication Protocols**: Standards like **MCP** (described in article 04) allow agents to share actions across disparate systems.

### 4. Profile Module

The identity layer. It defines the "Persona" (Role, Goals, Skills). This significantly impacts the LLM's probability distribution for reasoning (e.g., a "Security Auditor" profile will look for vulnerabilities more aggressively than a "General Assistant").
