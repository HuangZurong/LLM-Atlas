# Agent Theory Overview

*Prerequisite: [../../02_Prompting/01_Theory/02_Programmatic_Prompting.md](../../02_Prompting/01_Theory/02_Programmatic_Prompting.md), [../../02_Prompting/01_Theory/04_Structured_Output_and_Function_Calling.md](../../02_Prompting/01_Theory/04_Structured_Output_and_Function_Calling.md) (Tool Use foundation).*
*See Also: [../../../04_Solutions/08_Agent_Workflow_Design.md](../../../04_Solutions/08_Agent_Workflow_Design.md) (business process orchestration).*

---

This directory contains key theoretical concepts, architectural blueprints, and protocol standards for LLM-based Agents.

## Navigation

1.  **[02_Agent_Architecture.md](02_Agent_Architecture.md)**: Core components (Brain, Memory, Action) of an agent.
2.  **[03_Workflow_Patterns.md](03_Workflow_Patterns.md)**: Standard patterns like Prompt Chaining, Routing, and Orchestration.
3.  **[04_Multi_Agent_Systems.md](04_Multi_Agent_Systems.md)**: Distributed intelligence and collaborative archetypes.
4.  **[05_MCP_Protocol.md](05_MCP_Protocol.md)**: The Model Context Protocol standard for data integration.

## Five-Layer Paradigm of AI Agents

To understand the Agent industrial ecosystem, we use the following five-layer paradigm as our core cognitive framework:

### L1. Physical Foundation Layer (FM Engine) —— "The Recruit's Quality"

- **Definition**: Determines basic capabilities such as Context Window size, VRAM footprint, and Alignment (logic, JSON/Code output).
- **Core**: The reasoning capability of the base model sets the intelligence ceiling of the Agent.

### L2. Cognitive Inference Layer (Cognitive Logic) —— "Tactical Prowess"

- **Definition**: Solving "How to think".
- **Core Actions**:
  - **ReAct**: Spinal reflex; interleaved reasoning and acting.
  - **Plan-and-Solve**: Pre-frontal lobe planning; thinking before acting.

### L3. Architectural Support Layer (Support/Organs) —— "Individual Equipment"

- **Definition**: Solving "How to connect to the external world".
- **Core Components**:
  - **GraphRAG**: An external library for the brain providing deep relationship retrieval.
  - **Adapter/Tools**: The "hands" that control external tools and APIs.

### L4. Engineering Orchestration Layer (Orchestration) —— "Social Organization"

- **Definition**: Solving "How multiple brains collaborate".
- **Core Topologies**: **Handoffs**, **Delegation**, and **Role-playing Hierarchies**.
- **Industry Tools**:
  - **Multi-Agent Systems (MAS)**: Tools like **CamelAI** and **AutoGPT** (Agent-to-Agent communication).
  - **Development Frameworks**: **ADK** (Google's hierarchical management), **LangGraph** (Cyclic execution graphs).
  - **Social Patterns**: Defining how a "Founding" agent delegates to "Specialist" agents.

### L5. Industrial Governance Layer (Governance) —— "Command & QC Center"

The critical layer for scaling Agent production safely.

1. **Standardized Contracts (MCP Protocol)**: Like a USB interface, enabling plug-and-play across Agents and tools.
2. **Safety Guardrails**: Real-time monitoring of execution flows to prevent data loss or privacy leaks.
3. **Cognitive Audit Reports (Traceability & Evaluator)**: Back-tracing the "Intelligence Window" path (Trace) to produce validity reports that guide iteration.

### Summary Formula

- **L1-L2** = **The Soul** (IQ and Logic).
- **L3** = **The Body** (Memory and Senses).
- **L4** = **The Society** (Collaboration and Topology).
- **L5** = **The Civilization** (Standards, Safety, and Evolution).

## Core Implementation Concepts

### 1. From LLM to Agent

An Agent is an LLM wrapper that can use tools and reasoning loops to achieve a goal. While a standard LLM is "passive" (stateless response), an Agent is "active" (goal-oriented, stateful).

### 2. The Reasoning Loop

Modern agents primarily use the **ReAct** (Reasoning and Acting) pattern:

- **Thought**: Break down the task.
- **Action**: Call a tool or interact with the environment.
- **Observation**: Read the result of the action.
- **Repeat**: Iterate until the goal is met.

### 3. Memory Management

- **Short-term**: Context window management.
- **Long-term**: Vector stores and knowledge graphs.
- **Procedural**: Storing successful "plans" for future use.
