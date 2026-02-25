# Multi-Agent Systems (MAS)

*Prerequisite: [03_Workflow_Patterns.md](03_Workflow_Patterns.md).*

---

Multi-Agent Systems represent the L4 layer of the Agent paradigm, where intelligence is distributed across multiple autonomous entities that collaborate or compete to solve complex tasks.

## Core Concepts of MAS

### 1. Decentralized Intelligence

In a MAS, there is no single monolithic "brain." Instead, specialized agents handle specific domains, reducing the cognitive load on any single model and increasing system robustness.

### 2. Emerging Behavior

Complex systems and solutions emerge from simple, local interactions between agents. This allows for scalability and flexibility that single-agent systems cannot match.

### 3. Communication and Coordination

Agents must communicate their intentions, findings, and requests. This can be achieved through:

- **Direct Messaging**: Agent A explicitly calls Agent B.
- **Blackboard/State Systems**: Agents read and write to a shared memory space.
- **Negotiation**: Agents "bid" for tasks based on their capabilities and current load.

## Major MAS Archetypes

### 1. Cooperative Multi-Agent System

Agents work in a shared workspace, coordinated by a central manager, to synthesize a collective solution.

- **Mechanism**: Coordinator -> Agents -> Workspace -> Synthesizer -> Solution.

### 2. Competitive Multi-Agent System

Multiple agents propose independent solutions to the same problem. An Evaluator or Selection mechanism picks the best one.

- **Mechanism**: Problem -> Parallel Agents -> Evaluator -> Selection -> Solution.

### 3. Debate-Based / Consensus

Agents with different "roles" or "critics" debate a topic, evaluating each other's outputs until a consensus is reached.

- **Mechanism**: LLM Agents -> Evaluate/Critic -> Consensus -> Final Answer.

### 4. Role-Playing (CamelAI)

**CamelAI** uses a User Agent and an Assistant Agent interacting through role-playing. This reduces human-in-the-loop requirements.

### 5. Hierarchical Management (ADK)

Google's **Agent Development Kit (ADK)** focuses on a corporate-style hierarchy. A "Parent" agent (CEO) manages "Sub-agents" (VPs/Specialists), delegating work and aggregating results.

### 6. Dynamic Swarms (AutoGPT / CrewAI)

Systems where agents are dynamically spawned and assigned roles based on the task at hand. CrewAI, for example, uses a "Process" model (Sequential, Hierarchical) to manage agent interactions.

## ADK Agent Hierarchy and Types

Google's **Agent Development Kit (ADK)** provides a structured architecture where all agents extend a `BaseAgent`:

- **LLM-Based (LlmAgent)**: Handles reasoning, tool use, and task transfer.
- **Workflow Agents**: The orchestrators of complexity.
  - **SequentialAgent**: Assembly line execution.
  - **ParallelAgent**: Concurrent task execution.
  - **LoopAgent**: Iterative execution until condition met.
- **Custom Logic (CustomAgent)**: User-defined specialized behavior.

## Technical Trade-offs

- **Pros**: Parallelization of sub-tasks, specialized expertise, and modular scalability.
- **Cons**: High token consumption due to inter-agent communication, potential for "Infinite Loops" of reasoning, and higher orchestration complexity.

## Key References

- [Building Collaborative AI: A Developer's Guide to Multi-Agent Systems with ADK](https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk)
