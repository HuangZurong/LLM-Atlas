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

## Technical Trade-offs & Production Challenges

While MAS offers significant architectural benefits, it introduces several critical failure modes in production environments.

### 1. The Pros
- **Parallelization**: Sub-tasks execute concurrently, reducing wall-clock time.
- **Specialized Expertise**: Using smaller, fine-tuned models for specific routes reduces cost and increases accuracy.
- **Modular Scalability**: New capabilities can be added as new agents without modifying core logic.

### 2. The Cons (The "MAS Tax")
- **Token & Cost Explosion**: Inter-agent communication often carries massive overhead as full context/history is passed between entities.
- **Reasoning Overhead**: The "cost of thinking" (orchestration, planning, hand-offs) can exceed the benefit for deterministic or simpler tasks.
- **Latency**: Multiple LLM round-trips for coordination and synthesis significantly increase response time.

### 3. Critical Failure Modes
- **Context Contamination**: Irrelevant information from Agent A's sub-task "pollutes" Agent B's reasoning when sharing a workspace, leading to hallucination by suggestion.
- **State Drift**: An agent's internal model of the world (e.g., file system state) diverges from reality over long-running sessions, leading to actions based on stale data.
- **Infinite Loops**: Evaluator-Optimizer stalemates where agents oscillate between "fixed" and "rejected" states without progress.

---

## Case Study: The Vercel Lesson (Addition by Subtraction)

In 2024, Vercel reported a counter-intuitive finding: **they improved their agent success rate from 80% to 100% by removing 80% of their specialized tools and agents.**

### The Problem
Their initial "sophisticated" setup was fragile and slow. Every time the underlying model (Claude/GPT) improved, the hard-coded agentic logic and specialized tools became a "liability" rather than an asset.

### The Solution: "File System Agent"
- **Simplification**: They removed complex routing and custom tools.
- **Generalization**: They gave the agent raw access to the file system (via Bash) and documentation.
- **Result**: The smarter model (Claude 3.5 Sonnet) could explore, read, and "think" through the raw data more effectively than the hand-coded agentic workflow could guide it.

### Key Takeaway
**Addition by subtraction is real.** As models become more capable at reasoning, developers should shift from *constraining* the agent with rigid multi-agent workflows to *empowering* the agent with general-purpose tools and high-quality context.

---

## Key References

- [Building Collaborative AI: A Developer's Guide to Multi-Agent Systems with ADK](https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk)
