# OpenAI Swarm: Multi-Agent Orchestration

*Prerequisite: [../../01_Theory/04_Multi_Agent_Systems.md](../../01_Theory/04_Multi_Agent_Systems.md).*

---

**Swarm** is an experimental, lightweight multi-agent orchestration framework from OpenAI. It focuses on making agent coordination **ergonomic**, **lightweight**, and **highly controllable**.

## Core Philosophy

### Handoffs over Orchestrators
Instead of a central "Manager" agent, Swarm uses a **Handoff** pattern:
- Agent A performs its task.
- If it needs help, it "hands off" the conversation to Agent B.
- Agent B continues the task or hands it back.

### Statelessness
Swarm is designed to be stateless and easy to scale. It uses the `ChatCompletion` API directly without complex middleware.

## Key Primitives

### 1. Agent
A simple object containing `instructions` and `functions` (tools).

```python
from swarm import Agent

english_agent = Agent(
    name="EnglishAgent",
    instructions="You only speak English.",
)

spanish_agent = Agent(
    name="SpanishAgent",
    instructions="You only speak Spanish.",
)
```

### 2. Handoffs
A function that returns another Agent object. This is the magic of Swarm.

```python
def transfer_to_spanish():
    return spanish_agent

english_agent.functions.append(transfer_to_spanish)
```

## Why Use Swarm?

| Feature | Benefit |
| :--- | :--- |
| **Lightweight** | No heavy dependencies or complex abstractions. |
| **Controllable** | You define exactly how handoffs happen. |
| **Ergonomic** | Simple Python syntax that feels natural. |
| **Experimental** | Shows OpenAI's vision for multi-agent coordination. |

## Comparison with CrewAI

| Feature | CrewAI | Swarm |
| :--- | :--- | :--- |
| **Complexity** | High | Low |
| **State** | Shared Process | Handoff-based |
| **Orchestration** | Role-based / Sequential | Peer-to-peer |
| **Focus** | Collaborative Teams | Simple Coordination |

## Status Note
Swarm is an **experimental repository** for educational purposes, demonstrating how to build multi-agent systems with minimal code. It is not an officially supported OpenAI product but represents a significant design pattern.
