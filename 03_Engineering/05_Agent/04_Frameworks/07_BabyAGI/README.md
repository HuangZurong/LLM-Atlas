# BabyAGI: Task-Driven Autonomous Agent

*Prerequisite: [../../01_Theory/01_Theory_Overview.md](../../01_Theory/01_Theory_Overview.md).*

---

**BabyAGI** is a simplified version of the autonomous agent concept, focusing on an efficient task management loop. It was inspired by the original AutoGPT but optimized for simplicity and speed.

## Core Philosophy

### Task-Centric Loop

BabyAGI operates using three specialized agents in a continuous loop:
1. **Task Execution Agent**: Performs the top task on the list.
2. **Task Creation Agent**: Generates new tasks based on the result of the executed task and the overall objective.
3. **Task Prioritization Agent**: Reorders the task list to ensure the most important tasks are done first.

### The Loop Visualization

```
Objective
   ↓
[Task List] ←── [Task Prioritization Agent]
   ↓                   ↑
[Task Execution Agent] → [Task Creation Agent]
   ↓
 Result
```

## Key Features

### 1. Simple State
The entire state is just the Objective, the current Task List, and the Results so far.

### 2. Infinite Loop
It continues until the objective is met or stopped by the user.

### 3. Lightweight
Can be implemented in a single Python script with minimal dependencies.

## Comparison with AutoGPT

| Feature | BabyAGI | AutoGPT |
| :--- | :--- | :--- |
| **Complexity** | Low | High |
| **Logic** | Task-based | Reasoning-based |
| **Tools** | Basic | Extensive |
| **Focus** | Efficiency | Capability |

## Use Case
- Rapid prototyping of autonomous workflows
- Simple multi-step research
- Educational demonstrations of agent logic
