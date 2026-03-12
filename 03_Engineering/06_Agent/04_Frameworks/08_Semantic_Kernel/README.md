# Semantic Kernel: Microsoft's Enterprise Agent SDK

*Prerequisite: [../../01_Theory/02_Agent_Architecture.md](../../01_Theory/02_Agent_Architecture.md).*

---

**Semantic Kernel (SK)** is an open-source SDK that lets you easily combine AI services with conventional programming languages like C#, Python, and Java.

## Core Philosophy

### The "Kernel" as Orchestrator
SK acts as an operating system for AI. It manages "Plugins" (tools) and "Planners" (reasoning) to achieve user goals.

### Key Primitives

| Primitive | Analogy | Description |
| :--- | :--- | :--- |
| **Kernel** | CPU | The central hub that executes functions |
| **Plugins** | Apps | Skills the agent can use (Math, Search, etc.) |
| **Functions** | Methods | Atomic units of logic (Semantic or Native) |
| **Planners** | Logic | Brain that decides which functions to call |
| **Memories** | RAM/Disk | Context and long-term storage |

## Key Features

### 1. Multi-Language Support
Unlike most frameworks which are Python-only, SK is first-class on **C#/.NET**, making it the choice for enterprise Windows/Azure environments.

### 2. Semantic vs. Native Functions
- **Semantic Functions**: Written in natural language (prompts).
- **Native Functions**: Written in standard code (Python/C#).
SK allows them to call each other seamlessly.

### 3. Planners (Reasoning Engines)
SK includes several planners:
- **Sequential Planner**: Creates a step-by-step pipeline.
- **Function Calling Stepwise Planner**: Modern ReAct-style agent.
- **Action Planner**: Selects a single best tool for the task.

## Enterprise Features
- **Strong Typing**: Prevents common runtime errors in complex agents.
- **Observability**: Built-in support for telemetry and logging.
- **Azure Integration**: Optimized for Azure OpenAI and CosmosDB.

## When to Use Semantic Kernel
- Building enterprise applications in **C# / .NET**.
- Requiring a production-grade, strongly-typed SDK.
- Integrating LLMs into existing non-Python software stacks.
