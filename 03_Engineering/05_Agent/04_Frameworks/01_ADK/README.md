# Google ADK (Agent Development Kit)

*Prerequisite: [../../01_Theory/02_Agent_Architecture.md](../../01_Theory/02_Agent_Architecture.md).*

---

Google ADK is a framework for building hierarchical agent systems with enterprise-grade tooling. It follows a "corporate structure" metaphor where parent agents delegate tasks to specialized child agents.

## Core Architecture

### ADK's Philosophy

```
CEO (Parent Agent) → VP (Sub-agent) → Specialist (Worker)
```

ADK agents are organized in a tree hierarchy, not flat coordination. This matches enterprise software patterns where delegation flows through management layers.

### Key Components

| Component | Purpose | ADK Analogy |
| :--- | :--- | :--- |
| **BaseAgent** | Abstract base class for all agents | "Employee" |
| **LlmAgent** | LLM-powered agent with reasoning + tool use | "Knowledge Worker" |
| **SequentialAgent** | Executes tasks in a pipeline | "Assembly Line" |
| **ParallelAgent** | Concurrent task execution | "Project Team" |
| **LoopAgent** | Iterative execution until condition met | "Quality Assurance" |
| **CustomAgent** | User-defined specialized logic | "Consultant" |

### Tool Integration

ADK uses a declarative tool registry. Tools are Python functions decorated with type hints that ADK automatically converts into LLM function-calling schemas.

```python
def get_current_time(city: str) -> dict:
    return {"status": "success", "city": city, "time": "10:30 AM"}
```

## Examples Progression

### 01_Single_Agent/
- **Minimal working example**: A single agent with one tool
- **Purpose**: Demonstrate ADK's basic setup and tool integration
- **Key concepts**: Agent instantiation, tool registration, LLM configuration

### 02_MultiTools_Agent/
- **Multiple tools**: Time + weather tools in a single agent
- **Purpose**: Show tool composition and parameter validation
- **Key concepts**: Tool organization, error handling, multi-domain agents

### 03_Agent_Team/
- **Hierarchical structure**: Multi-agent coordination
- **Purpose**: Demonstrate ADK's hierarchical delegation pattern
- **Key concepts**: Parent-child relationships, task distribution, result synthesis

## Prerequisites

```bash
# Install Google ADK
pip install google-adk

# Set up environment
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://api.openai.com/v1"
```

**Note**: These examples use `flashboot_core.utils` for local development convenience. In production, replace with standard configuration management.

## Production Considerations

### Strengths
- **Enterprise-ready**: Built-in logging, metrics, and error handling
- **Hierarchical**: Matches organizational structures naturally
- **Tool-first**: Excellent tool discovery and validation
- **Google ecosystem**: Integrates with Vertex AI, BigQuery, etc.

### Limitations
- **Google-centric**: Optimized for Google Cloud services
- **Learning curve**: Hierarchical thinking requires mental model shift
- **Overhead**: More boilerplate than minimalist frameworks

## When to Choose ADK

- Building enterprise assistant systems
- Need hierarchical task delegation
- Already invested in Google Cloud ecosystem
- Require production-grade observability and security

## Key References

- [Google ADK Documentation](https://cloud.google.com/adk)
- [Building Collaborative AI with ADK](https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk)
- [ADK GitHub Examples](https://github.com/google/adk-examples)