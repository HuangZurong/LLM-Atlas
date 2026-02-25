# Agno AI Agent Framework

*Prerequisite: [../../01_Theory/02_Agent_Architecture.md](../../01_Theory/02_Agent_Architecture.md).*

---

**Agno** is a modern, lightweight agent framework focused on **simplicity**, **composability**, and **production readiness**. It takes a minimalist approach compared to heavyweight frameworks like LangChain, emphasizing clean APIs and straightforward agent patterns.

## Core Philosophy

### Minimalist Design

```
Other Frameworks:    Agno:
┌──────────────┐    ┌──────────────┐
│   Bloat      │    │  Essentials  │
│  Complexity  │    │   Simplicity │
│ Many Abstractions│ Few Abstractions│
└──────────────┘    └──────────────┘
```

**Agno's Principles**:
1. **Zero Magic**: Explicit over implicit, no hidden behaviors
2. **Composability**: Small, focused components that combine easily
3. **Pythonic**: Uses standard Python patterns, not custom DSLs
4. **Async-First**: Built for modern async/await patterns

## Key Features

### 1. Simple Agent Definition

```python
from agno import Agent

# Define an agent in 3 lines
agent = Agent(
    name="ResearchAssistant",
    instructions="You are a helpful research assistant.",
    model="gpt-4o",
)
```

### 2. Tool Integration

```python
from agno import Tool

@Tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Integration with Tavily, SerpAPI, etc.
    return fetch_from_web(query)

agent = Agent(
    tools=[search_web],
    # Tool descriptions auto-generated from docstrings
)
```

### 3. Streaming & Async

```python
# Natural async support
async def process_queries(queries: list[str]):
    async for response in agent.stream_responses(queries):
        print(response)
```

### 4. Built-in Observability

```python
# Automatic logging and tracing
agent.run("What is AI?", trace=True)
# Outputs structured logs with timing, token counts, etc.
```

## Architecture

### Component-Based Design

```
┌─────────────────────────────────────────┐
│            Agent                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │  Model  │ │  Tools  │ │  Memory │  │
│  └─────────┘ └─────────┘ └─────────┘  │
│  ┌─────────────────────────────────┐  │
│  │          Workflow Engine        │  │
│  └─────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Compared to Other Frameworks

| Feature | Agno | LangChain | CrewAI |
| :--- | :--- | :--- | :--- |
| **Learning Curve** | Low | High | Medium |
| **Boilerplate** | Minimal | Significant | Moderate |
| **Abstraction Level** | Just right | Too high | Medium |
| **Production Ready** | Yes | Yes | Yes |
| **Community** | Growing | Mature | Growing |
| **Customization** | High | Medium | Medium |

## Quick Start

### Installation

```bash
pip install agno
```

### Basic Usage

```python
from agno import Agent

# 1. Create an agent
agent = Agent(
    name="TravelPlanner",
    instructions="Help users plan trips.",
    model="gpt-4o",
)

# 2. Run it
response = agent.run("Plan a 3-day trip to Tokyo")
print(response)
```

### With Tools

```python
from agno import Agent, Tool

@Tool
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp": "22°C", "condition": "Sunny"}

@Tool
def book_hotel(city: str, nights: int) -> str:
    """Book a hotel."""
    return f"Hotel booked in {city} for {nights} nights"

agent = Agent(
    tools=[get_weather, book_hotel],
    instructions="Help users with travel planning.",
)

# Agent automatically learns when to use tools
response = agent.run("What's the weather in Paris and book 3 nights?")
```

### Multi-Agent Systems

```python
from agno import Agent, Team

researcher = Agent(
    name="Researcher",
    instructions="Research topics thoroughly.",
    model="gpt-4o",
)

writer = Agent(
    name="Writer",
    instructions="Write polished content.",
    model="gpt-4o",
)

editor = Agent(
    name="Editor",
    instructions="Review and improve writing.",
    model="claude-3-5-sonnet",
)

# Create a team
team = Team(
    name="ContentTeam",
    agents=[researcher, writer, editor],
    workflow="sequential",  # researcher → writer → editor
)

# Run the team
result = team.run("Write a blog post about quantum computing")
```

## Advanced Patterns

### 1. Guardrails & Safety

```python
from agno import Agent, Guard

safety_guard = Guard(
    rules=["no harmful content", "no personal data"],
    action="reject",  # or "redact", "modify"
)

agent = Agent(
    guard=safety_guard,
    # All responses go through guard first
)
```

### 2. Memory & Context

```python
from agno import Agent, MemoryStore

# Persistent memory across sessions
memory = MemoryStore(backend="redis")

agent = Agent(
    memory=memory,
    instructions="Remember user preferences.",
)

# Conversation continues across sessions
session1 = agent.run("I like Italian food")
session2 = agent.run("What restaurants do you recommend?")
# Agent remembers the preference
```

### 3. Cost Control

```python
from agno import Agent, Budget

budget = Budget(
    monthly_limit=100.0,  # $100/month
    alert_threshold=0.8,  # Alert at 80%
)

agent = Agent(
    budget=budget,
    model="gpt-4o",  # Automatically falls back to cheaper model if budget tight
)
```

## Production Deployment

### Docker Example

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agno-agent
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: agent
        image: myregistry/agno-agent:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        - name: AGNO_ENV
          value: "production"
```

### Monitoring

Agno provides built-in OpenTelemetry integration:

```python
from agno import Agent, setup_telemetry

# Configure telemetry
setup_telemetry(
    service_name="travel-agent",
    endpoint="http://jaeger:4317",
)

# All agent interactions are traced
agent = Agent(...)
```

## When to Choose Agno

### Choose Agno when:

- **You value simplicity**: Want to get started quickly without learning complex abstractions
- **You're building production systems**: Need reliability, monitoring, and scalability
- **You prefer explicit code**: Don't like magic or hidden behaviors
- **You work with async Python**: Want native async/await support
- **You need lightweight agents**: Don't need all the features of heavyweight frameworks

### Consider alternatives when:

- **You need extensive ecosystem**: LangChain has more integrations
- **You're doing academic research**: Might need more experimental features
- **You're heavily invested in another framework**: Migration cost may not be worth it

## Ecosystem Integration

### Supported Models

- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude series)
- Google (Gemini)
- Open-source (via Ollama, vLLM)
- Azure OpenAI

### Tool Providers

- Web search (Tavily, SerpAPI)
- Databases (PostgreSQL, MongoDB)
- APIs (REST, GraphQL)
- File systems
- Code execution (sandboxed)

### Storage Backends

- Redis (for memory/knowledge)
- PostgreSQL (for structured data)
- S3/MinIO (for files)
- Local file system

## Learning Resources

### Official Documentation

- [Agno Documentation](https://agno-agi.github.io/agno/)
- [GitHub Repository](https://github.com/agno-agi/agno)
- [Examples Gallery](https://github.com/agno-agi/agno-examples)

### Tutorials

1. **Getting Started**: Basic agent creation
2. **Tool Integration**: Adding external capabilities
3. **Multi-Agent Systems**: Building teams of agents
4. **Production Deployment**: Monitoring, scaling, security
5. **Cost Optimization**: Managing API costs effectively

## Community & Support

- **GitHub Issues**: Bug reports and feature requests
- **Discord Community**: Real-time discussions
- **Stack Overflow**: `agno` tag for questions
- **Twitter/X**: `@agno_agi` for updates

## Migration Guide

### From LangChain

```python
# LangChain
from langchain.agents import create_react_agent
agent = create_react_agent(...)

# Agno equivalent
from agno import Agent
agent = Agent(...)  # Much simpler!
```

### Key Differences

1. **Less boilerplate**: Agno requires 50-70% less code
2. **Clearer APIs**: Methods do what they say, no hidden side effects
3. **Better error messages**: Human-readable errors with suggestions
4. **Modern Python**: Uses type hints, async/await, dataclasses

## Future Roadmap

- **Vector store integration**: For RAG capabilities
- **More model providers**: Expanding LLM support
- **Advanced workflows**: Complex multi-agent patterns
- **Enterprise features**: SSO, audit logs, compliance

**Agno is actively developed** with a focus on making agent development accessible, reliable, and enjoyable.