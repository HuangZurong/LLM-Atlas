# LangGraph: Stateful Workflow Orchestration

*Prerequisite: [../../01_Theory/03_Workflow_Patterns.md](../../01_Theory/03_Workflow_Patterns.md).*

---

**LangGraph** is LangChain's framework for building stateful, multi-agent applications with cycles and persistence. It extends LangChain with a graph-based approach to agent orchestration.

## Core Philosophy

### Graphs Over Chains

While LangChain uses linear chains, LangGraph introduces **graphs** where:
- Nodes can be any function or LangChain Runnable
- Edges define control flow
- State is persisted across the entire graph
- Cycles are allowed (agents can loop back)

### Stateful Workflows

```
Traditional Chain:   A → B → C → Done
LangGraph Workflow:  A → B → C → [Condition] → Loop back to A or Continue
                    ↑              ↓
                    └──────────────┘
```

## Key Concepts

### 1. State Management

```python
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END

# Define your state
class State(TypedDict):
    messages: Annotated[List[str], "add_messages"]  # Annotation for reducer
    question: str
    answer: str
    iteration: int
```

### 2. Nodes & Edges

```python
# Define nodes (any callable)
def research_node(state: State):
    # Research logic
    return {"messages": ["Researched topic"]}

def write_node(state: State):
    # Writing logic
    return {"messages": ["Wrote content"]}

# Build graph
graph = StateGraph(State)
graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_edge("research", "write")
graph.add_edge("write", END)
```

### 3. Conditional Edges

```python
def should_continue(state: State):
    # Decide based on state
    if state["iteration"] < 3:
        return "research"  # Loop back
    else:
        return END

graph.add_conditional_edges(
    "write",
    should_continue,
    {"research": "research", END: END}
)
```

## Architecture

### Core Components

```
┌─────────────────────────────────────────┐
│            LangGraph Runtime            │
│                                         │
│  ┌─────────┐     ┌─────────┐     ┌─────┐ │
│  │ Checkpoint │   │  Memory  │   │ State │ │
│  │  Store    │   │  Layer   │   │ Graph │ │
│  └─────────┘     └─────────┘     └─────┘ │
│                                         │
│  ┌─────────────────────────────────────┐ │
│  │         Execution Engine            │ │
│  │  • Node execution                   │ │
│  │  • State transition                 │ │
│  │  • Error handling                   │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Comparison with LangChain

| Feature | LangChain | LangGraph |
| :--- | :--- | :--- |
| **Flow Control** | Linear chains | Complex graphs |
| **State** | Limited context | Full persistence |
| **Cycles** | Not supported | Fully supported |
| **Memory** | Conversation memory | Checkpoint-based |
| **Use Case** | Simple pipelines | Complex workflows |

## Quick Start

### Installation

```bash
pip install langgraph langchain langchain-openai
```

### Basic Example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated

# Define state
class State(TypedDict):
    messages: Annotated[List[str], "add_messages"]
    question: str
    answer: str

# Define nodes
def research(state: State):
    # Simulate research
    new_message = f"Researched: {state['question']}"
    return {"messages": [new_message]}

def write(state: State):
    # Simulate writing
    last_msg = state["messages"][-1]
    answer = f"Based on {last_msg}, here's the answer..."
    return {"answer": answer}

# Build graph
graph = StateGraph(State)
graph.add_node("research", research)
graph.add_node("write", write)
graph.add_edge("research", "write")
graph.add_edge("write", END)

# Compile and run
app = graph.compile()
result = app.invoke({"question": "What is AI?"})
```

### With LLM Integration

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated

class State(TypedDict):
    messages: Annotated[List, "add_messages"]

llm = ChatOpenAI(model="gpt-4o")

def call_llm(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def human_input(state: State):
    user_input = input("Your response: ")
    return {"messages": [HumanMessage(content=user_input)]}

graph = StateGraph(State)
graph.add_node("ai", call_llm)
graph.add_node("human", human_input)
graph.add_edge("ai", "human")
graph.add_edge("human", "ai")  # Creates a loop
```

## Advanced Patterns

### 1. Multi-Agent Teams

```python
class TeamState(TypedDict):
    messages: Annotated[List, "add_messages"]
    research: str
    writing: str
    review: str

def researcher(state: TeamState):
    # Research agent logic
    return {"research": "Research findings..."}

def writer(state: TeamState):
    # Writing agent logic
    return {"writing": "Written content based on research"}

def reviewer(state: TeamState):
    # Review agent logic
    return {"review": "Reviewed and improved content"}

# Orchestrate team workflow
graph = StateGraph(TeamState)
graph.add_node("research", researcher)
graph.add_node("write", writer)
graph.add_node("review", reviewer)

graph.add_edge("research", "write")
graph.add_edge("write", "review")

# Conditional: does it need more research?
def needs_more_research(state: TeamState):
    if state["review"] == "needs_more_info":
        return "research"
    return END

graph.add_conditional_edges("review", needs_more_research)
```

### 2. Checkpointing & Persistence

```python
from langgraph.checkpoint import MemorySaver

# Add checkpointing
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Run with thread ID for persistence
config = {"configurable": {"thread_id": "user-123"}}
result1 = app.invoke({"question": "First question"}, config)
result2 = app.invoke({"question": "Follow-up"}, config)  # Continues from previous state
```

### 3. Human-in-the-Loop

```python
def human_approval_node(state: State):
    print(f"Proposed action: {state['proposed_action']}")
    approval = input("Approve? (y/n): ")

    if approval.lower() == 'y':
        return {"approved": True, "action": state['proposed_action']}
    else:
        return {"approved": False, "feedback": "User rejected"}

def execute_if_approved(state: State):
    if state["approved"]:
        # Execute the action
        return {"result": f"Executed: {state['action']}"}
    else:
        return {"result": "Action cancelled"}
```

## Production Features

### 1. Error Handling

```python
from langgraph.graph import StateGraph, END
import traceback

def safe_node(state: State):
    try:
        # Your node logic
        return {"result": "success"}
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "should_retry": True
        }

def error_handler(state: State):
    if "error" in state:
        print(f"Error occurred: {state['error']}")
        if state.get("should_retry", False):
            return "retry_node"
    return END
```

### 2. Timeouts & Rate Limiting

```python
import asyncio
from functools import wraps

def with_timeout(timeout_seconds: float):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                return {"error": f"Timeout after {timeout_seconds}s"}
        return wrapper
    return decorator

@with_timeout(30.0)
async def slow_node(state: State):
    # Long-running operation
    await asyncio.sleep(25)
    return {"result": "completed"}
```

### 3. Monitoring & Observability

```python
from langgraph.graph import StateGraph
import time

class InstrumentedGraph(StateGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {
            "node_executions": {},
            "total_time": 0,
            "errors": []
        }

    def compile(self, *args, **kwargs):
        original_app = super().compile(*args, **kwargs)

        def instrumented_invoke(inputs, config=None):
            start_time = time.time()
            try:
                result = original_app.invoke(inputs, config)
                duration = time.time() - start_time

                # Record metrics
                self.metrics["total_time"] += duration
                return result
            except Exception as e:
                self.metrics["errors"].append(str(e))
                raise

        return type('InstrumentedApp', (), {'invoke': instrumented_invoke})()
```

## Integration Examples

### With LangChain

```python
from langchain.agents import create_react_agent
from langgraph.prebuilt import create_react_agent_executor

# Create a LangChain agent
agent = create_react_agent(...)

# Wrap in LangGraph for stateful execution
agent_executor = create_react_agent_executor(agent)
```

### With Vector Stores

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Create RAG pipeline with LangGraph
def retrieve_node(state: State):
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
    docs = vectorstore.similarity_search(state["question"])
    return {"context": docs}

def generate_node(state: State):
    context = "\n".join([doc.page_content for doc in state["context"]])
    prompt = f"Context: {context}\nQuestion: {state['question']}"
    # Generate answer with LLM
    return {"answer": llm.invoke(prompt)}
```

### With External APIs

```python
import httpx

async def api_call_node(state: State):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/process",
            json={"input": state["query"]}
        )
        return {"api_response": response.json()}
```

## When to Use LangGraph

### ✅ Good For

- **Complex workflows** with multiple steps and conditions
- **Stateful applications** that need memory across sessions
- **Multi-agent systems** with coordination needs
- **Human-in-the-loop** workflows requiring approval steps
- **Recursive/iterative processes** that need cycles

### ❌ Not Ideal For

- **Simple prompt chains** (use LangChain instead)
- **Stateless APIs** (use FastAPI + LangChain)
- **One-off requests** without persistence needs
- **Simple chatbots** without complex workflows

## Performance Considerations

### Memory Usage

```python
# Use streaming for large state
class StreamingState(TypedDict):
    chunks: List[str]  # Stream chunks instead of full content

# Use checkpoints selectively
checkpointer = MemorySaver(
    serde="json",  # Use efficient serialization
    ttl=3600  # Expire after 1 hour
)
```

### Parallel Execution

```python
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# Run nodes in parallel where possible
graph.add_node("parallel_1", node1)
graph.add_node("parallel_2", node2)
graph.add_edge(START, ["parallel_1", "parallel_2"])  # Both start together
```

## Ecosystem

### Official Integrations

- **LangChain**: First-class integration
- **LangSmith**: Tracing and monitoring
- **LangServe**: Deployment as API
- **LangChain Templates**: Pre-built workflows

### Community Extensions

- **LangGraph-UI**: Visual workflow editor
- **LangGraph-Studio**: Development environment
- **Various adapters**: Database, message queues, etc.

## Learning Resources

- [Official Documentation](https://python.langchain.com/docs/langgraph/)
- [GitHub Repository](https://github.com/langchain-ai/langgraph)
- [Example Gallery](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [Community Discord](https://discord.gg/langchain)

## Migration Guide

### From LangChain Chains

```python
# Old: LangChain chain
chain = prompt | llm | output_parser

# New: LangGraph workflow
def process(state):
    result = chain.invoke(state["input"])
    return {"output": result}
```

### Key Benefits

1. **State persistence**: Keep context across invocations
2. **Complex flows**: Conditional logic, loops, branches
3. **Error recovery**: Built-in error handling patterns
4. **Observability**: Detailed tracing of workflow execution

**LangGraph brings the power of workflow engines to the LLM ecosystem, making complex agentic applications practical to build and maintain.**