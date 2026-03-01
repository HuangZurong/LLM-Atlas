# smolagents — Agent Design

## 1. Overview

smolagents is a minimalist agent framework by HuggingFace. Its core philosophy is:

> **The LLM is the reasoning engine. The agent is the loop around it.**

Everything else — tools, memory, models, executors — is pluggable infrastructure that serves this loop.

```
┌─────────────────────────────────────────────────────────────┐
│                        smolagents                           │
│                                                             │
│   User Task ──► MultiStepAgent (ReAct loop)                 │
│                       │                                     │
│          ┌────────────┴────────────┐                        │
│          ▼                         ▼                        │
│   ToolCallingAgent           CodeAgent                      │
│   (JSON tool calls)          (Python code)                  │
│          │                         │                        │
│          └────────────┬────────────┘                        │
│                       ▼                                     │
│                 Model Backend                               │
│          (OpenAI / HF / LiteLLM / local …)                  │
│                       │                                     │
│                       ▼                                     │
│                    Tools / Managed Agents                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Class Hierarchy

```
MultiStepAgent  (ABC — agents.py)
├── ToolCallingAgent
└── CodeAgent
```

`MultiStepAgent` is the abstract base. It owns the **run loop**, **memory**, **callbacks**, and **planning**. The two concrete subclasses differ only in how they express one step of action:

| Class              | How the LLM expresses an action       |
|--------------------|---------------------------------------|
| `ToolCallingAgent` | Structured JSON / function-call spec  |
| `CodeAgent`        | Valid Python code in a fenced block   |

Both classes implement the abstract method `_step_stream()` to define their per-step behavior.

---

## 3. The ReAct Loop

Both agents run the same outer loop defined in `MultiStepAgent._run_stream()`:

```
agent.run(task)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                  ReAct Loop                         │
│                                                     │
│   ┌─────────────────────────────────────────────┐   │
│   │  [Optional] Planning Step                   │   │
│   │  · LLM writes a high-level plan             │   │
│   │  · Stored in memory as PlanningStep         │   │
│   └──────────────────┬──────────────────────────┘   │
│                      │                              │
│   ┌──────────────────▼──────────────────────────┐   │
│   │  Action Step  (repeated up to max_steps)    │   │
│   │                                             │   │
│   │  1. THINK  – LLM reads memory + task        │   │
│   │             and produces an action          │   │
│   │                                             │   │
│   │  2. ACT    – framework parses and           │   │
│   │             executes the action             │   │
│   │                                             │   │
│   │  3. OBSERVE – result (or error) is          │   │
│   │             written back to memory          │   │
│   │                                             │   │
│   │  If action == final_answer ──► EXIT LOOP    │   │
│   └──────────────────┬──────────────────────────┘   │
│                      │                              │
│              step_number += 1                       │
│              repeat until done or max_steps         │
└─────────────────────────────────────────────────────┘
      │
      ▼
  Return final answer  (or RunResult if return_full_result=True)
```

If `max_steps` is reached without a final answer, the agent makes one last `provide_final_answer()` call to the LLM using the accumulated memory.

---

## 4. Two Action Paradigms

### 4a. ToolCallingAgent

The LLM uses the provider's native **function-calling / tool-use** API.

```
LLM output (ChatMessage.tool_calls)
      │
      ▼
  [{"id": "c1",
    "function": {
      "name": "get_weather",
      "arguments": {"location": "Paris"}
    }}]
      │
      ▼
 execute_tool_call("get_weather", {"location": "Paris"})
      │
      ▼
 Observation stored in memory
```

Multiple tool calls in one step are executed **in parallel** via `ThreadPoolExecutor`.

**Best for:** APIs that expose tool-calling natively (OpenAI, Anthropic, etc.).

---

### 4b. CodeAgent

The LLM outputs a **Python code block**. The framework parses and executes it.

```
LLM output (text)
      │
      ▼
  "I need to check the weather.
   <code>
   result = get_weather(location='Paris')
   final_answer(result)
   </code>"
      │
      ▼
 parse_code_blobs()  →  "result = get_weather(...)\nfinal_answer(result)"
      │
      ▼
 PythonExecutor(code_action)
      │
      ├── LocalPythonExecutor   (sandboxed in-process)
      ├── E2BExecutor           (cloud sandbox)
      ├── DockerExecutor        (container)
      ├── ModalExecutor         (serverless)
      ├── BlaxelExecutor        (Blaxel cloud)
      └── WasmExecutor          (WebAssembly)
      │
      ▼
 CodeOutput { logs, output, is_final_answer }
      │
      ▼
 Observation stored in memory
```

Code can contain **loops, conditionals, and multi-step logic in a single step**, which makes `CodeAgent` more powerful for complex reasoning.

**Best for:** tasks where the LLM benefits from expressing multi-step logic in a single action (data processing, arithmetic pipelines, compound searches).

---

## 5. Memory System

Memory converts the agent's history into the message list sent to the LLM at each step.

```
AgentMemory
├── SystemPromptStep        →  [SYSTEM] system prompt
├── TaskStep                →  [USER]   "New task: ..."
├── PlanningStep (optional) →  [ASSISTANT] plan text
│                              [USER] "Now proceed..."
├── ActionStep (×N)
│   ├── model_output        →  [ASSISTANT] LLM's raw output
│   ├── tool_calls          →  [TOOL_CALL] what was called
│   └── observations        →  [TOOL_RESPONSE] "Observation: ..."
└── FinalAnswerStep         →  (not serialized to messages; returned to user)
```

**summary_mode=True** strips `PlanningStep` entries before sending to the LLM for plan updates — this prevents the model from being anchored to the old plan.

The full memory can be exported via `memory.get_full_steps()` and replayed via `agent.replay()`.

---

## 6. Model Backends

All models implement the abstract `Model` base class with a uniform interface:

```
Model  (ABC)
├── Local / on-device
│   ├── TransformersModel   (HuggingFace Transformers)
│   ├── VLLMModel           (vLLM server)
│   └── MLXModel            (Apple Silicon)
│
└── ApiModel
    ├── InferenceClientModel  (HuggingFace Inference API)
    ├── OpenAIModel           (OpenAI / compatible)
    ├── AzureOpenAIModel      (Azure OpenAI)
    ├── LiteLLMModel          (100+ providers via LiteLLM)
    ├── LiteLLMRouterModel    (load-balanced LiteLLM)
    └── AmazonBedrockModel    (AWS Bedrock)
```

The uniform interface provides:
- `model.generate(messages, stop_sequences, tools_to_call_from)` → `ChatMessage`
- `model.generate_stream(...)` → `Generator[ChatMessageStreamDelta]`

This means **any agent can be paired with any model** with no agent-side changes.

---

## 7. Tool System

```
BaseTool  (ABC)
└── Tool
    ├── @tool decorator   (wraps a plain Python function)
    ├── Tool.from_hub()   (loads from HuggingFace Hub)
    └── Tool.from_code()  (loads from source string)
```

A tool exposes:
- `name` — identifier used by the LLM
- `description` — what it does (from docstring)
- `inputs` — parameter schema (derived from type hints + docstring)
- `output_type` — return type
- `forward()` — the actual Python function

The `@tool` decorator is the simplest way to define a tool:

```python
@tool
def get_weather(location: str) -> str:
    """
    Returns the weather at a given location.

    Args:
        location: the city name
    """
    ...
```

The docstring **is** the tool's API contract to the LLM. It drives both the system prompt and JSON schema generation.

---

## 8. Multi-Agent Architecture

Any `MultiStepAgent` can act as a **managed agent** inside another agent. From the orchestrator's perspective, a managed agent is just another tool.

```
Orchestrator (CodeAgent or ToolCallingAgent)
│   tools: [search_tool, managed_agent_A, managed_agent_B]
│
├── managed_agent_A  (e.g. CodeAgent — data analysis specialist)
│       tools: [python_interpreter, file_reader]
│
└── managed_agent_B  (e.g. ToolCallingAgent — web search specialist)
        tools: [web_search, scraper]
```

When the orchestrator calls a managed agent:
1. The task is wrapped in a `managed_agent.task` prompt template.
2. The managed agent runs its own full ReAct loop.
3. The result is returned to the orchestrator as a string observation.

The managed agent's `name` and `description` fields are mandatory — they are what the orchestrator's LLM reads to decide when to call it.

---

## 9. Planning (Optional)

Controlled by `planning_interval: int`:

```
planning_interval = 3  means:

Step 1  ──► Planning Step (initial plan)
Step 2  ──► Action Step
Step 3  ──► Action Step
Step 4  ──► Planning Step (updated plan, summary_mode=True)
Step 5  ──► Action Step
Step 6  ──► Action Step
Step 7  ──► Planning Step (updated plan)
...
```

The plan is injected into memory as text. On update, `summary_mode=True` suppresses previous planning messages to avoid anchoring the LLM to stale plans.

---

## 10. Callback & Monitoring

```
MultiStepAgent
└── step_callbacks: CallbackRegistry
        ├── ActionStep  callbacks  (e.g. Monitor.update_metrics)
        ├── PlanningStep callbacks
        └── FinalAnswerStep callbacks
```

`Monitor` tracks token usage and timing per step, aggregated into `RunResult.token_usage` and `RunResult.timing`.

Custom callbacks can be registered as:
```python
# All ActionSteps
agent = CodeAgent(tools=[...], model=model, step_callbacks=[my_fn])

# Specific step types
agent = CodeAgent(tools=[...], model=model, step_callbacks={
    ActionStep: [fn_a, fn_b],
    PlanningStep: [fn_c],
})
```

---

## 11. Execution Flow Summary

```
agent.run("What is the weather in Paris?")
   │
   ├── reset memory
   ├── log task
   │
   └── _run_stream()
         │
         ├── [if planning_interval] _generate_planning_step()
         │         └── LLM call → PlanningStep → append to memory
         │
         ├── ActionStep #1
         │     ├── write_memory_to_messages()  → full message list
         │     ├── _step_stream()
         │     │     ├── [ToolCallingAgent] model.generate(..., tools_to_call_from)
         │     │     │     └── parse tool_calls → execute_tool_call() → observation
         │     │     └── [CodeAgent] model.generate(...)
         │     │           └── parse_code_blobs() → python_executor() → observation
         │     ├── store ActionStep in memory
         │     └── yield ActionStep
         │
         ├── ActionStep #2 … (repeat)
         │
         └── FinalAnswerStep
               └── yield final answer to caller
```

---

## 12. Key Design Decisions

| Decision | Rationale |
|---|---|
| Only 2 agent classes | Simplicity. Most tasks fit into "call a tool" or "write code". |
| Code-as-action (CodeAgent) | Python is expressive enough to compose multi-step logic in a single action, reducing round trips. |
| Docstring-driven tools | No separate schema files. The tool definition lives with its code. |
| Pluggable executors | CodeAgent can run locally for speed or remotely for isolation/safety. |
| Memory as message list | Stateless LLM calls; the full context is reconstructed each step. |
| Managed agents as tools | Composability without a new abstraction. Hierarchical agents fall naturally from tool-calling. |
| `@tool` decorator | Zero-boilerplate tool creation for simple functions. |
