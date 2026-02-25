# Prompt Template Architecture

*Prerequisite: [01_Foundations_and_Anatomy.md](01_Foundations_and_Anatomy.md), [02_Programmatic_Prompting.md](02_Programmatic_Prompting.md).*
*Practical: [../../01_LLMs/02_Practical/03_Prompt_Infrastructure.py](../../01_LLMs/02_Practical/03_Prompt_Infrastructure.py)*

---

Prompts sit at the boundary between code and natural language. Once a system has dozens or hundreds of prompts, the core challenge shifts from "how to write a prompt" to "how to organize and manage prompts".

A template system must address four concerns:
1. **Decoupling** — Separate prompt content from business logic.
2. **Variable Injection Safety** — Type-safe, validated variable passing into templates.
3. **Multi-stage Organization** — Orchestrate prompts across roles and lifecycle phases.
4. **Maintainability** — Editable by non-engineers, version-controllable, reusable.

---

## 1. Design Method Spectrum

### 1.1 String Templates (Simplest)

```python
template = "You are {role}. Answer: {question}"
prompt = template.format(role="assistant", question=user_input)
```

`str.format` / f-string. Zero dependencies, suitable for one-off scripts.

Limitation: Cannot handle conditionals, loops, or nested structures. When prompts need to dynamically add/remove sections based on context, the code becomes littered with `if/else` concatenation logic.

### 1.2 Template Engines (Jinja2)

Template languages absorb conditionals, loops, and filters, keeping logic cohesive within the template:

```jinja2
You are a helpful agent with access to these tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

{% if custom_instructions %}
Additional instructions: {{ custom_instructions }}
{% endif %}
```

Key practice — **StrictUndefined**:

```python
env = jinja2.Environment(undefined=jinja2.StrictUndefined)
```

Raises an error when a variable is missing, instead of silently rendering an empty string. This is the "fail fast" principle applied at the template layer.

Trade-off: Templates become "hidden code". Errors occur in the rendering layer, with stack traces pointing into the template engine internals rather than business logic. Debugging is harder.

> Practical advice: Keep complex logic on the Python side; templates should only do simple variable insertion and loop expansion.

### 1.3 Structured Message Templates (LangChain Style)

Instead of treating a prompt as a single string, model it as a sequence of role-tagged messages:

```python
ChatPromptTemplate.from_messages([
    ("system", "You are {role}"),
    ("human",  "{user_input}"),
    ("ai",     "{previous_response}"),  # few-shot
])
```

Core idea: **Roles are first-class citizens.** The prompt structure directly maps to the LLM chat format. The framework handles role mapping, not the template.

This is more robust when switching between LLM APIs — role mapping is handled uniformly by the framework; changing models doesn't require changing templates.

### 1.4 Lifecycle-phased Templates

In Agent systems, prompts are not one-shot. They follow the execution lifecycle in distinct phases.

This pattern is exemplified by **smolagents**, which uses `TypedDict` to enforce structural completeness:

```python
# smolagents design: each lifecycle phase has its own prompt slot
class PromptTemplates(TypedDict):
    system_prompt: str
    planning: PlanningPromptTemplate          # initial_plan / update_plan
    managed_agent: ManagedAgentPromptTemplate  # task / report
    final_answer: FinalAnswerPromptTemplate    # pre/post_messages
```

```
System init   →  system_prompt
Planning      →  planning.initial_plan / update_plan
Execution     →  (driven by code logic)
Sub-agent     →  managed_agent.task / report
Fallback      →  final_answer.pre/post_messages
```

Each phase's prompt is managed independently. This maps directly to the ReAct framework's phase decomposition.

### 1.5 Externalized Storage (YAML/JSON)

Industry consensus: **Prompts are configuration, not code.**

```yaml
system_prompt: |
  You are an expert agent...
planning:
  initial_plan: |
    Here is your task: {{task}}
    Available tools: {{tools}}
```

Benefits:
- Non-engineers can edit prompts directly (no Python required).
- Cleaner version control (readable diffs, not mixed into business logic commits).
- Supports multi-language / multi-scenario switching (by filename or directory).

> Note: Externalized storage is a **storage strategy**, orthogonal to the template methods above. It combines with any of them (e.g., YAML files containing Jinja2 templates, or YAML files containing structured message definitions).

> Full implementation reference: [03_Prompt_Infrastructure.py](../../01_LLMs/02_Practical/03_Prompt_Infrastructure.py) (Jinja2 + YAML + StrictUndefined + PromptManager)

---

## 2. Template Composition & Inheritance

When a system has dozens of Agents, each with multi-phase prompts, single template files are insufficient. Composition and inheritance mechanisms are needed.

### 2.1 Composition

Assemble small templates into larger ones. LangChain approaches:

**Operator composition** — `+` merges `ChatPromptTemplate` instances:

```python
system = ChatPromptTemplate.from_messages([("system", "You are {role}")])
examples = ChatPromptTemplate.from_messages([("human", "Q: {q}"), ("ai", "A: {a}")])

full = system + examples + ChatPromptTemplate.from_messages([("human", "{input}")])
```

**MessagesPlaceholder** — dynamically inject message lists (e.g., chat history):

```python
ChatPromptTemplate.from_messages([
    ("system", "You are {role}"),
    MessagesPlaceholder("chat_history"),  # injected at render time
    ("human", "{input}"),
])
```

This complements the `+` operator: `+` is for static composition at definition time, `MessagesPlaceholder` is for dynamic injection at render time.

**Prefix / examples / suffix** — the classic three-part pattern:

```python
FewShotPromptWithTemplates(
    prefix=system_template,       # system instructions
    example_prompt=example_tmpl,  # format for a single example
    suffix=query_template,        # user input
    example_separator="\n\n"
)
```

Each part is maintained and tested independently, merged only at assembly time.

### 2.2 Inheritance

Define a base template, then override specific parts for specialized scenarios:

```jinja2
{# base_agent.jinja2 #}
{% block system %}You are a helpful assistant.{% endblock %}
{% block tools %}{% endblock %}
{% block constraints %}Be concise.{% endblock %}
```

```jinja2
{# code_review_agent.jinja2 #}
{% extends "base_agent.jinja2" %}
{% block system %}You are a senior code reviewer.{% endblock %}
{% block tools %}
Available tools: {{ tools | join(", ") }}
{% endblock %}
```

> In practice, Jinja2 `extends/block` inheritance is rarely used for LLM prompts directly — prompts are typically flat text, not hierarchical documents. The more common pattern is **Python-level dict merge/override**: define a base config dict and let each scenario override specific keys. The Jinja2 example above illustrates the *concept*; the implementation is usually simpler.

### 2.3 Trade-off: Composition Depth vs. Debuggability

Composition and inheritance enable reuse but introduce indirection:

- Beyond 2–3 layers of composition, the final rendered prompt becomes hard to trace back to its source.
- Debugging requires "unfolding" the entire composition chain to see the complete prompt.
- Recommendation: Stay flat. Prefer composition over inheritance. Provide a `render_debug()` method that outputs the fully resolved prompt.

---

## 3. Template Variable Safety

Template variables are the entry point for external input into prompts. This is not about prompt injection defense (→ see [07_Security](../../07_Security/)), but about the template system's own engineering robustness.

### 3.1 Type Validation

Use Pydantic / TypedDict to constrain the template input schema, catching invalid input before rendering:

```python
from pydantic import BaseModel

class PromptVars(BaseModel):
    role: str
    tools: list[str]
    max_steps: int = 5

# Validate before rendering
vars = PromptVars(role="analyst", tools=["search", "calc"])
prompt = template.render(**vars.model_dump())
```

Benefits: IDE autocompletion, type checking, missing field errors caught early.

### 3.2 User Input Escaping

When user input is passed directly as a template variable, guard against template injection (distinct from LLM prompt injection):

```python
# Safe: Jinja2 does NOT recursively parse template syntax inside variable values
template.render(user_input=raw_user_input)  # {{ }} in raw_user_input won't be interpreted

# Dangerous: if user-provided strings are used to BUILD templates
env.from_string(user_provided_string)  # user can inject arbitrary template logic
```

Jinja2's `{{ variable }}` is safe by default — variable values are not re-parsed. But if code uses `env.from_string(user_provided_string)`, the user can inject arbitrary template logic.

Principle: **User input may only serve as variable values, never as the template itself.**

### 3.3 Defense Layers

`StrictUndefined` solves the "omission" problem, not the "injection" problem:

| Defense Layer | Problem Solved | Mechanism |
| :--- | :--- | :--- |
| StrictUndefined | Missing variables | Raises exception on undefined variables |
| Pydantic schema | Type errors | Validates variable types and constraints before rendering |
| Input escaping | Template injection | User input only as variable values |
| Prompt Guard | LLM-level injection | → [07_Security](../../07_Security/) |

---

## 4. Design Trade-off Matrix

| Dimension | String Template | Jinja2 | Structured Messages | Lifecycle-phased |
| :--- | :--- | :--- | :--- | :--- |
| Expressiveness | Low | High | Medium | High |
| Debuggability | High | Low | High | Medium |
| LLM Format Alignment | Poor | Medium | Good | Good |
| Non-engineer Friendly | Good | Medium | Depends on tooling* | Poor |
| Composition Support | None | Native (extends/block) | Operator (`+`) + Placeholder | TypedDict constraints |
| Best Fit | Scripts / prototypes | Complex dynamic prompts | Multi-turn conversations | Agent systems |

> *Structured messages can be very intuitive with a UI editor (draggable message cards), potentially more so than Jinja2 templates. This dimension depends on tooling, not the method itself.

In practice these methods are **not mutually exclusive**. Common combinations:
- **YAML externalization + Jinja2 rendering + TypedDict structural constraints** — balances flexibility and safety.
- **ChatPromptTemplate + lifecycle-phased management** — standard approach for LangChain/LangGraph Agents.

---

## 5. Industry Trends

### 5.1 Prompt as Code

An increasing number of frameworks treat prompts as compilable, optimizable code artifacts:

- **Guidance / LMQL**: Mix prompts with generation constraints, controlling the generation process at the token level.

```python
# Guidance style
with system():
    lm += "You are a helpful assistant"
with user():
    lm += f"Question: {question}"
with assistant():
    lm += gen("answer", stop=".")  # constrain generation to stop at period
```

- **DSPy**: Write program logic instead of prompts; the framework auto-generates and optimizes prompts (→ see [02_Programmatic_Prompting.md](02_Programmatic_Prompting.md) for details).

### 5.2 Prompt Version Management & Observability

YAML externalization solves the "can edit" problem, but not the "what changed and did it help" problem.

In production, prompts iterate far more frequently than code. Dedicated management tools are emerging:

| Tool | Core Capability |
| :--- | :--- |
| **Langfuse** | Prompt versioning + tracing + evaluation (open source) |
| **PromptLayer** | Prompt registry + A/B testing + analytics |
| **Humanloop** | Prompt editor + evaluation + deployment pipelines |

Core idea: Treat prompts as versioned, evaluated, rollback-capable configuration artifacts — not just strings.

---

## Summary

The choice of template system is fundamentally a trade-off between **manual control** and **automation**:

```
String Template ──→ Jinja2 ──→ Structured Messages ──→ Lifecycle-phased ──→ DSPy Auto-generation
  Full manual control                                                     Automated but opaque
```

A pragmatic middle ground: **TypedDict for structural constraints + Jinja2 for flexibility + YAML for externalization + StrictUndefined for fail-fast + Pydantic for input validation.** This is currently the most balanced choice for Agent scenarios that demand controllability.
