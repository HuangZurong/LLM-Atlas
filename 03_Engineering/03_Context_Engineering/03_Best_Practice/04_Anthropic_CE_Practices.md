# 04 · Anthropic Context Engineering Practices

*Compiled from Anthropic's official engineering blog, API documentation, Claude Code best practices, and public research.*

---

## 1. Core Philosophy

### 1.1 Defining Context Engineering

Anthropic defines Context Engineering as the natural successor to Prompt Engineering:

> Building with LLMs is becoming "less about finding the right words and phrases for your prompts, and more about answering the broader question of: **what configuration of context is most likely to generate our model's desired behavior?**"

Context includes everything the model sees before generating a response: system prompts, tools, few-shot examples, message history, retrieved documents, and any other data. Prompt engineering is a subset of context engineering — it is writing a good instruction, whereas context engineering is curating *everything* the AI sees.

The term "context engineering" was coined by Tobi Lutke (CEO of Shopify) on June 19, 2025, and subsequently adopted and formalized by Anthropic.

### 1.2 Guiding Principle

> Given that LLMs are constrained by a finite attention budget, good context engineering means **finding the smallest possible set of high-signal tokens that maximize the likelihood of some desired outcome.**

---

## 2. Context Rot — The Core Problem

### 2.1 Definition

**Context Rot**: as the number of tokens in the context window increases, the model's ability to accurately recall information from that context decreases.

This stems from the transformer architecture itself — every token attends to every other token, creating n² pairwise relationships. As context length increases, the model's "attention budget" gets stretched thin.

### 2.2 Key Data Points

| Finding | Data |
|---------|------|
| Performance ceiling | ~1 million tokens, performance degrades meaningfully past this point |
| Universality | Chroma tested 18 frontier models — **every single one** gets worse as input length increases |
| MRCR v2 benchmark | Claude Opus 4.6 achieved 76% accuracy at 1M tokens |

### 2.3 Implication

Every unnecessary word, every redundant tool description, every piece of stale data actively degrades your agent's performance. More context ≠ better.

---

## 3. System Prompt Design

### 3.1 The "Right Altitude"

Anthropic recommends system prompts at the **Goldilocks zone** between two failure modes:

| Failure Mode | Description |
|-------------|-------------|
| **Too prescriptive (too low)** | Hardcoded complex, brittle logic to elicit exact behavior — creates fragility and maintenance burden |
| **Too vague (too high)** | High-level guidance that fails to give concrete signals, or falsely assumes shared context |

### 3.2 Recommended Structure (Contract Format)

```
- Role (one line)
- Success Criteria (bullets)
- Constraints (bullets)
- Uncertainty Handling Rule
- Output Format Specification
```

### 3.3 Key Mental Model

> Think of Claude as a **brilliant but new employee** who lacks context on your norms and workflows. Be clear and explicit.

### 3.4 Long-Horizon Task Prompting

For agents that use compaction, Anthropic recommends including:

```
Your context window will be automatically compacted as it approaches its limit,
allowing you to continue working indefinitely from where you left off.
Therefore, do not stop tasks early due to token budget concerns.
As you approach your token budget limit, save your current progress and state
to memory before the context window refreshes.
Always be as persistent and autonomous as possible and complete tasks fully.
```

### 3.5 Self-Check Blocks

For important prompts, append a tiny self-check block. Claude will often catch mistakes when explicitly asked to check its own output.

---

## 4. Tool Design — Writing for Agents, Not Developers

### 4.1 Paradigm Shift

Tools are a new kind of software reflecting a **contract between deterministic systems and non-deterministic agents**. Instead of writing tools the way you would write APIs for other developers, you need to design them for agents.

### 4.2 Concrete Best Practices

| Practice | Rationale |
|----------|-----------|
| **Write descriptions in third person** | The description is injected into the system prompt; inconsistent POV causes discovery problems |
| **Be specific and include triggers** | Include both what the tool does AND specific triggers/contexts for when to use it |
| **Namespace and differentiate tools** | Clear, distinct names help agents avoid confusion when choosing from 100+ tools |
| **Return meaningful context** | Human-readable fields and simplified outputs > raw technical IDs |
| **Optimize for token efficiency** | Implement pagination, truncation, filtering; **keep responses under 25,000 tokens** |
| **Provide input examples** | Especially for complex tools with nested objects, optional parameters, or format-sensitive inputs |
| **Curate a minimal viable tool set** | If a human engineer can't definitively say which tool to use, an AI agent can't either |

### 4.3 Impact of Tool Descriptions

Even small refinements to tool descriptions can yield dramatic improvements. Claude Sonnet 3.5 achieved SOTA on SWE-bench Verified after precise refinements to tool descriptions.

**Real-world example**: When Anthropic launched Claude's web search tool, they discovered Claude was needlessly appending "2025" to search queries, biasing results. Fixed by improving the tool description.

### 4.4 Evaluation-Driven Tool Improvement

> "Simply concatenate the transcripts from your evaluation agents and paste them into Claude Code. Claude is an expert at analyzing transcripts and refactoring lots of tools all at once."

---

## 5. Few-Shot Examples

Anthropic strongly advises few-shot prompting but warns against a common failure mode: **stuffing a laundry list of edge cases into a prompt** in an attempt to articulate every possible rule.

Recommendation: Curate a set of **diverse, canonical examples** that effectively portray expected behavior. **Diversity and representativeness > exhaustive coverage.**

---

## 6. Long Context Prompting Tips

| Technique | Description |
|-----------|-------------|
| **Put longform data at the top** | Place 20K+ token documents near the top, above queries/instructions/examples |
| **Place queries at the end** | Queries at the end improve response quality by up to **30%**, especially with complex multi-document inputs |
| **Structure with XML tags** | Wrap each document in `<document>` tags with `<document_content>` and `<source>` subtags |
| **Ground responses in quotes** | For long document tasks, ask Claude to quote relevant parts first before carrying out its task |

---

## 7. Prompt Caching

### 7.1 Impact

Cost reduction up to **90%**, latency reduction up to **85%** for long prompts.

### 7.2 Two Approaches

| Approach | Description |
|----------|-------------|
| **Automatic (recommended)** | Add `cache_control={"type": "ephemeral"}` at the top level; system handles the rest |
| **Explicit breakpoints** | Place `cache_control` on individual content blocks for fine-grained control |

### 7.3 Key Rules

- Cache prefixes are created in order: **tools → system → messages**
- Up to **4 cache breakpoints** can be defined
- **Cache keys are cumulative**: the hash for each block depends on ALL content before it
- **Minimum token requirement**: at least 1,024 tokens per cache checkpoint
- **TTL**: 5-minute default, resets with each cache hit. Claude Opus 4.5/Haiku 4.5/Sonnet 4.5 also support 1-hour TTL
- **20-block lookback window**: after 20 checks without a match, the system stops looking

### 7.4 Multi-Turn Best Practice

Always set an explicit cache breakpoint at the end of your conversation to maximize cache hits. Multi-turn caching incrementally caches conversation history.

---

## 8. Compaction — The First Lever for Long-Horizon Tasks

### 8.1 Definition

Compaction is the practice of taking a conversation nearing the context window limit, summarizing its contents, and reinitiating a new context window with the summary.

### 8.2 How Claude Code Implements Compaction

- Passes message history to the model to summarize and compress critical details
- **Preserves**: architectural decisions, unresolved bugs, implementation details
- **Discards**: redundant tool outputs or messages
- Agent continues with compressed context **plus the five most recently accessed files**

### 8.3 Tuning Compaction

- The art lies in selection of what to keep vs. discard
- **Start by maximizing recall** (capture every relevant piece of information), then **iterate to improve precision** (eliminate superfluous content)
- Tool result clearing is a safe, lightweight form of compaction

### 8.4 Complementary Practices

| Practice | Description |
|----------|-------------|
| **Git commits as checkpoints** | Commit progress with descriptive messages so the agent can use `git log` and `git diff` to reconstruct state |
| **Progress files** | Write summaries to `claude-progress.txt` that the agent reads after compaction to understand completed/in-progress/blocked work |

### 8.5 Server-Side Compaction API

Available in beta (`compact-2026-01-12`). When enabled, Claude automatically summarizes the conversation when approaching the configured token threshold, generates a summary, creates a compaction block, and drops all prior message blocks.

### 8.6 Customizing Compaction

In `CLAUDE.md`, add instructions like "When compacting, always preserve the full list of modified files and any test commands" to ensure critical context survives.

---

## 9. Context Editing (Beta)

### 9.1 Tool Result Clearing (`clear_tool_uses_20250919`)

- Clears tool results when conversation context grows beyond a threshold
- Configurable: trigger point, number of recent tool uses to keep, minimum tokens to clear, tools to exempt
- Cleared content is replaced with placeholder text so the model knows something was removed

### 9.2 Thinking Block Clearing (`clear_thinking_20251015`)

- Removes thinking/reasoning blocks from earlier turns, keeping only the most recent ones
- Configure the `keep` parameter based on whether you prioritize cache performance or context window availability

### 9.3 Performance Impact

In a 100-turn web search evaluation:
- Context editing enabled agents to complete workflows that would otherwise fail due to context exhaustion
- **Token consumption reduced by 84%**
- Combining memory tool with context editing improved performance by **39% over baseline**

---

## 10. Just-in-Time Context Loading

### 10.1 Strategy

Instead of loading everything upfront, agents maintain lightweight references and dynamically load data at runtime.

### 10.2 Implementation Pattern

- Maintain **lightweight identifiers** (file paths, stored queries, web links, API endpoints)
- Use these references to **dynamically load data into context at runtime** using tools
- Claude Code's tool lazy loading reduces context by **95%** by not loading tool definitions until needed

### 10.3 Pre-Loading vs. Just-in-Time

| Approach | Characteristics |
|----------|----------------|
| **Pre-loading** | Pre-processes all relevant data upfront; context becomes bloated |
| **Just-in-Time** | Loads only what is needed at the moment of use; keeps context lean and focused |

---

## 11. Multi-Agent Architecture — Context Isolation

### 11.1 Orchestrator-Worker Pattern

- A **lead agent** (orchestrator) maintains high-level strategy and user intent
- **Subagents** (workers) each run in isolated context windows, receive specific directives, execute tasks, and report back only summaries or results
- Each agent accesses only the tools and context needed for its specific role

### 11.2 Performance Data

- Multi-agent research system (Opus 4 lead + Sonnet 4 subagents) outperformed a single Opus 4 agent by **90.2%** on research tasks
- **Token usage alone explains 80% of performance variance** in complex evaluations

### 11.3 Delegation Best Practices

| Practice | Description |
|----------|-------------|
| **Specific objectives** | Allocate queries into subtasks with specific objectives, output formats, tool guidance, and clear boundaries |
| **Artifact-based communication** | Subagents store outputs in external systems and pass lightweight references back, preventing information loss and reducing token overhead |
| **Tool restriction for sandboxing** | Remove tools like Edit to create a sandbox where the agent cannot break the build |

### 11.4 Model Routing for Cost

Multi-agent systems consume approximately **15x more tokens** than single-agent interactions. Use model routing:

| Model | Use Case |
|-------|----------|
| **Haiku** | Linting and simple logic |
| **Sonnet** | Majority of coding and debugging |
| **Opus** | Orchestrator or complex architectural reasoning |

### 11.5 When to Use Multi-Agent

| Good Fit | Poor Fit |
|----------|----------|
| Heavy parallelization | Domains requiring shared context among all agents |
| Information exceeding single context windows | Many dependencies between agents |
| Interfacing with multiple complex tools | |

---

## 12. Long-Running Agent Harnesses

### 12.1 Initializer + Coding Agent Pattern

| Agent | Responsibility |
|-------|---------------|
| **Initializer** | Runs on the first context window only; sets up all necessary context for future coding agents |
| **Coding Agent** | Runs on every subsequent context window; makes incremental progress and leaves clear artifacts |

### 12.2 The `claude-progress.txt` Pattern

External artifacts become the agent's memory:

- **Progress files** persist across sessions. Each session opens by reading these artifacts to recover full state
- **Git history** provides a log of what has been done and checkpoints that can be restored
- **Feature checklists** track what was completed, in progress, or blocked

### 12.3 End-of-Session Protocol

Before a session ends, it updates the progress log with what was completed and what remains, ensuring the next session has an accurate starting point.

### 12.4 Stress Test: Building a C Compiler

Anthropic validated this approach by tasking 16 agents with writing a Rust-based C compiler from scratch, capable of compiling the Linux kernel. Over nearly **2,000 Claude Code sessions** and **$20,000 in API costs**, the agents produced a 100,000-line compiler that can build Linux 6.9 on x86, ARM, and RISC-V.

---

## 13. CLAUDE.md and Auto Memory

### 13.1 CLAUDE.md Files

Persistent markdown instruction files read at the start of every Claude Code session. Organized hierarchically with cascading precedence (project-level overrides global user-level).

**Best Practices**:

| Practice | Rationale |
|----------|-----------|
| Keep under 200 lines per file | Longer files consume more context and reduce adherence |
| Use markdown headers and bullets | Group related instructions |
| Be specific enough to verify | "Use 2-space indentation" not "Format code properly" |
| Use `.claude/rules/` for splitting | Split large instruction sets into multiple files |
| Use `CLAUDE.local.md` for private settings | Auto-added to .gitignore |

### 13.2 Auto Memory (MEMORY.md)

Claude Code automatically creates and maintains a `MEMORY.md` file, capturing:

- Debugging patterns (recurring issues, root causes, fixes)
- Project context (architecture decisions, naming conventions, gotchas)
- Preferences (code structure, solution preferences, testing approach)

### 13.3 "Fading Memory" Pitfall

As CLAUDE.md files grow larger and more monolithic, the model's ability to pinpoint the most relevant information diminishes — the signal gets lost in the noise. Keep memory files concise and well-structured.

---

## 14. Strategy Hierarchy Summary

Context management strategies from simplest to most complex:

```
Level  1: Write clear system prompts at the "right altitude"
Level  2: Curate minimal tool sets with excellent descriptions
Level  3: Few-shot examples — diverse and canonical, not exhaustive
Level  4: Place long documents at the top, queries at the bottom
Level  5: Enable Prompt Caching
Level  6: Implement Context Editing (tool result clearing, thinking block clearing)
Level  7: Implement Compaction with git checkpoints and progress files
Level  8: Use Just-in-Time Context Loading instead of upfront pre-loading
Level  9: Adopt multi-agent architectures for context isolation
Level 10: Build agent harnesses with Initializer/Coding Agent patterns
```

Overarching advice: **"Do the simplest thing that works."** Only increase complexity when needed. Start simple, measure, and iterate.

---

## References

- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — Anthropic Engineering Blog (Sep 2025)
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — Anthropic Engineering Blog (2026)
- [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents) — Anthropic Engineering Blog
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — Anthropic Research
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — Anthropic Engineering Blog
- [Prompting best practices (Claude 4)](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices) — Claude API Docs
- [Long context prompting tips](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) — Claude API Docs
- [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — Claude Platform Docs
- [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) — Claude Platform Docs
- [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) — Claude Platform Docs
- [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction) — Claude Platform Docs
- [How Claude remembers your project](https://code.claude.com/docs/en/memory) — Claude Code Docs
- [Claude Code best practices](https://www.anthropic.com/engineering/claude-code-best-practices) — Anthropic Engineering Blog
