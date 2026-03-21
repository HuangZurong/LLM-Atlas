# 07 · Cursor Context Engineering Practices

*Compiled from Cursor/Anysphere blog posts, documentation, founder interviews (Lex Fridman, Latent Space podcast), and observable behavior analysis.*

---

## 1. Core Philosophy

### 1.1 Better Selection Beats Larger Windows

Cursor's central thesis:

> **Better context selection beats larger context windows.** Even with 128K or 200K token windows, stuffing everything in degrades quality. The hard problem is choosing *what* to include.

The Cursor team views context engineering as their core differentiator — not which LLM they use, but how they assemble context for it.

### 1.2 Tiered Context Architecture

Cursor organizes context into four tiers with decreasing priority:

```
Tier 1: Immediate Context   (always loaded, highest priority)
Tier 2: Local Context        (high priority, recently active)
Tier 3: Global Context       (retrieved on demand)
Tier 4: Static Rules         (project-level instructions)
```

---

## 2. Codebase Indexing and Embedding

### 2.1 Indexing Pipeline

When a workspace is opened, Cursor indexes the entire codebase:

| Step | Description |
|------|-------------|
| **Chunking** | Files are split into semantically meaningful segments respecting code structure (functions, classes, logical blocks) — not fixed-size windows |
| **Embedding** | Each chunk is embedded using custom-trained embedding models optimized for code (not off-the-shelf models) |
| **Storage** | Embeddings stored in a local vector database on the user's machine |
| **Updates** | Index is built asynchronously in the background and updated incrementally as files change |

### 2.2 Filtering

Files matching `.gitignore` patterns, binary files, `node_modules`, build artifacts, etc. are excluded from indexing entirely.

---

## 3. Context Selection for Chat

When a user asks a question or requests a code change in Cursor's chat:

### 3.1 Explicit Context via @ Symbols

Users can manually attach context using references:

| Reference | Description |
|-----------|-------------|
| `@file` | Pin a specific file into context |
| `@folder` | Include an entire folder |
| `@symbol` | Include a specific function/class/type |
| `@web` | Search the web for information |
| `@docs` | Search documentation |
| `@codebase` | Trigger full codebase retrieval |
| `@git` | Include git history/diff information |

### 3.2 Automatic Codebase Retrieval (`@codebase`)

When `@codebase` is used (or automatic codebase context is enabled), Cursor performs a multi-step retrieval pipeline:

```
Step 1: Embed the user's query using the same embedding model
    ↓
Step 2: Vector similarity search → retrieve top-N candidate chunks
    ↓
Step 3: Reranking with cross-encoder model → re-score candidates
    ↓
Step 4: Assemble top-ranked chunks into prompt within token budget
```

The **reranking step is critical** — initial embedding retrieval has decent recall but mediocre precision; the cross-encoder reranker dramatically improves which chunks actually make it into the context.

### 3.3 Priority Boosting

- Recently edited files and open tabs are given priority/boosted in retrieval
- The current file and cursor position are always included as high-priority context

---

## 4. Tab Completion Context

Tab completion is Cursor's most latency-sensitive feature. The context engineering is fundamentally different from chat.

### 4.1 Custom Model

Cursor trains and serves their own small, fast model for tab completions (not GPT-4 or Claude). Speed is paramount — completions need to arrive in under ~200-300ms.

### 4.2 Smaller Context Window

The tab completion model uses a much smaller context window than chat models. Every token counts.

### 4.3 Context Assembly

| Source | Description |
|--------|-------------|
| **Current file around cursor** | A window of code above and below the cursor position |
| **Recent edits** | Recently edited lines/regions in the current file |
| **Other recent files** | Snippets from recently visited/edited files |
| **Imports and types** | Import statements and type definitions relevant to current scope |
| **File metadata** | Language identifier and file path |

### 4.4 Fill-in-the-Middle (FIM)

Tab completion uses a FIM format where the model sees code before and after the cursor and predicts what goes in between:

```
<prefix>code above cursor</prefix>
<suffix>code below cursor</suffix>
<middle>← model predicts this</middle>
```

### 4.5 Speculative Edits

Cursor pre-computes likely next edits based on recent changes and caches them, so when you press Tab, the suggestion appears instantly. This is a form of **predictive context pre-loading**.

---

## 5. Multi-File Editing Context (Composer / Agent Mode)

### 5.1 Dependency Graph Awareness

Cursor analyzes imports, references, and call sites to understand which files are related. When editing one file, it pulls in files that import from or are imported by the current file.

### 5.2 Agent Mode Tool Use

In agent mode, Cursor gives the LLM tools to search the codebase, read files, run terminal commands, etc. The model iteratively gathers context as needed rather than having everything upfront — a **tool-use / agentic pattern** rather than pure RAG.

### 5.3 Diff-Based Context

For multi-file edits, Cursor shows the model the diffs/changes it has already made in other files, so the model maintains consistency across files.

### 5.4 Apply Model

Cursor uses a separate, fast "apply" model to take the LLM's suggested changes and merge them into the actual file. This is a distinct model from the one generating the edit suggestions — it handles the mechanical task of producing a clean diff.

---

## 6. Token Budget Management

### 6.1 Budget Allocation

Cursor allocates a "token budget" for different context sources:

| Source | Priority | Budget Share |
|--------|----------|-------------|
| Current file | Highest | Largest allocation |
| Explicit @-references | High | Guaranteed inclusion |
| Auto-retrieved codebase chunks | Medium | Fills remaining budget |
| Conversation history | Lower | Trimmed when needed |

### 6.2 Truncation with Priority

When context exceeds the budget, lower-priority items are truncated first:

```
1. Auto-retrieved chunks (lowest priority, cut first)
2. Older conversation history
3. Less relevant open tabs
4. Current file (highest priority, never cut)
5. Explicit @-references (guaranteed, never cut)
```

---

## 7. Cursor Rules (.cursorrules)

### 7.1 Project-Level Rules

Cursor supports project-level rules files (`.cursorrules` or files in `.cursor/rules/`) that inject persistent instructions into every prompt:

- Project-specific conventions
- Tech stack details
- Coding patterns and preferences
- Architecture decisions

### 7.2 Implementation

Rules are prepended to the system prompt or injected as high-priority context. This is analogous to Anthropic's `CLAUDE.md` pattern.

---

## 8. Architecture Summary

```
User Action (tab / chat / composer)
    │
    ▼
Context Assembly Pipeline
    ├── Current file + cursor position          (Tier 1: Immediate)
    ├── Recent edits / open tabs                (Tier 2: Local)
    ├── Explicit @ references                   (Tier 2: Local)
    ├── Vector retrieval from codebase index    (Tier 3: Global)
    ├── Reranking (cross-encoder)               (Tier 3: Global)
    ├── Token budgeting & truncation
    └── .cursorrules injection                  (Tier 4: Static)
    │
    ▼
Prompt Construction
    ├── System prompt (with rules)
    ├── Retrieved context chunks
    ├── Conversation history (for chat)
    └── User query / current code state
    │
    ▼
Model Inference
    ├── Tab:   Custom fast model (FIM format)
    ├── Chat:  Claude / GPT-4 / etc.
    └── Apply: Custom fast apply model
    │
    ▼
Response Processing
    ├── Tab:      Inline suggestion
    ├── Chat:     Streamed response
    └── Composer: Diff generation + apply model
```

---

## 9. Key Insights for CE Framework Design

| Insight | Description |
|---------|-------------|
| **Two-stage retrieval** | Embedding recall → cross-encoder rerank is far more effective than embedding alone |
| **Tiered priority** | Not all context is equal; explicit references > auto-retrieved > history |
| **Feature-specific context** | Tab completion, chat, and composer each have fundamentally different context needs |
| **Custom models for speed** | Use small, fast models for latency-sensitive context tasks (tab completion, apply) |
| **Speculative pre-loading** | Pre-compute likely next contexts for instant delivery |
| **AST-aware chunking** | Code-specific chunking at logical boundaries (functions, classes) > fixed-size windows |
| **Environment as context source** | Terminal output, linter errors, and IDE signals are implicit context |
| **Client-side assembly** | Significant context engineering happens client-side before sending to the model |

---

## References

- [Cursor Blog — Codebase Indexing](https://cursor.com/blog)
- [Cursor Blog — Speculative Edits](https://cursor.com/blog)
- [Cursor Blog — Custom Models](https://cursor.com/blog)
- [Cursor Documentation](https://docs.cursor.com)
- Lex Fridman Podcast — Interview with Anysphere founders
- Latent Space Podcast — Cursor team interviews
- Various technical analyses and reverse-engineering blog posts
