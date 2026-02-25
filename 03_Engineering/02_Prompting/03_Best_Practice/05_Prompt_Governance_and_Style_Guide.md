# Prompt Governance & Style Guide

*Prerequisite: [01_Automated_Prompt_CI_CD.md](01_Automated_Prompt_CI_CD.md).*

---

In a professional engineering team, prompts are **Source Code**. They must be managed with the same rigor as Python or TypeScript. This guide defines the standards for prompt organization, versioning, and documentation.

## 1. Directory Structure & Organization

Prompts should **never** live as raw strings inside application logic.

### Recommended Structure:
```text
/prompts
  /templates
    summarizer.yaml    # Production templates
    classifier.v2.yaml # Versioned templates
  /tests
    test_summarizer.py # Unit tests for prompt rendering
  /golden_sets
    summarizer_eval.json # Ground truth for regression testing
```

## 2. The Prompt Header (Metadata)

Every prompt file should start with a metadata block.
```yaml
prompt_name: "CustomerSupportClassifier"
version: "1.2.3"
author: "PlatformTeam"
created_at: "2026-02-24"
target_models: ["gpt-4o", "claude-3-5-sonnet"]
description: "Classifies incoming support tickets into 5 categories."
```

## 3. Style & Formatting Standards

### 3.1 Use Delimiters
Always use clear delimiters for different sections (Role, Context, Task).
- **XML Tags**: `<context>...</context>` (Highly recommended for Claude).
- **Markdown Headers**: `### Instructions` (Effective for GPT).

### 3.2 Variable Placeholder Convention
Use `{{ variable_name }}` (Jinja2 style) for consistency.
- Avoid single braces `{}` as they clash with JSON strings in many languages.

### 3.3 The "Negative Constraint" section
Group all "DO NOT" instructions in a dedicated section at the bottom of the system prompt to increase compliance.

## 4. Versioning Strategy (Semantic Prompting)

Follow a simplified SemVer for prompts:
- **MAJOR**: Structural change (e.g., changing from JSON output to XML). Requires code changes.
- **MINOR**: Performance optimization or adding new instructions without breaking the output schema.
- **PATCH**: Fixing a typo or a minor edge case failure.

## 5. The "No-Magic" Rule

If a prompt requires "magic words" (e.g., "I will tip you $200", "Take a deep breath"), it must be documented with a **Comment** explaining:
1. Which model this helps.
2. The benchmark improvement seen during testing.
3. Why it was necessary.

*Note: In 2025, modern models (o1, GPT-4o) generally do not need these psychological hacks and respond better to clear logic.*
