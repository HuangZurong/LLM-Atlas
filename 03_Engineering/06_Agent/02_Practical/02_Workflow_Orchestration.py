"""
02_Workflow_Orchestration.py — Agentic Workflow Patterns

Implements three core orchestration patterns from 01_Theory/03_Workflow_Patterns.md:
1. Prompt Chaining (Sequential Pipeline with Gates)
2. Parallelization (Map-Reduce)
3. Evaluator-Optimizer (Self-Correction Loop)

All patterns use a shared LLM client and are composable.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Shared LLM Call
# ---------------------------------------------------------------------------

async def llm_call(
    client: AsyncOpenAI,
    prompt: str,
    system: str = "",
    model: str = "gpt-4o-mini",
    temperature: float = 0,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
    )
    return resp.choices[0].message.content or ""


# ===========================================================================
# Pattern 1: Prompt Chaining (Sequential Pipeline)
# ===========================================================================

@dataclass
class ChainStep:
    """A single step in a prompt chain."""
    name: str
    system_prompt: str
    gate: Callable[[str], bool] | None = None  # Optional quality gate


async def prompt_chain(
    client: AsyncOpenAI,
    steps: list[ChainStep],
    initial_input: str,
    model: str = "gpt-4o-mini",
) -> dict[str, str]:
    """
    Execute steps sequentially. Each step's output becomes the next step's input.
    If a gate fails, the chain halts and returns partial results.
    """
    results = {}
    current = initial_input

    for step in steps:
        output = await llm_call(client, current, system=step.system_prompt, model=model)
        results[step.name] = output

        if step.gate and not step.gate(output):
            results["_halted_at"] = step.name
            break

        current = output

    return results


# ===========================================================================
# Pattern 2: Parallelization (Map-Reduce)
# ===========================================================================

async def map_reduce(
    client: AsyncOpenAI,
    chunks: list[str],
    map_system: str,
    reduce_system: str,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Map: Process each chunk in parallel with the same instruction.
    Reduce: Synthesize all partial results into a final output.
    """
    # Map phase — all chunks processed concurrently
    map_tasks = [
        llm_call(client, chunk, system=map_system, model=model)
        for chunk in chunks
    ]
    partial_results = await asyncio.gather(*map_tasks)

    # Reduce phase — synthesize
    combined = "\n\n---\n\n".join(
        f"[Section {i+1}]\n{r}" for i, r in enumerate(partial_results)
    )
    final = await llm_call(
        client, combined, system=reduce_system, model=model,
    )

    return {
        "partial_results": list(partial_results),
        "final": final,
    }


# ===========================================================================
# Pattern 3: Evaluator-Optimizer (Self-Correction Loop)
# ===========================================================================

@dataclass
class EvalResult:
    passed: bool
    score: float
    feedback: str


async def evaluator_optimizer(
    client: AsyncOpenAI,
    task: str,
    generator_system: str,
    evaluator_system: str,
    max_rounds: int = 3,
    pass_threshold: float = 0.8,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Loop: Generate → Evaluate → Revise until pass or max_rounds.
    The evaluator returns JSON with {score, feedback}.
    """
    history: list[dict] = []
    current_prompt = task

    for round_num in range(1, max_rounds + 1):
        # Generate
        draft = await llm_call(client, current_prompt, system=generator_system, model=model)

        # Evaluate
        eval_prompt = (
            f"## Task\n{task}\n\n## Draft\n{draft}\n\n"
            "Evaluate the draft. Respond in JSON: "
            '{"score": 0.0-1.0, "feedback": "specific improvement suggestions"}'
        )
        eval_raw = await llm_call(client, eval_prompt, system=evaluator_system, model=model)

        try:
            eval_data = json.loads(eval_raw)
            result = EvalResult(
                passed=eval_data["score"] >= pass_threshold,
                score=eval_data["score"],
                feedback=eval_data["feedback"],
            )
        except (json.JSONDecodeError, KeyError):
            result = EvalResult(passed=False, score=0.0, feedback="Evaluator returned invalid JSON")

        history.append({
            "round": round_num,
            "draft": draft,
            "score": result.score,
            "feedback": result.feedback,
        })

        if result.passed:
            return {"final": draft, "rounds": round_num, "history": history}

        # Revise — feed back the evaluation
        current_prompt = (
            f"## Original Task\n{task}\n\n"
            f"## Your Previous Draft\n{draft}\n\n"
            f"## Evaluator Feedback (score: {result.score})\n{result.feedback}\n\n"
            "Please revise your draft to address the feedback."
        )

    return {"final": history[-1]["draft"], "rounds": max_rounds, "history": history}


# ===========================================================================
# Demo
# ===========================================================================

async def main():
    client = AsyncOpenAI()

    # --- Pattern 1: Prompt Chaining ---
    print("=" * 60)
    print("Pattern 1: Prompt Chaining (Extract → Translate → Format)")
    chain_result = await prompt_chain(
        client,
        steps=[
            ChainStep(
                name="extract",
                system_prompt="Extract all named entities (people, places, orgs) from the text. Output as a bullet list.",
            ),
            ChainStep(
                name="translate",
                system_prompt="Translate the following bullet list into Chinese. Keep the bullet format.",
            ),
            ChainStep(
                name="format",
                system_prompt="Convert the bullet list into a markdown table with columns: Entity, Type.",
                gate=lambda out: "|" in out,  # Gate: must contain a table
            ),
        ],
        initial_input="Elon Musk announced that SpaceX will launch a mission to Mars from Cape Canaveral in 2026.",
    )
    for step_name, output in chain_result.items():
        print(f"\n[{step_name}]\n{output}")

    # --- Pattern 2: Map-Reduce ---
    print("\n" + "=" * 60)
    print("Pattern 2: Map-Reduce (Parallel Summarization)")
    chunks = [
        "Chapter 1: The history of artificial intelligence began in the 1950s with Alan Turing's foundational work...",
        "Chapter 2: Machine learning emerged as a subfield, with neural networks gaining popularity in the 1980s...",
        "Chapter 3: The transformer architecture, introduced in 2017, revolutionized NLP and led to modern LLMs...",
    ]
    mr_result = await map_reduce(
        client,
        chunks=chunks,
        map_system="Summarize this chapter in exactly 2 sentences.",
        reduce_system="Synthesize these chapter summaries into a single coherent paragraph of 3-4 sentences.",
    )
    print(f"\nFinal Summary:\n{mr_result['final']}")

    # --- Pattern 3: Evaluator-Optimizer ---
    print("\n" + "=" * 60)
    print("Pattern 3: Evaluator-Optimizer (Self-Correcting Code)")
    eo_result = await evaluator_optimizer(
        client,
        task="Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring case and non-alphanumeric characters.",
        generator_system="You are a Python developer. Write clean, correct code. Output only the function, no explanation.",
        evaluator_system=(
            "You are a code reviewer. Evaluate the code for correctness, edge cases, and style. "
            "Score 0.0-1.0. Deduct for: missing edge cases, poor naming, no type hints."
        ),
        max_rounds=3,
        pass_threshold=0.8,
    )
    print(f"\nFinal (after {eo_result['rounds']} rounds):\n{eo_result['final']}")


if __name__ == "__main__":
    asyncio.run(main())
