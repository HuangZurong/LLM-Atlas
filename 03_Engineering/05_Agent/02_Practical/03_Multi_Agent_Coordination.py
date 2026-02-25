"""
03_Multi_Agent_Coordination.py — Orchestrator-Workers Pattern

Implements a hierarchical multi-agent system where:
- An Orchestrator (powerful model) decomposes tasks and delegates to Workers
- Workers (cheap models) execute specialized sub-tasks
- A Synthesizer merges worker outputs into a final deliverable

This mirrors production patterns used in Claude Code, Devin, and similar systems.

Key patterns:
- Orchestrator dynamically plans (not hardcoded steps)
- Workers are specialized via system prompts
- Handoff protocol with structured task/result messages
- Budget guard (max total LLM calls across all agents)
"""

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum

from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Worker Agent
# ---------------------------------------------------------------------------

class WorkerRole(str, Enum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"


WORKER_PROMPTS: dict[WorkerRole, str] = {
    WorkerRole.RESEARCHER: (
        "You are a research specialist. Given a research task, provide factual, "
        "well-sourced findings. Be thorough but concise. Output structured bullet points."
    ),
    WorkerRole.ANALYST: (
        "You are a data analyst. Given data or findings, identify patterns, "
        "draw conclusions, and highlight risks. Use quantitative reasoning where possible."
    ),
    WorkerRole.WRITER: (
        "You are a professional writer. Given an outline and supporting data, "
        "produce polished, well-structured prose. Match the requested tone and format."
    ),
    WorkerRole.CRITIC: (
        "You are a critical reviewer. Evaluate the given content for factual accuracy, "
        "logical gaps, and clarity. Provide specific, actionable feedback."
    ),
}


@dataclass
class TaskResult:
    worker: str
    task: str
    output: str


async def run_worker(
    client: AsyncOpenAI,
    role: WorkerRole,
    task: str,
    model: str = "gpt-4o-mini",
) -> TaskResult:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": WORKER_PROMPTS[role]},
            {"role": "user", "content": task},
        ],
        temperature=0,
    )
    return TaskResult(
        worker=role.value,
        task=task,
        output=resp.choices[0].message.content or "",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM = """\
You are a project orchestrator. Given a user request, you must:
1. Decompose it into sub-tasks for specialist workers.
2. Specify which worker handles each task and whether tasks can run in parallel.

Available workers: researcher, analyst, writer, critic

Respond in JSON:
{
  "plan": [
    {"step": 1, "worker": "researcher", "task": "...", "depends_on": []},
    {"step": 2, "worker": "analyst", "task": "...", "depends_on": [1]},
    {"step": 3, "worker": "writer", "task": "...", "depends_on": [1, 2]}
  ]
}

Rules:
- Use the fewest steps necessary.
- Mark dependencies so independent tasks can run in parallel.
- The final step should produce the user-facing deliverable.
"""

SYNTHESIS_SYSTEM = """\
You are a synthesis agent. Given the original request and all worker outputs,
produce the final polished deliverable. Integrate all findings coherently.
Do not add information beyond what the workers provided.
"""


@dataclass
class Orchestrator:
    client: AsyncOpenAI
    orchestrator_model: str = "gpt-4o"
    worker_model: str = "gpt-4o-mini"
    max_calls: int = 15
    _call_count: int = field(default=0, init=False)

    async def run(self, user_request: str) -> str:
        # Step 1: Plan
        plan = await self._plan(user_request)
        if not plan:
            return "[Orchestrator failed to produce a valid plan]"

        # Step 2: Execute plan respecting dependencies
        results = await self._execute_plan(plan)

        # Step 3: Synthesize
        return await self._synthesize(user_request, results)

    async def _plan(self, request: str) -> list[dict] | None:
        self._call_count += 1
        resp = await self.client.chat.completions.create(
            model=self.orchestrator_model,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {"role": "user", "content": request},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            return data.get("plan", [])
        except json.JSONDecodeError:
            return None

    async def _execute_plan(self, plan: list[dict]) -> dict[int, TaskResult]:
        results: dict[int, TaskResult] = {}
        completed: set[int] = set()

        while len(completed) < len(plan):
            # Find steps whose dependencies are all satisfied
            ready = [
                s for s in plan
                if s["step"] not in completed
                and all(d in completed for d in s.get("depends_on", []))
            ]
            if not ready:
                break  # Deadlock guard

            # Run ready steps in parallel
            tasks = []
            for step in ready:
                if self._call_count >= self.max_calls:
                    results[step["step"]] = TaskResult(
                        worker=step["worker"], task=step["task"],
                        output="[Skipped: budget exhausted]",
                    )
                    completed.add(step["step"])
                    continue

                # Inject prior results into the task context
                dep_context = ""
                for dep_id in step.get("depends_on", []):
                    if dep_id in results:
                        dep_context += f"\n[{results[dep_id].worker} output]\n{results[dep_id].output}\n"

                full_task = step["task"]
                if dep_context:
                    full_task += f"\n\n--- Context from prior steps ---{dep_context}"

                self._call_count += 1
                role = WorkerRole(step["worker"])
                tasks.append((step["step"], run_worker(
                    self.client, role, full_task, model=self.worker_model,
                )))

            # Await all parallel tasks
            for step_id, coro in tasks:
                results[step_id] = await coro
                completed.add(step_id)

        return results

    async def _synthesize(self, request: str, results: dict[int, TaskResult]) -> str:
        if self._call_count >= self.max_calls:
            # Fallback: return last worker output
            last = max(results.keys()) if results else 0
            return results[last].output if last in results else "[No output]"

        context = "\n\n".join(
            f"## [{r.worker}] {r.task}\n{r.output}"
            for r in results.values()
        )
        self._call_count += 1
        resp = await self.client.chat.completions.create(
            model=self.orchestrator_model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user", "content": f"## Original Request\n{request}\n\n## Worker Outputs\n{context}"},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main():
    client = AsyncOpenAI()
    orch = Orchestrator(client=client)

    request = (
        "Write a brief competitive analysis of the top 3 cloud vector databases "
        "(Pinecone, Weaviate, Qdrant) for a startup evaluating options for their "
        "RAG system. Include pricing considerations and a recommendation."
    )

    print(f"Request: {request}\n")
    print("Orchestrator planning and delegating...\n")

    result = await orch.run(request)
    print("=" * 60)
    print("FINAL DELIVERABLE:")
    print("=" * 60)
    print(result)
    print(f"\nTotal LLM calls: {orch._call_count}")


if __name__ == "__main__":
    asyncio.run(main())
