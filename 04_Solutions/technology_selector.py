"""
LLM Technology Selector — Interactive decision tool for choosing the right LLM approach.

Usage:
    python technology_selector.py

Walks through a series of questions and recommends:
- Approach: Prompt Engineering vs RAG vs Fine-tuning vs Hybrid
- Model tier: API (frontier) vs API (mid) vs Self-hosted
- Estimated monthly cost range
- Key risks and next steps
"""

from dataclasses import dataclass


@dataclass
class Recommendation:
    approach: str
    model_tier: str
    cost_range: str
    confidence: str
    risks: list[str]
    next_steps: list[str]


def ask_choice(question: str, options: list[str]) -> int:
    print(f"\n{'='*60}")
    print(f"  {question}")
    print(f"{'='*60}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        try:
            choice = int(input("\n  → Your choice: "))
            if 1 <= choice <= len(options):
                return choice
        except (ValueError, EOFError):
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def run_selector() -> Recommendation:
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  LLM Technology Selector v1.0".center(58) + "║")
    print("║" + "  Interactive Decision Tool".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    # ── Q1: Task complexity ──
    q1 = ask_choice(
        "What is the primary task complexity?",
        [
            "Simple extraction / classification / formatting",
            "Knowledge-intensive Q&A over domain documents",
            "Complex reasoning, multi-step analysis, or generation",
            "Autonomous actions (tool use, API calls, workflows)",
        ]
    )

    # ── Q2: Domain specificity ──
    q2 = ask_choice(
        "How specialized is your domain knowledge?",
        [
            "General knowledge (common sense, public info)",
            "Moderately specialized (industry terms, internal docs)",
            "Highly specialized (medical, legal, engineering standards)",
        ]
    )

    # ── Q3: Data availability ──
    q3 = ask_choice(
        "How much domain-specific data do you have?",
        [
            "Little to none (< 100 documents)",
            "Moderate (100 - 10K documents)",
            "Large corpus (10K+ documents or structured databases)",
        ]
    )

    # ── Q4: Latency requirement ──
    q4 = ask_choice(
        "What is your latency requirement?",
        [
            "Real-time (< 2 seconds)",
            "Near real-time (2 - 10 seconds)",
            "Batch / async (minutes acceptable)",
        ]
    )

    # ── Q5: Data privacy ──
    q5 = ask_choice(
        "What are your data privacy constraints?",
        [
            "No restrictions (public data, can use any API)",
            "Moderate (no PII to external APIs, but cloud OK)",
            "Strict (on-premise only, regulated industry)",
        ]
    )

    # ── Q6: Budget ──
    q6 = ask_choice(
        "What is your monthly budget for LLM infrastructure?",
        [
            "Minimal (< $500/month)",
            "Moderate ($500 - $5,000/month)",
            "Significant ($5,000 - $50,000/month)",
            "Enterprise (> $50,000/month)",
        ]
    )

    # ── Q7: Volume ──
    q7 = ask_choice(
        "Expected request volume?",
        [
            "Low (< 1K requests/day)",
            "Medium (1K - 50K requests/day)",
            "High (> 50K requests/day)",
        ]
    )

    # ── Decision Logic ──
    scores = {"prompt_eng": 0, "rag": 0, "finetune": 0, "agent": 0}

    # Task complexity
    if q1 == 1:
        scores["prompt_eng"] += 3
    elif q1 == 2:
        scores["rag"] += 3
    elif q1 == 3:
        scores["finetune"] += 2; scores["rag"] += 1
    elif q1 == 4:
        scores["agent"] += 3

    # Domain specificity
    if q2 == 1:
        scores["prompt_eng"] += 2
    elif q2 == 2:
        scores["rag"] += 2
    elif q2 == 3:
        scores["rag"] += 1; scores["finetune"] += 2

    # Data availability
    if q3 == 1:
        scores["prompt_eng"] += 2
    elif q3 == 2:
        scores["rag"] += 2
    elif q3 == 3:
        scores["rag"] += 1; scores["finetune"] += 2

    # Latency
    if q4 == 1:
        scores["prompt_eng"] += 1; scores["finetune"] += 1
    elif q4 == 3:
        scores["rag"] += 1; scores["finetune"] += 1

    # Privacy pushes toward self-hosting / fine-tuning
    if q5 == 3:
        scores["finetune"] += 2

    # Agent boost if task is autonomous
    if q1 == 4:
        scores["agent"] += 2
        scores["rag"] += 1  # Agents often use RAG

    # Determine primary approach
    primary = max(scores, key=scores.get)
    approach_map = {
        "prompt_eng": "Prompt Engineering",
        "rag": "RAG (Retrieval-Augmented Generation)",
        "finetune": "Fine-tuning (SFT/DPO)",
        "agent": "Agentic RAG (Agent + RAG + Tools)",
    }
    approach = approach_map[primary]

    # Check for hybrid
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_scores[1][1] >= sorted_scores[0][1] - 1:
        secondary = approach_map[sorted_scores[1][0]]
        approach = f"Hybrid: {approach} + {secondary}"

    # Model tier
    if q5 == 3 or (q6 >= 3 and q7 == 3):
        model_tier = "Self-hosted (vLLM/TGI + open-weight model)"
    elif q6 <= 2 and q7 <= 2:
        model_tier = "API — mid-tier (GPT-4o-mini, Claude Haiku, Gemini Flash)"
    else:
        model_tier = "API — frontier (GPT-4o, Claude Sonnet, Gemini Pro)"

    # Cost estimate
    cost_map = {1: "$100 - $500", 2: "$500 - $5,000", 3: "$5,000 - $30,000", 4: "$30,000+"}
    cost_range = cost_map.get(q6, "Variable")

    # Confidence
    max_score = sorted_scores[0][1]
    if max_score >= 6:
        confidence = "HIGH — clear winner"
    elif max_score >= 4:
        confidence = "MEDIUM — consider prototyping top 2 approaches"
    else:
        confidence = "LOW — run a PoC for each approach before deciding"

    # Risks
    risks = []
    if primary == "finetune":
        risks.append("Fine-tuning requires significant data prep and compute investment.")
        risks.append("Model may lose general capabilities (catastrophic forgetting).")
    if primary == "rag":
        risks.append("RAG quality depends heavily on chunking and retrieval quality.")
        risks.append("Latency increases with retrieval pipeline complexity.")
    if primary == "agent":
        risks.append("Agent reliability is lower than deterministic pipelines.")
        risks.append("Tool call errors can cascade; implement circuit breakers.")
    if q5 == 3:
        risks.append("Self-hosting requires dedicated GPU infrastructure and MLOps.")

    # Next steps
    next_steps = [
        "1. Build a 2-week PoC with 5 representative test cases.",
        "2. Create a Golden Dataset (≥50 examples) for evaluation.",
        "3. Set up automated eval pipeline before iterating.",
    ]
    if "RAG" in approach:
        next_steps.append("4. Start with hybrid search (BM25 + vector) + reranker.")
    if "Fine-tuning" in approach:
        next_steps.append("4. Start with LoRA/QLoRA on a 7B-14B model before scaling up.")
    if "Agent" in approach:
        next_steps.append("4. Define tool schemas and implement HITL gates for high-risk actions.")

    return Recommendation(
        approach=approach,
        model_tier=model_tier,
        cost_range=cost_range,
        confidence=confidence,
        risks=risks,
        next_steps=next_steps,
    )


def print_report(rec: Recommendation):
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + "  RECOMMENDATION REPORT".center(58) + "║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Approach:    {rec.approach:<43}║")
    print(f"║  Model Tier:  {rec.model_tier:<43}║")
    print(f"║  Cost Range:  {rec.cost_range:<43}║")
    print(f"║  Confidence:  {rec.confidence:<43}║")
    print("╠" + "═"*58 + "╣")
    print("║  RISKS:".ljust(59) + "║")
    for r in rec.risks:
        # Wrap long lines
        while len(r) > 55:
            print(f"║    {r[:55]}║")
            r = r[55:]
        print(f"║    {r:<55}║")
    print("╠" + "═"*58 + "╣")
    print("║  NEXT STEPS:".ljust(59) + "║")
    for s in rec.next_steps:
        while len(s) > 55:
            print(f"║    {s[:55]}║")
            s = s[55:]
        print(f"║    {s:<55}║")
    print("╚" + "═"*58 + "╝")


if __name__ == "__main__":
    rec = run_selector()
    print_report(rec)
