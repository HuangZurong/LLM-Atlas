# Alignment Frontiers

*Prerequisite: [../04_Post_Training/02_Alignment/](../04_Post_Training/02_Alignment/). Continuously updated. Last update: 2025-06*

---

## 1. Reasoning Alignment

### [DeepSeek-R1] Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025)
- **Institution**: DeepSeek
- **Link**: arXiv:2501.12948
- **One-line Summary**: Trains strong reasoning via GRPO + rule-based rewards (no learned RM), observing emergent behaviors like the "Aha Moment."
- **Core Innovation**: Proves pure RL (without SFT reasoning data) can elicit self-reflection and long-chain reasoning from scratch; Cold Start SFT + large-scale GRPO two-stage pipeline.
- **Key Results**: R1 achieves 79.8% on AIME 2024, 97.3% on MATH-500, matching OpenAI o1-0912.
- **Implications**: Detailed in 04_Post_Training/02_Alignment/03_Reasoning_Alignment/02_GRPO.md.

### [o1] Learning to Reason with LLMs (OpenAI, 2024)
- **Institution**: OpenAI
- **Link**: openai.com/index/learning-to-reason-with-llms
- **One-line Summary**: First large-scale inference-time compute scaling commercial model; significantly improves reasoning via internal chain-of-thought search.
- **Core Innovation**: "Thinking time" as a scalable dimension — more reasoning tokens = better results.
- **Key Results**: AIME 2024 top-500 ranking, Codeforces 89th percentile.
- **Limitations**: Architecture undisclosed; cannot confirm whether MCTS or other search algorithms are used.

### [QwQ] QwQ: Reflect Deeply on the Boundaries of the Unknown (Qwen, 2024)
- **Institution**: Alibaba Qwen
- **Link**: qwenlm.github.io/blog/qwq-32b-preview
- **One-line Summary**: 32B open-source reasoning model trained via RLVR for long-chain reasoning.
- **Relation to Existing Work**: Contrasts with R1-Distill-32B — QwQ uses direct RLVR training while R1-Distill uses distillation; the latter achieves better results.

---

## 2. Preference Alignment

### [SimPO] SimPO: Simple Preference Optimization with a Reference-Free Reward (Meng et al., 2024)
- **Institution**: University of Virginia
- **Link**: arXiv:2405.14734
- **One-line Summary**: Removes the reference model from DPO, using length-normalized sequence log-probability as implicit reward.
- **Core Innovation**: $r(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$ — length-normalized log-prob directly as reward, no $\pi_{ref}$ needed.
- **Key Results**: Outperforms DPO and ORPO on AlpacaEval 2.0 LC.
- **Implications**: Further simplification direction for the DPO family.

### [ORPO] ORPO: Monolithic Preference Optimization without Reference Model (Hong et al., 2024)
- **Institution**: KAIST
- **Link**: arXiv:2403.07691
- **One-line Summary**: Merges SFT and preference alignment into a single training stage without a reference model.
- **Core Innovation**: Adds an odds ratio penalty to the standard NLL loss, penalizing the generation probability of rejected responses.

---

## 3. Safety & Alignment Robustness

### [RepE] Representation Engineering: A Top-Down Approach to AI Transparency (Zou et al., 2023)
- **Institution**: Center for AI Safety
- **Link**: arXiv:2310.01405
- **One-line Summary**: Controls model behavior by directly manipulating internal representations (rather than fine-tuning).
- **Core Innovation**: Identifies direction vectors encoding specific concepts (e.g., "honesty") in the model's representation space; adds/subtracts these vectors to steer outputs.

### [Circuit Breakers] Improving Alignment and Robustness with Circuit Breakers (Zou et al., 2024)
- **Institution**: Center for AI Safety / Gray Swan AI
- **Link**: arXiv:2406.04313
- **One-line Summary**: Installs "circuit breakers" at the representation level that interrupt generation when harmful output patterns are detected.
- **Core Innovation**: Bypasses behavioral-level alignment (RLHF/DPO) to directly block harmful generation paths in representation space.
- **Key Results**: >90% defense success rate against GCG, AutoDAN, and similar attacks while maintaining normal task performance.

---

## 4. Reasoning Alignment — 2025 Updates

### [o3] OpenAI o3 and o4-mini (OpenAI, 2025)
- **Institution**: OpenAI
- **Link**: openai.com/index/introducing-o3-and-o4-mini
- **One-line Summary**: Next-generation reasoning models with tool use (web search, code execution, image analysis) integrated into the thinking loop.
- **Core Innovation**: "Thinking with tools" — the model can invoke tools mid-reasoning, not just as final actions; significantly extends the effective reasoning depth.
- **Key Results**: o3 achieves 96.7% on MATH-500, 88.9% on GPQA Diamond; o4-mini matches o3 on most benchmarks at 1/10 cost.

### [DeepSeek-R1-0528] DeepSeek-R1-0528 (DeepSeek, 2025)
- **Institution**: DeepSeek
- **Link**: huggingface.co/deepseek-ai/DeepSeek-R1-0528
- **One-line Summary**: Major R1 update doubling reasoning depth (23K avg tokens vs 12K), matching o3 and Gemini 2.5 Pro.
- **Core Innovation**: Enhanced post-training algorithms with increased compute; deeper chain-of-thought reasoning with sustained 10+ minute thinking on hard problems.
- **Key Results**: Matches o3 on AIME 2024 and MATH-500; open-weight under MIT license.

### [Claude 4] Claude Opus 4 and Sonnet 4 (Anthropic, 2025)
- **Institution**: Anthropic
- **Link**: anthropic.com/news/claude-4
- **One-line Summary**: 200K-token context with multi-hour "Extended Thinking" and hybrid reasoning + tool use in a single loop.
- **Core Innovation**: Extended Thinking allows sustained reasoning over hours; agentic coding capabilities (7-hour autonomous refactoring demonstrated).
- **Key Results**: #1 on SWE-bench at launch; Opus 4 sets new SOTA on agentic coding tasks.

### [Qwen3 Thinking] Qwen3 Hybrid Thinking Mode (Alibaba, 2025)
- **Institution**: Alibaba Qwen
- **One-line Summary**: Single model supports toggling between "thinking" (deep CoT) and "non-thinking" (fast) modes via system prompt.
- **Core Innovation**: Two-phase training — standard SFT first, then RL with verifiable rewards to train the thinking mode; no separate reasoning model needed.
- **Implications**: Unifies fast chat and deep reasoning into one model, reducing deployment complexity.
