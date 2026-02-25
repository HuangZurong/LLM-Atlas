# Evaluation Loop: Measuring and Iterating Domain LLM Systems

*Prerequisite: [../02_Scientist/05_Evaluation/](../02_Scientist/05_Evaluation/).*

---

For evaluation theory and metrics, see 02_Scientist/05_Evaluation. This document addresses the practical question: how do you set up a continuous evaluation and improvement cycle for a domain LLM system?

## 1. Why Evaluation is the Bottleneck

In most domain LLM projects, the limiting factor is not training — it's knowing whether the model is actually getting better. Without rigorous evaluation:

- You can't tell if a change helped or hurt
- You optimize for the wrong things (loss goes down, but output quality doesn't improve)
- You can't communicate progress to stakeholders
- You don't know when to stop iterating

Evaluation is not a phase — it's a continuous process that runs in parallel with every other activity.

## 2. Building the Evaluation Set

### 2.1 Requirements

The evaluation set is the single most important artifact in your project. It must be:

- **Representative**: Covers all task types and difficulty levels the system will encounter
- **Expert-validated**: Every answer has been verified by a domain expert
- **Static**: Never used for training. Never modified based on model performance.
- **Versioned**: Track changes if you must add new examples
- **Sufficient size**: Minimum 200 examples for statistical significance; 500+ preferred

### 2.2 Composition

| Category | Proportion | Examples |
| :--- | :--- | :--- |
| **Factual recall** | 20-30% | "What is the standard curing time for C40 concrete?" |
| **Reasoning** | 20-30% | "Given these project constraints, what is the optimal construction sequence?" |
| **Multi-step** | 15-20% | "If material delivery is delayed by 2 weeks, what downstream milestones are affected and what mitigation options exist?" |
| **Edge cases** | 10-15% | Ambiguous questions, questions with insufficient information, out-of-scope requests |
| **Format compliance** | 10-15% | Tasks requiring specific output structure (reports, tables, checklists) |

### 2.3 Creating Gold-Standard Answers

For each evaluation question, create:

1. **Reference answer**: The ideal response, written by a domain expert
2. **Key facts**: A checklist of facts/points that must appear in any correct answer
3. **Unacceptable elements**: Things that would make an answer wrong (common misconceptions, dangerous advice)
4. **Difficulty rating**: Easy / Medium / Hard
5. **Category tags**: For slicing evaluation results by topic

Example:
```yaml
question: "What are the key considerations for transitioning an airport from construction to operations?"
reference_answer: "The transition requires coordinating across five dimensions: (1) commissioning and testing of all systems..."
key_facts:
  - mentions commissioning/testing
  - mentions staff training and certification
  - mentions documentation handover
  - mentions regulatory approval process
  - mentions phased transition (not big-bang)
unacceptable:
  - suggests skipping safety certification
  - ignores regulatory requirements
difficulty: medium
categories: [operations, transition, planning]
```

## 3. Evaluation Methods

### 3.1 The Evaluation Stack

Use multiple methods in combination. No single method is sufficient.

```
                    ┌─────────────────────┐
                    │   Human Expert       │  Gold standard. Expensive. Use for final validation.
                    │   Evaluation         │  50-100 examples per round.
                    ├─────────────────────┤
                    │   LLM-as-Judge       │  Scalable proxy for human judgment.
                    │                      │  Run on full eval set (200-500 examples).
                    ├─────────────────────┤
                    │   Automated Metrics  │  Fast, cheap, run on every change.
                    │   (key fact check,   │  Catches regressions immediately.
                    │    format validation) │
                    └─────────────────────┘
```

### 3.2 Automated Metrics

Run these on every model change (fast feedback):

| Metric | What It Checks | Implementation |
| :--- | :--- | :--- |
| **Key fact recall** | Does the answer contain required facts? | String matching or embedding similarity against key_facts checklist |
| **Format compliance** | Does output follow required structure? | Regex or schema validation |
| **Refusal accuracy** | Does model correctly refuse out-of-scope questions? | Check against labeled out-of-scope test cases |
| **Consistency** | Same question → same answer (across runs)? | Run each question 3 times, measure variance |
| **Latency** | Response time within acceptable range? | Time each inference call |
| **Token efficiency** | Answer length reasonable? | Check against expected length range |

### 3.3 LLM-as-Judge

Use a strong model to evaluate outputs at scale. This is the workhorse of modern LLM evaluation.

**Prompt template**:
```
You are an expert evaluator for a domain-specific AI assistant focused on infrastructure construction and operations.

Evaluate the following response on these dimensions (score 1-5 each):

1. **Accuracy**: Are all stated facts correct? (1=major errors, 5=fully accurate)
2. **Completeness**: Does the response address all aspects of the question? (1=misses key points, 5=comprehensive)
3. **Relevance**: Does the response stay on topic? (1=off-topic, 5=precisely relevant)
4. **Clarity**: Is the response well-organized and easy to understand? (1=confusing, 5=crystal clear)
5. **Safety**: Does the response avoid harmful or misleading advice? (1=dangerous, 5=fully safe)

Question: {question}
Reference answer: {reference}
Model response: {response}

Provide scores and brief justification for each dimension.
```

**Calibration**: Before relying on LLM-as-judge, validate it against human judgments on 50-100 examples. If agreement (Cohen's kappa) is below 0.6, refine the evaluation prompt.

### 3.4 Human Expert Evaluation

Reserve for:
- Final validation before deployment
- Calibrating LLM-as-judge
- Evaluating subjective quality (tone, professionalism, trustworthiness)
- Catching errors that automated methods miss

**Protocol**:
1. Blind evaluation (evaluator doesn't know which model version produced the output)
2. Standardized rubric (same scoring criteria for all evaluators)
3. Inter-annotator agreement check (at least 2 evaluators per example, measure agreement)
4. Structured feedback form (not just scores, but specific comments on what's wrong)

## 4. Evaluation Dimensions by System Type

### 4.1 RAG System Evaluation

| Layer | Metrics | How to Measure |
| :--- | :--- | :--- |
| **Retrieval** | Recall@k, MRR, NDCG | Compare retrieved docs against relevance labels |
| **Generation** | Faithfulness, relevance, completeness | LLM-as-judge against retrieved context |
| **End-to-end** | Answer accuracy, user satisfaction | Compare against gold-standard answers |

Critical: Evaluate retrieval and generation separately. If the final answer is wrong, you need to know whether retrieval failed (right answer wasn't retrieved) or generation failed (right context was retrieved but model ignored it).

### 4.2 Fine-tuned Model Evaluation

| Dimension | Metrics | How to Measure |
| :--- | :--- | :--- |
| **Domain performance** | Accuracy on domain eval set | Automated + LLM-as-judge |
| **General capability** | Performance on general benchmarks (MMLU, etc.) | Automated benchmarks |
| **Catastrophic forgetting** | Delta between base model and fine-tuned on general tasks | Before/after comparison |
| **Safety** | Refusal rate on adversarial prompts | Red-teaming test set |

### 4.3 KG-Integrated System Evaluation

| Dimension | Metrics | How to Measure |
| :--- | :--- | :--- |
| **Graph quality** | Triple accuracy, completeness, consistency | Expert sampling |
| **Query accuracy** | Text-to-Cypher correctness | Compare generated queries against gold-standard |
| **Reasoning depth** | Multi-hop answer accuracy | Test cases requiring 2-3 hop traversal |
| **Fallback quality** | Performance when KG doesn't have the answer | Test with out-of-graph questions |

## 5. The Iteration Workflow

### 5.1 Continuous Improvement Cycle

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Evaluate    │────→│  Analyze     │────→│  Improve     │────→│  Validate    │──┐
│  (full eval  │     │  (categorize │     │  (targeted   │     │  (re-run     │  │
│   set)       │     │   failures)  │     │   fixes)     │     │   eval set)  │  │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘  │
       ↑                                                                         │
       └─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Error Analysis Framework

After each evaluation round, categorize every failure:

```
Total errors: 47/200 (76.5% accuracy)

Breakdown:
├── Knowledge gaps: 18 (38%)
│   ├── Missing regulation knowledge: 8
│   ├── Missing technical procedures: 6
│   └── Missing domain terminology: 4
├── Reasoning errors: 12 (26%)
│   ├── Wrong causal chain: 5
│   ├── Incomplete analysis: 4
│   └── Contradictory statements: 3
├── Format violations: 9 (19%)
│   ├── Missing required sections: 5
│   └── Wrong output structure: 4
├── Hallucinations: 5 (11%)
│   ├── Fabricated statistics: 3
│   └── Non-existent regulations cited: 2
└── Refusal failures: 3 (6%)
    └── Answered out-of-scope questions: 3
```

This breakdown directly tells you what to fix next:
1. Add training data covering missing regulations (addresses 8 errors)
2. Add chain-of-thought examples for causal reasoning (addresses 5 errors)
3. Add more format-compliant examples (addresses 9 errors)

### 5.3 Regression Testing

Every improvement risks breaking something that previously worked. Maintain a regression test set:

- **Regression set**: Examples the model previously answered correctly. After any change, verify these still pass.
- **Improvement set**: Examples the model previously failed. After targeted fixes, verify these now pass.
- **Net improvement**: improvement_set gains - regression_set losses. Must be positive.

### 5.4 When to Stop Iterating

Stop when:
- You've met your predefined success criteria (Section 2 of 04_Finetuning_Playbook)
- Marginal improvement per iteration drops below a threshold (e.g., <1% accuracy gain per round)
- Remaining errors are in categories that can't be fixed with more data (fundamental model limitations)
- The cost of further iteration exceeds the value of improvement

## 6. Production Monitoring

Evaluation doesn't end at deployment. In production, monitor:

### 6.1 Online Metrics

| Metric | What It Tracks | Alert Threshold |
| :--- | :--- | :--- |
| **User feedback** | Thumbs up/down ratio | < 80% positive |
| **Query failure rate** | % of queries with no useful response | > 10% |
| **Latency P95** | 95th percentile response time | > target SLA |
| **Retrieval empty rate** | % of queries with no relevant documents retrieved | > 15% |
| **Hallucination flags** | User-reported or auto-detected fabrications | Any increase |

### 6.2 Offline Evaluation Cadence

| Frequency | Activity |
| :--- | :--- |
| **Weekly** | Run automated metrics on eval set (catch regressions from any system changes) |
| **Monthly** | LLM-as-judge evaluation on full eval set + new real user queries |
| **Quarterly** | Human expert evaluation, update eval set with new question types |

### 6.3 Data Flywheel

Production usage generates the most valuable evaluation data:

```
User queries → Log (with consent) → Sample interesting cases → Expert annotation → Add to eval set / training set
```

Prioritize logging:
- Queries where the user gave negative feedback
- Queries the model refused to answer
- Queries with unusually long/short responses
- Novel query patterns not seen in training

This closes the loop: production data improves evaluation, which improves the model, which improves production quality.

---

## Key References

1. **Zheng et al. (2023)**: *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*.
