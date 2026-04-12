# 09 · Context Engineering Evaluation Methods

A specialized evaluation framework for measuring context engineering effectiveness beyond general LLM/RAG metrics.

---

## Why CE-Specific Evaluation?

**General LLM Evaluation** focuses on output quality (BLEU, ROUGE, Accuracy, Perplexity). Context is treated as given input.

**CE-Specific Evaluation** focuses on context efficiency and composition quality:
- What to include in the context window
- How to order information
- When to compress
- How to track state across turns

**General RAG Evaluation** focuses on retrieval quality (Precision@K, Recall@K, MRR). Context is the retrieved documents.

**CE-Specific for RAG** focuses on token budget + retrieval quality tradeoff:
- Retrieval efficiency (relevant tokens / total tokens)
- Budget-constrained selection from retrieved pool
- Hierarchical routing accuracy

---

## Metric Categories

### 1. Token Budget Efficiency Metrics

#### Compression Ratio with Information Retention

**Compression Ratio**
```
Compression Ratio = Original Tokens / Compressed Tokens
```

**Information Retention Rate**
- Measured via BERTScore or semantic similarity
- Compares compressed context against original
- Target: ≥85% retention for acceptable compression

**Combined Metric**
```
Efficiency Score = Compression Ratio × Information Retention
```
- Higher is better
- Example: 3x compression with 0.90 retention = 2.7 efficiency score

#### Token Utilization Efficiency

**Effective Token Ratio**
```
Effective Token Ratio = Tokens Used in Reasoning / Total Tokens in Context
```

Measurement methods:
1. **Attention Weight Analysis**: Sum attention weights for each token, threshold at 10% of max
2. **Gradient-Based Attribution**: Compute ∂(output)/∂(input_tokens), rank by magnitude
3. **Perturbation Test**: Remove token, measure output change

Interpretation:
- <0.3: Many passive tokens (candidates for pruning)
- 0.3-0.6: Balanced context
- >0.6: Dense context, may need expansion

---

### 2. Context Composition Quality Metrics

#### Position-Aware Retrieval Accuracy

Tests for "Lost in the Middle" phenomenon.

```python
def evaluate_position_bias(context: str, query: str, relevant_info: str) -> float:
    """Measure accuracy variance across positions."""
    accuracies = []

    for position in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Insert relevant info at specified position
        positioned_context = insert_at_position(context, relevant_info, position)

        # Test model accuracy
        answer = llm.generate(positioned_context + query)
        accuracy = evaluate_answer(answer, ground_truth)
        accuracies.append(accuracy)

    # Lower variance = better composition (less position bias)
    return np.var(accuracies)
```

**Target**: Variance < 0.05 (model performs consistently regardless of position)

#### Context Attribution Score

Measures which context chunks contributed to the answer.

**Attribution Precision**
```
Attribution Precision = Attributed Tokens Actually Relevant / Total Attributed Tokens
```

**Attribution Recall**
```
Attribution Recall = Relevant Tokens Attributed / Total Relevant Tokens
```

Measurement via attention weights:
```python
def compute_attribution(context_chunks: list[str], answer: str) -> dict:
    """Compute attribution scores using attention weights."""
    # Get attention weights from model
    attention_weights = model.get_attention(context_chunks + answer)

    # Attributed chunks: top-k by attention
    attributed = get_top_k_chunks(attention_weights, k=3)

    # Compute precision/recall
    relevant = get_ground_truth_relevant_chunks()

    precision = len(attributed ∩ relevant) / len(attributed)
    recall = len(attributed ∩ relevant) / len(relevant)

    return {"precision": precision, "recall": recall}
```

#### Semantic Coherence Score

Measures semantic flow between adjacent context chunks.

```python
def coherence_score(context_chunks: list[str]) -> float:
    """Measure semantic flow between adjacent chunks."""
    embeddings = [embed(chunk) for chunk in context_chunks]
    transitions = [
        cosine_similarity(embeddings[i], embeddings[i+1])
        for i in range(len(embeddings) - 1)
    ]
    return np.mean(transitions)
```

Interpretation:
- <0.5: Abrupt topic changes (poor composition)
- 0.5-0.7: Moderate flow
- >0.7: Smooth semantic transitions

#### Context Redundancy Rate

```python
def redundancy_rate(context_chunks: list[str]) -> float:
    """Measure redundant information across chunks."""
    # Extract semantic units (entities, facts, claims)
    all_units = []
    for chunk in context_chunks:
        units = extract_semantic_units(chunk)
        all_units.extend(units)

    unique_units = set(all_units)

    redundancy = 1 - (len(unique_units) / len(all_units))
    return redundancy
```

Target: Redundancy < 20%

---

### 3. Budget Controller Performance Metrics

#### Trigger Accuracy

Measures correctness of compression triggers.

```python
def evaluate_trigger_accuracy(scenarios: list[dict]) -> float:
    """
    scenarios = [
        {"usage": 0.65, "expected": "ok"},
        {"usage": 0.80, "expected": "soft"},
        {"usage": 0.92, "expected": "hard"},
        ...
    ]
    """
    correct = 0
    for scenario in scenarios:
        level = budget_controller.check_level(scenario["usage"])
        if level == scenario["expected"]:
            correct += 1

    return correct / len(scenarios)
```

Target: ≥90% accuracy

#### Budget Exhaustion Pattern

Tracks at what token count the budget is exhausted.

**Optimal Range**: 85-95% utilization

```python
def evaluate_budget_exhaustion(queries: list) -> dict:
    """Analyze budget exhaustion patterns."""
    exhaustion_counts = {
        "under": 0,      # <70% utilization
        "optimal": 0,    # 70-95%
        "over": 0,       # >95% (truncation occurred)
    }

    for query in queries:
        context = ce_manager.build_context(query)
        utilization = count_tokens(context) / budget

        if utilization < 0.70:
            exhaustion_counts["under"] += 1
        elif utilization <= 0.95:
            exhaustion_counts["optimal"] += 1
        else:
            exhaustion_counts["over"] += 1

    return {
        "distribution": exhaustion_counts,
        "optimal_rate": exhaustion_counts["optimal"] / len(queries)
    }
```

Target: Optimal rate ≥80%

#### Threshold Calibration Score

Test system at different budget thresholds and measure performance degradation.

```python
def evaluate_threshold_calibration(budget_levels: list[float]) -> dict:
    """Plot performance degradation curve."""
    results = []

    for level in budget_levels:  # [0.6, 0.7, 0.8, 0.9, 0.95]
        test_budget = int(total_budget * level)

        accuracies = []
        for query in test_queries:
            context = ce_manager.build_context(query, budget=test_budget)
            answer = llm.generate(context + query)
            accuracies.append(evaluate_answer(answer))

        results.append({
            "budget_level": level,
            "accuracy": np.mean(accuracies)
        })

    # Compute degradation curve
    degradation = [
        results[i]["accuracy"] - results[i-1]["accuracy"]
        for i in range(1, len(results))
    ]

    return {
        "results": results,
        "degradation": degradation,
        "is_smooth": all(d > -0.05 for d in degradation)  # No cliff
    }
```

Target: Smooth degradation (no sudden drops >5%)

---

### 4. Compression Quality Metrics

#### Factual Consistency Score

Uses NLI model to detect compression artifacts.

```python
def factual_consistency(original: str, compressed: str) -> float:
    """Check if compression introduces contradictions."""
    # Split into claims
    original_claims = extract_claims(original)
    compressed_claims = extract_claims(compressed)

    # Use NLI to check consistency
    nli_model = load_nli_model()

    consistent = 0
    for claim in compressed_claims:
        # Check if claim is entailed by original
        result = nli_model.check_entailment(original, claim)
        if result in ["entailment", "neutral"]:
            consistent += 1

    return consistent / len(compressed_claims)
```

Target: ≥0.95 (95% of claims are consistent)

#### Entity Preservation Rate

```python
def entity_preservation(original: str, compressed: str) -> dict:
    """Track entity retention in compression."""
    entities_original = extract_entities(original)
    entities_compressed = extract_entities(compressed)

    # Group by entity type
    results = {}
    for entity_type in ["PERSON", "ORG", "DATE", "MONEY", "PRODUCT"]:
        original_set = {e.text for e in entities_original if e.type == entity_type}
        compressed_set = {e.text for e in entities_compressed if e.type == entity_type}

        if original_set:
            preservation = len(compressed_set) / len(original_set)
        else:
            preservation = 1.0  # No entities to preserve

        results[entity_type] = preservation

    # Overall
    all_original = {e.text for e in entities_original}
    all_compressed = {e.text for e in entities_compressed}
    results["overall"] = len(all_compressed) / len(all_original) if all_original else 1.0

    return results
```

Target: ≥90% for critical entity types (MONEY, DATE, PRODUCT)

#### Compression Stability

```python
def compression_stability(text: str, n_runs: int = 5) -> float:
    """Test determinism of compression."""
    compressed_versions = []

    for _ in range(n_runs):
        compressed = compressor.compress(text, budget=500)
        compressed_versions.append(compressed)

    # Compute pairwise similarity
    similarities = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            sim = compute_similarity(compressed_versions[i], compressed_versions[j])
            similarities.append(sim)

    return np.mean(similarities)
```

Target: ≥0.95 (high determinism)

---

### 5. Multi-Turn Context Management Metrics

#### State Tracking Accuracy

```python
def evaluate_state_tracking(conversations: list) -> float:
    """Test state tracking across conversation turns."""
    accuracies = []

    for conv in conversations:
        manager = SupportContextManager()

        for turn_idx, turn in enumerate(conv.turns):
            # Update state
            manager.update_state(turn.extracted_info)

            # Verify state every 3 turns
            if turn_idx % 3 == 0:
                tracked_state = manager.get_state()
                ground_truth = conv.ground_truth[turn_idx]

                # Compute accuracy
                matches = sum(
                    1 for key in ground_truth
                    if tracked_state.get(key) == ground_truth[key]
                )
                accuracy = matches / len(ground_truth)
                accuracies.append(accuracy)

    return np.mean(accuracies)
```

Target: ≥90%

#### Context Drift Detection Rate

```python
def evaluate_context_drift(conversations: list) -> dict:
    """Measure how quickly context becomes stale."""
    drift_scores = []

    for conv in conversations:
        manager = SupportContextManager()

        for turn_idx, turn in enumerate(conv.turns):
            # Track when entities were last mentioned
            manager.track_mentions(turn.content)

            # Compute drift for each tracked entity
            for entity_id in manager.tracked_entities:
                turns_since_mention = turn_idx - manager.last_mentioned[entity_id]
                drift_rate = turns_since_mention / len(conv.turns)
                drift_scores.append(drift_rate)

    return {
        "avg_drift": np.mean(drift_scores),
        "high_drift_rate": sum(1 for s in drift_scores if s > 0.5) / len(drift_scores)
    }
```

Target: high_drift_rate < 20%

#### Cross-Turn Consistency

```python
def evaluate_cross_turn_consistency(conversations: list) -> float:
    """Test if model gives consistent answers across turns."""
    consistency_scores = []

    for conv in conversations:
        manager = SupportContextManager()

        # Ask same question at different turns
        test_question = "What is the status of my order?"

        for turn_idx in [0, 3, 6, 9]:
            if turn_idx < len(conv.turns):
                context = manager.build_context()
                answer = llm.generate(context + test_question)

                # Store answer
                if turn_idx == 0:
                    reference_answer = answer
                else:
                    # Compare with reference
                    similarity = compute_similarity(answer, reference_answer)
                    consistency_scores.append(similarity)

    return np.mean(consistency_scores)
```

Target: ≥0.85

---

### 6. Hierarchical Retrieval Metrics

#### Routing Accuracy

Tests if summary level enables correct routing to detail sections.

```python
def evaluate_routing_accuracy(queries: list) -> float:
    """Test hierarchical routing."""
    correct = 0

    for query_data in queries:
        # Use summary level to route
        summary = hierarchical_manager.get_summary()
        selected_section = hierarchical_manager.route(query_data["query"], summary)

        # Check against ground truth
        if selected_section == query_data["relevant_section"]:
            correct += 1

    return correct / len(queries)
```

Target: ≥85%

#### Detail Retrieval Efficiency

```python
def evaluate_retrieval_efficiency(queries: list) -> dict:
    """Measure precision of detail-level retrieval."""
    efficiencies = []
    precisions = []

    for query_data in queries:
        # Retrieve detail chunks
        detail_chunks = hierarchical_manager.retrieve_details(query_data["query"])

        # Compute efficiency
        relevant_chunks = query_data["relevant_chunks"]
        relevant_loaded = [c for c in detail_chunks if c in relevant_chunks]

        efficiency = len(relevant_loaded) / len(detail_chunks)
        precision = len(relevant_loaded) / len(relevant_chunks)

        efficiencies.append(efficiency)
        precisions.append(precision)

    return {
        "efficiency": np.mean(efficiencies),  # Relevant / Loaded
        "precision": np.mean(precisions),      # Relevant / Total Relevant
    }
```

Target: efficiency ≥0.7, precision ≥0.8

#### Two-Stage Latency

```python
def evaluate_latency(queries: list) -> dict:
    """Compare two-stage vs single-stage latency."""
    two_stage_times = []
    single_stage_times = []

    for query in queries:
        # Two-stage approach
        start = time.time()
        summary = hierarchical_manager.get_summary()  # Stage 1
        detail = hierarchical_manager.retrieve_details(query)  # Stage 2
        answer = llm.generate(summary + detail + query)
        two_stage_times.append(time.time() - start)

        # Single-stage baseline
        start = time.time()
        full_context = load_full_context()
        answer = llm.generate(full_context + query)
        single_stage_times.append(time.time() - start)

    return {
        "two_stage_avg": np.mean(two_stage_times),
        "single_stage_avg": np.mean(single_stage_times),
        "speedup": np.mean(single_stage_times) / np.mean(two_stage_times)
    }
```

Target: speedup ≥1.5x

---

### 7. Information-Theoretic Metrics

#### Perplexity Reduction per Token

```python
def perplexity_reduction_per_token(query: str, context: str) -> float:
    """Measure marginal value of each token."""
    # Perplexity without context
    ppl_without = compute_perplexity(query)

    # Perplexity with context
    ppl_with = compute_perplexity(query, context=context)

    # Reduction per token
    reduction = ppl_without - ppl_with
    tokens = count_tokens(context)

    return reduction / tokens
```

Interpretation:
- >0.01: High-value context
- 0.001-0.01: Moderate value
- <0.001: Low-value context (candidate for pruning)

#### Mutual Information Score

```python
def mutual_information_score(context: str, query: str, answer: str) -> float:
    """Quantify how much context reduces uncertainty about answer."""
    # Estimate P(context, answer)
    joint_prob = estimate_joint_probability(context, answer)

    # Estimate P(context), P(answer)
    context_prob = estimate_probability(context)
    answer_prob = estimate_probability(answer)

    # Compute MI
    mi = joint_prob * np.log(joint_prob / (context_prob * answer_prob))

    return mi
```

Higher MI = context more informative for the answer

---

### 8. Cost-Effectiveness Metrics

#### Cost per Quality Unit

```python
def cost_per_quality_unit(
    accuracy: float,
    avg_tokens: int,
    cost_per_1k_tokens: float = 0.003  # Claude Sonnet pricing
) -> float:
    """Compute cost per quality unit."""
    cost = (avg_tokens / 1000) * cost_per_1k_tokens
    return cost / accuracy
```

Example:
- Baseline: 5000 tokens, 80% accuracy → $0.015 / 0.80 = $0.01875 per quality unit
- CE system: 2000 tokens, 85% accuracy → $0.006 / 0.85 = $0.00706 per quality unit
- **Improvement**: 62% cost reduction

#### Break-Even Analysis

```python
def break_even_analysis(
    ce_implementation_cost: float,  # One-time
    cost_savings_per_1k_queries: float,
    queries_per_month: int
) -> dict:
    """Calculate when CE investment pays off."""
    monthly_savings = (queries_per_month / 1000) * cost_savings_per_1k_queries

    months_to_break_even = ce_implementation_cost / monthly_savings

    return {
        "monthly_savings": monthly_savings,
        "months_to_break_even": months_to_break_even,
        "annual_roi": (monthly_savings * 12 - ce_implementation_cost) / ce_implementation_cost
    }
```

---

## Evaluation Implementation

### Complete Evaluation Framework

```python
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class CEMetrics:
    """Comprehensive CE evaluation metrics."""

    # Token efficiency
    compression_ratio: float
    information_retention: float
    token_utilization: float

    # Budget management
    budget_adherence: float
    trigger_accuracy: float

    # Composition quality
    position_bias_variance: float
    coherence_score: float
    redundancy_rate: float

    # Compression quality
    factual_consistency: float
    entity_preservation: float

    # Multi-turn (optional)
    state_tracking_accuracy: float = None
    cross_turn_consistency: float = None

    # Hierarchical retrieval (optional)
    routing_accuracy: float = None
    retrieval_efficiency: float = None

    # Cost-effectiveness
    cost_per_quality_unit: float = None

    # Task performance
    task_accuracy: float = None

    def summary(self) -> str:
        report = f"""
CE Evaluation Report
═══════════════════════════════════════
Token Efficiency:
  Compression Ratio: {self.compression_ratio:.2f}x
  Information Retention: {self.information_retention:.1%}
  Token Utilization: {self.token_utilization:.1%}

Budget Management:
  Budget Adherence: {self.budget_adherence:.1%}
  Trigger Accuracy: {self.trigger_accuracy:.1%}

Composition Quality:
  Position Bias Variance: {self.position_bias_variance:.3f}
  Coherence Score: {self.coherence_score:.2f}
  Redundancy Rate: {self.redundancy_rate:.1%}

Compression Quality:
  Factual Consistency: {self.factual_consistency:.1%}
  Entity Preservation: {self.entity_preservation:.1%}
"""

        if self.state_tracking_accuracy is not None:
            report += f"""
Multi-Turn Performance:
  State Tracking Accuracy: {self.state_tracking_accuracy:.1%}
  Cross-Turn Consistency: {self.cross_turn_consistency:.1%}
"""

        if self.routing_accuracy is not None:
            report += f"""
Hierarchical Retrieval:
  Routing Accuracy: {self.routing_accuracy:.1%}
  Retrieval Efficiency: {self.retrieval_efficiency:.1%}
"""

        if self.task_accuracy is not None:
            report += f"""
Task Performance:
  Accuracy: {self.task_accuracy:.1%}
"""

        if self.cost_per_quality_unit is not None:
            report += f"""
Cost-Effectiveness:
  Cost per Quality Unit: ${self.cost_per_quality_unit:.4f}
"""

        return report


class CEEvaluator:
    """Context Engineering System Evaluator."""

    def __init__(self, budget: int, llm_client, embedder):
        self.budget = budget
        self.llm = llm_client
        self.embedder = embedder

    def evaluate_compression(self, original: str, compressed: str) -> dict:
        """Evaluate compression quality."""
        return {
            "compression_ratio": count_tokens(original) / count_tokens(compressed),
            "information_retention": bert_score(compressed, original),
            "factual_consistency": nli_consistency(compressed, original),
            "entity_preservation": entity_overlap(compressed, original),
        }

    def evaluate_position_bias(self, test_cases: list) -> float:
        """Test Lost in the Middle effect."""
        variances = []

        for case in test_cases:
            accuracies = []
            for position in [0.0, 0.25, 0.5, 0.75, 1.0]:
                context = insert_at_position(
                    case["context"],
                    case["relevant_info"],
                    position
                )
                answer = self.llm.generate(context + case["query"])
                accuracy = evaluate_answer(answer, case["ground_truth"])
                accuracies.append(accuracy)

            variances.append(np.var(accuracies))

        return np.mean(variances)

    def evaluate_budget_controller(self, scenarios: list) -> float:
        """Test budget controller decisions."""
        correct = 0
        for scenario in scenarios:
            level = self.budget_controller.check_level(scenario["usage"])
            if level == scenario["expected"]:
                correct += 1
        return correct / len(scenarios)

    def run_full_evaluation(
        self,
        test_queries: list,
        ce_system,
        baseline_system=None
    ) -> CEMetrics:
        """Run comprehensive CE evaluation."""

        # Compression evaluation
        compression_results = []
        for query in test_queries[:20]:  # Sample
            original = load_full_context(query)
            compressed = ce_system.compress(original, budget=self.budget)
            compression_results.append(
                self.evaluate_compression(original, compressed)
            )

        # Position bias
        position_bias = self.evaluate_position_bias(test_queries[:20])

        # Budget controller
        budget_scenarios = [
            {"usage": u, "expected": e}
            for u, e in [
                (0.65, "ok"), (0.72, "ok"), (0.80, "soft"),
                (0.88, "hard"), (0.95, "emergency")
            ]
        ] * 10
        trigger_acc = self.evaluate_budget_controller(budget_scenarios)

        # Task performance
        accuracies = []
        for query in test_queries:
            context = ce_system.build_context(query)
            answer = self.llm.generate(context + query["text"])
            accuracies.append(evaluate_answer(answer, query["ground_truth"]))

        # Compile metrics
        metrics = CEMetrics(
            compression_ratio=np.mean([r["compression_ratio"] for r in compression_results]),
            information_retention=np.mean([r["information_retention"] for r in compression_results]),
            token_utilization=np.mean([...]),  # Implement token utilization measurement
            budget_adherence=np.mean([...]),
            trigger_accuracy=trigger_acc,
            position_bias_variance=position_bias,
            coherence_score=np.mean([...]),
            redundancy_rate=np.mean([...]),
            factual_consistency=np.mean([r["factual_consistency"] for r in compression_results]),
            entity_preservation=np.mean([r["entity_preservation"] for r in compression_results]),
            task_accuracy=np.mean(accuracies),
        )

        return metrics


# Usage
def evaluate_ce_system():
    """Evaluate a CE system implementation."""
    evaluator = CEEvaluator(
        budget=2000,
        llm_client=get_llm_client(),
        embedder=get_embedder()
    )

    test_queries = load_test_dataset()

    metrics = evaluator.run_full_evaluation(
        test_queries=test_queries,
        ce_system=my_ce_implementation
    )

    print(metrics.summary())

    return metrics
```

---

## Benchmarking Protocols

### Benchmark 1: Compression Strategy Comparison

Compare different compression strategies on standard dataset.

```python
def benchmark_compression_strategies():
    """Compare compression strategies."""
    test_docs = load_benchmark_documents()  # 100 docs, varied lengths

    strategies = {
        "Truncation": TruncationStrategy(),
        "Sliding Window": SlidingWindowStrategy(),
        "Extractive Summary": ExtractiveSummaryStrategy(),
        "Abstractive Summary": AbstractiveSummaryStrategy(),
        "Perplexity Pruning": PerplexityPruningStrategy(),
        "Adaptive Compression": AdaptiveCompressor(),
    }

    results = {}
    for name, strategy in strategies.items():
        metrics = []
        for doc in test_docs:
            compressed = strategy.compress(doc, budget=500)
            metrics.append(evaluate_compression(doc, compressed))

        results[name] = {
            "avg_compression_ratio": np.mean([m["compression_ratio"] for m in metrics]),
            "avg_information_retention": np.mean([m["information_retention"] for m in metrics]),
            "avg_factual_consistency": np.mean([m["factual_consistency"] for m in metrics]),
            "avg_entity_preservation": np.mean([m["entity_preservation"] for m in metrics]),
        }

    # Print comparison table
    print_comparison_table(results)
    return results
```

### Benchmark 2: Multi-Turn Conversation Stress Test

```python
def benchmark_multi_turn():
    """Test multi-turn context management."""
    conversations = load_conversation_dataset()  # 50 conversations, 10+ turns

    manager = SupportContextManager(total_budget=2000)

    results = {
        "state_tracking_accuracy": [],
        "context_drift": [],
        "consistency": [],
    }

    for conv in conversations:
        for turn_idx, turn in enumerate(conv.turns):
            manager.update_state(turn.extracted_info)

            if turn_idx % 3 == 0:
                # Test state tracking
                accuracy = verify_state(manager.state, conv.ground_truth[turn_idx])
                results["state_tracking_accuracy"].append(accuracy)

                # Test drift
                drift = compute_drift(manager)
                results["context_drift"].append(drift)

        # Test cross-turn consistency
        consistency = test_consistency(manager, conv.test_questions)
        results["consistency"].append(consistency)

    return {k: np.mean(v) for k, v in results.items()}
```

### Benchmark 3: Needle in a Haystack

Context window stress test.

```python
def benchmark_niah():
    """Needle in a Haystack test."""
    results = []

    for context_length in [1000, 5000, 10000, 50000]:
        for position in [0.0, 0.25, 0.5, 0.75, 1.0]:
            # Generate haystack
            haystack = generate_haystack(context_length)

            # Insert needle at position
            needle = "The secret code is 7294"
            context = insert_at_position(haystack, needle, position)

            # Test retrieval
            query = "What is the secret code?"
            answer = llm.generate(context + query)

            success = "7294" in answer
            results.append({
                "context_length": context_length,
                "position": position,
                "success": success
            })

    # Analyze results
    return analyze_niah_results(results)
```

---

## Evaluation Best Practices

### 1. Establish Baselines

Always compare against:
- **Naive baseline**: No CE (full context or truncation)
- **Simple baseline**: Fixed-size chunking
- **Previous version**: When iterating on CE system

### 2. Use Multiple Metrics

Don't rely on single metric:
- Combine efficiency (compression) + quality (retention)
- Balance cost vs performance
- Test both token-level and task-level metrics

### 3. Test Edge Cases

- Very long documents (>50k tokens)
- Very short queries (<10 tokens)
- Multi-turn conversations (>20 turns)
- Budget emergencies (>95% utilization)

### 4. Real-World Evaluation

Beyond synthetic benchmarks:
- A/B test with real users
- Monitor production metrics
- Track cost savings over time

### 5. Continuous Monitoring

```python
class CEProductionMonitor:
    """Monitor CE system in production."""

    def __init__(self):
        self.metrics_history = []

    def log_request(self, query: str, context: str, answer: str):
        """Log request metrics."""
        self.metrics_history.append({
            "timestamp": time.time(),
            "tokens_used": count_tokens(context),
            "budget": self.budget,
            "utilization": count_tokens(context) / self.budget,
            # ... other metrics
        })

    def get_aggregate_metrics(self, window_hours: int = 24) -> dict:
        """Get aggregate metrics for time window."""
        recent = filter_recent(self.metrics_history, window_hours)

        return {
            "avg_utilization": np.mean([m["utilization"] for m in recent]),
            "p95_utilization": np.percentile([m["utilization"] for m in recent], 95),
            "budget_violations": sum(1 for m in recent if m["utilization"] > 1.0),
            # ... other aggregates
        }
```

---

## Conclusion

CE-specific evaluation focuses on:

1. **Efficiency**: Token budget utilization, compression effectiveness
2. **Composition**: Position bias, coherence, redundancy
3. **Quality**: Factual consistency, entity preservation
4. **Management**: State tracking, drift detection
5. **Cost**: Cost per quality unit, ROI

These metrics distinguish CE from general LLM/RAG evaluation by treating context as the primary optimization target, not just input.
