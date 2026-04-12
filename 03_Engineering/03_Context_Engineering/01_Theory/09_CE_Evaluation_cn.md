# 09 · 上下文工程评估方法

一个专门用于衡量上下文工程有效性的评估框架，关注点超越通用 LLM / RAG 指标。

---

## 为什么需要上下文工程专属评估？

**通用 LLM 评估**关注输出质量（BLEU、ROUGE、Accuracy、Perplexity）。上下文被视作既定输入。

**CE 专属评估**关注上下文效率与编排质量：
- 什么该放进上下文窗口
- 信息如何排序
- 何时压缩
- 如何跨轮次跟踪状态

**通用 RAG 评估**关注检索质量（Precision@K、Recall@K、MRR）。上下文就是检索到的文档。

**面向 RAG 的 CE 专属评估**关注 token 预算与检索质量之间的权衡：
- 检索效率（relevant tokens / total tokens）
- 从已检索池中进行预算约束下的选择
- 分层路由准确率

---

## 指标类别

### 1. Token 预算效率类指标

#### 带信息保留率的压缩比

**压缩比（Compression Ratio）**
```
Compression Ratio = Original Tokens / Compressed Tokens
```

**信息保留率（Information Retention Rate）**
- 通过 BERTScore 或语义相似度衡量
- 比较压缩后的上下文与原始上下文
- 目标：≥85% 保留率可视为可接受的压缩

**组合指标**
```
Efficiency Score = Compression Ratio × Information Retention
```
- 越高越好
- 例：压缩 3 倍且保留率 0.90 = 2.7 的效率分数

#### Token 利用效率

**有效 Token 比例（Effective Token Ratio）**
```
Effective Token Ratio = Tokens Used in Reasoning / Total Tokens in Context
```

测量方法：
1. **Attention Weight Analysis**：汇总每个 token 的注意力权重，以最大值的 10% 为阈值
2. **Gradient-Based Attribution**：计算 ∂(output)/∂(input_tokens)，按幅值排序
3. **Perturbation Test**：移除 token，观察输出变化

解释：
- <0.3：存在大量被动 token（可作为裁剪候选）
- 0.3–0.6：上下文较平衡
- >0.6：上下文很密集，可能需要扩容

---

### 2. 上下文编排质量指标

#### 位置感知检索准确率

用于测试 “Lost in the Middle” 现象。

```python
def evaluate_position_bias(context: str, query: str, relevant_info: str) -> float:
    """测量不同位置上的准确率方差。"""
    accuracies = []

    for position in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # 在指定位置插入相关信息
        positioned_context = insert_at_position(context, relevant_info, position)

        # 测试模型准确率
        answer = llm.generate(positioned_context + query)
        accuracy = evaluate_answer(answer, ground_truth)
        accuracies.append(accuracy)

    # 方差越低 = 编排越好（位置偏差越小）
    return np.var(accuracies)
```

**目标**：方差 < 0.05（无论信息在何处，模型表现都较稳定）

#### 上下文归因分数

衡量哪些上下文 chunk 对答案有贡献。

**归因精确率（Attribution Precision）**
```
Attribution Precision = Attributed Tokens Actually Relevant / Total Attributed Tokens
```

**归因召回率（Attribution Recall）**
```
Attribution Recall = Relevant Tokens Attributed / Total Relevant Tokens
```

通过注意力权重测量：
```python
def compute_attribution(context_chunks: list[str], answer: str) -> dict:
    """使用注意力权重计算归因分数。"""
    # 获取模型的注意力权重
    attention_weights = model.get_attention(context_chunks + answer)

    # 被归因的 chunks：按注意力取 top-k
    attributed = get_top_k_chunks(attention_weights, k=3)

    # 计算 precision/recall
    relevant = get_ground_truth_relevant_chunks()

    precision = len(attributed ∩ relevant) / len(attributed)
    recall = len(attributed ∩ relevant) / len(relevant)

    return {"precision": precision, "recall": recall}
```

#### 语义连贯性分数

衡量相邻上下文 chunks 之间的语义流动是否自然。

```python
def coherence_score(context_chunks: list[str]) -> float:
    """衡量相邻 chunks 之间的语义流动。"""
    embeddings = [embed(chunk) for chunk in context_chunks]
    transitions = [
        cosine_similarity(embeddings[i], embeddings[i+1])
        for i in range(len(embeddings) - 1)
    ]
    return np.mean(transitions)
```

解释：
- <0.5：主题跳跃过大（编排较差）
- 0.5–0.7：流动中等
- >0.7：语义过渡平滑

#### 上下文冗余率

```python
def redundancy_rate(context_chunks: list[str]) -> float:
    """衡量 chunks 之间的冗余信息。"""
    # 提取语义单元（实体、事实、主张）
    all_units = []
    for chunk in context_chunks:
        units = extract_semantic_units(chunk)
        all_units.extend(units)

    unique_units = set(all_units)

    redundancy = 1 - (len(unique_units) / len(all_units))
    return redundancy
```

目标：冗余率 < 20%

---

### 3. 预算控制器性能指标

#### 触发器准确率

衡量压缩触发逻辑是否正确。

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

目标：准确率 ≥90%

#### 预算耗尽模式

跟踪预算是在什么 token 数附近被耗尽的。

**最优区间**：85–95% 利用率

```python
def evaluate_budget_exhaustion(queries: list) -> dict:
    """分析预算耗尽模式。"""
    exhaustion_counts = {
        "under": 0,      # <70% 利用率
        "optimal": 0,    # 70-95%
        "over": 0,       # >95%（发生截断）
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

目标：optimal rate ≥80%

#### 阈值校准分数

在不同预算阈值下测试系统，并观察性能退化曲线。

```python
def evaluate_threshold_calibration(budget_levels: list[float]) -> dict:
    """绘制性能退化曲线。"""
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

    # 计算退化曲线
    degradation = [
        results[i]["accuracy"] - results[i-1]["accuracy"]
        for i in range(1, len(results))
    ]

    return {
        "results": results,
        "degradation": degradation,
        "is_smooth": all(d > -0.05 for d in degradation)  # 没有断崖
    }
```

目标：平滑退化（无超过 5% 的骤降）

---

### 4. 压缩质量指标

#### 事实一致性分数

使用 NLI 模型检测压缩伪影。

```python
def factual_consistency(original: str, compressed: str) -> float:
    """检查压缩是否引入矛盾。"""
    # 切分为多个主张
    original_claims = extract_claims(original)
    compressed_claims = extract_claims(compressed)

    # 用 NLI 检查一致性
    nli_model = load_nli_model()

    consistent = 0
    for claim in compressed_claims:
        # 检查该主张是否被原文蕴含
        result = nli_model.check_entailment(original, claim)
        if result in ["entailment", "neutral"]:
            consistent += 1

    return consistent / len(compressed_claims)
```

目标：≥0.95（95% 的主张保持一致）

#### 实体保留率

```python
def entity_preservation(original: str, compressed: str) -> dict:
    """跟踪压缩中的实体保留情况。"""
    entities_original = extract_entities(original)
    entities_compressed = extract_entities(compressed)

    # 按实体类型分组
    results = {}
    for entity_type in ["PERSON", "ORG", "DATE", "MONEY", "PRODUCT"]:
        original_set = {e.text for e in entities_original if e.type == entity_type}
        compressed_set = {e.text for e in entities_compressed if e.type == entity_type}

        if original_set:
            preservation = len(compressed_set) / len(original_set)
        else:
            preservation = 1.0  # 没有实体需要保留

        results[entity_type] = preservation

    # Overall
    all_original = {e.text for e in entities_original}
    all_compressed = {e.text for e in entities_compressed}
    results["overall"] = len(all_compressed) / len(all_original) if all_original else 1.0

    return results
```

目标：对关键实体类型（MONEY、DATE、PRODUCT）达到 ≥90%

#### 压缩稳定性

```python
def compression_stability(text: str, n_runs: int = 5) -> float:
    """测试压缩结果的确定性。"""
    compressed_versions = []

    for _ in range(n_runs):
        compressed = compressor.compress(text, budget=500)
        compressed_versions.append(compressed)

    # 计算两两相似度
    similarities = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            sim = compute_similarity(compressed_versions[i], compressed_versions[j])
            similarities.append(sim)

    return np.mean(similarities)
```

目标：≥0.95（高度确定）

---

### 5. 多轮上下文管理指标

#### 状态跟踪准确率

```python
def evaluate_state_tracking(conversations: list) -> float:
    """测试跨对话轮次的状态跟踪。"""
    accuracies = []

    for conv in conversations:
        manager = SupportContextManager()

        for turn_idx, turn in enumerate(conv.turns):
            # 更新状态
            manager.update_state(turn.extracted_info)

            # 每 3 轮验证一次状态
            if turn_idx % 3 == 0:
                tracked_state = manager.get_state()
                ground_truth = conv.ground_truth[turn_idx]

                # 计算准确率
                matches = sum(
                    1 for key in ground_truth
                    if tracked_state.get(key) == ground_truth[key]
                )
                accuracy = matches / len(ground_truth)
                accuracies.append(accuracy)

    return np.mean(accuracies)
```

目标：≥90%

#### 上下文漂移检测率

```python
def evaluate_context_drift(conversations: list) -> dict:
    """衡量上下文变陈旧的速度。"""
    drift_scores = []

    for conv in conversations:
        manager = SupportContextManager()

        for turn_idx, turn in enumerate(conv.turns):
            # 跟踪实体最后一次被提到的时间
            manager.track_mentions(turn.content)

            # 为每个被跟踪实体计算漂移
            for entity_id in manager.tracked_entities:
                turns_since_mention = turn_idx - manager.last_mentioned[entity_id]
                drift_rate = turns_since_mention / len(conv.turns)
                drift_scores.append(drift_rate)

    return {
        "avg_drift": np.mean(drift_scores),
        "high_drift_rate": sum(1 for s in drift_scores if s > 0.5) / len(drift_scores)
    }
```

目标：high_drift_rate < 20%

#### 跨轮一致性

```python
def evaluate_cross_turn_consistency(conversations: list) -> float:
    """测试模型是否在不同轮次给出一致回答。"""
    consistency_scores = []

    for conv in conversations:
        manager = SupportContextManager()

        # 在不同轮次问相同问题
        test_question = "What is the status of my order?"

        for turn_idx in [0, 3, 6, 9]:
            if turn_idx < len(conv.turns):
                context = manager.build_context()
                answer = llm.generate(context + test_question)

                # 保存答案
                if turn_idx == 0:
                    reference_answer = answer
                else:
                    # 与参考答案比较
                    similarity = compute_similarity(answer, reference_answer)
                    consistency_scores.append(similarity)

    return np.mean(consistency_scores)
```

目标：≥0.85

---

### 6. 分层检索指标

#### 路由准确率

测试摘要层是否能把请求正确路由到细节区段。

```python
def evaluate_routing_accuracy(queries: list) -> float:
    """测试分层路由。"""
    correct = 0

    for query_data in queries:
        # 使用摘要层做路由
        summary = hierarchical_manager.get_summary()
        selected_section = hierarchical_manager.route(query_data["query"], summary)

        # 与真值比较
        if selected_section == query_data["relevant_section"]:
            correct += 1

    return correct / len(queries)
```

目标：≥85%

#### 细节检索效率

```python
def evaluate_retrieval_efficiency(queries: list) -> dict:
    """衡量细节层检索的精度。"""
    efficiencies = []
    precisions = []

    for query_data in queries:
        # 检索细节 chunks
        detail_chunks = hierarchical_manager.retrieve_details(query_data["query"])

        # 计算效率
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

目标：efficiency ≥0.7，precision ≥0.8

#### 两阶段延迟

```python
def evaluate_latency(queries: list) -> dict:
    """比较两阶段方案与单阶段方案的延迟。"""
    two_stage_times = []
    single_stage_times = []

    for query in queries:
        # 两阶段方案
        start = time.time()
        summary = hierarchical_manager.get_summary()  # Stage 1
        detail = hierarchical_manager.retrieve_details(query)  # Stage 2
        answer = llm.generate(summary + detail + query)
        two_stage_times.append(time.time() - start)

        # 单阶段基线
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

目标：speedup ≥1.5x

---

### 7. 信息论指标

#### 每个 Token 带来的困惑度下降

```python
def perplexity_reduction_per_token(query: str, context: str) -> float:
    """衡量每个 token 的边际价值。"""
    # 无上下文时的困惑度
    ppl_without = compute_perplexity(query)

    # 有上下文时的困惑度
    ppl_with = compute_perplexity(query, context=context)

    # 每个 token 带来的下降值
    reduction = ppl_without - ppl_with
    tokens = count_tokens(context)

    return reduction / tokens
```

解释：
- >0.01：高价值上下文
- 0.001–0.01：中等价值
- <0.001：低价值上下文（适合裁剪）

#### 互信息分数

```python
def mutual_information_score(context: str, query: str, answer: str) -> float:
    """量化上下文对答案不确定性的降低程度。"""
    # 估计 P(context, answer)
    joint_prob = estimate_joint_probability(context, answer)

    # 估计 P(context), P(answer)
    context_prob = estimate_probability(context)
    answer_prob = estimate_probability(answer)

    # 计算 MI
    mi = joint_prob * np.log(joint_prob / (context_prob * answer_prob))

    return mi
```

MI 越高 = 上下文对答案的信息量越大

---

### 8. 成本效益指标

#### 每单位质量成本

```python
def cost_per_quality_unit(
    accuracy: float,
    avg_tokens: int,
    cost_per_1k_tokens: float = 0.003  # Claude Sonnet pricing
) -> float:
    """计算每单位质量成本。"""
    cost = (avg_tokens / 1000) * cost_per_1k_tokens
    return cost / accuracy
```

示例：
- 基线：5000 tokens，80% 准确率 → $0.015 / 0.80 = $0.01875 per quality unit
- CE 系统：2000 tokens，85% 准确率 → $0.006 / 0.85 = $0.00706 per quality unit
- **提升**：成本降低 62%

#### 盈亏平衡分析

```python
def break_even_analysis(
    ce_implementation_cost: float,  # One-time
    cost_savings_per_1k_queries: float,
    queries_per_month: int
) -> dict:
    """计算 CE 投资何时回本。"""
    monthly_savings = (queries_per_month / 1000) * cost_savings_per_1k_queries

    months_to_break_even = ce_implementation_cost / monthly_savings

    return {
        "monthly_savings": monthly_savings,
        "months_to_break_even": months_to_break_even,
        "annual_roi": (monthly_savings * 12 - ce_implementation_cost) / ce_implementation_cost
    }
```

---

## 评估实现

### 完整评估框架

```python
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class CEMetrics:
    """全面的 CE 评估指标。"""

    # Token 效率
    compression_ratio: float
    information_retention: float
    token_utilization: float

    # 预算管理
    budget_adherence: float
    trigger_accuracy: float

    # 编排质量
    position_bias_variance: float
    coherence_score: float
    redundancy_rate: float

    # 压缩质量
    factual_consistency: float
    entity_preservation: float

    # 多轮（可选）
    state_tracking_accuracy: float = None
    cross_turn_consistency: float = None

    # 分层检索（可选）
    routing_accuracy: float = None
    retrieval_efficiency: float = None

    # 成本效益
    cost_per_quality_unit: float = None

    # 任务表现
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
    """上下文工程系统评估器。"""

    def __init__(self, budget: int, llm_client, embedder):
        self.budget = budget
        self.llm = llm_client
        self.embedder = embedder

    def evaluate_compression(self, original: str, compressed: str) -> dict:
        """评估压缩质量。"""
        return {
            "compression_ratio": count_tokens(original) / count_tokens(compressed),
            "information_retention": bert_score(compressed, original),
            "factual_consistency": nli_consistency(compressed, original),
            "entity_preservation": entity_overlap(compressed, original),
        }

    def evaluate_position_bias(self, test_cases: list) -> float:
        """测试 Lost in the Middle 效应。"""
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
        """测试 budget controller 的决策。"""
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
        """运行完整 CE 评估。"""

        # 压缩评估
        compression_results = []
        for query in test_queries[:20]:  # Sample
            original = load_full_context(query)
            compressed = ce_system.compress(original, budget=self.budget)
            compression_results.append(
                self.evaluate_compression(original, compressed)
            )

        # 位置偏差
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

        # 任务表现
        accuracies = []
        for query in test_queries:
            context = ce_system.build_context(query)
            answer = self.llm.generate(context + query["text"])
            accuracies.append(evaluate_answer(answer, query["ground_truth"]))

        # 汇总指标
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


# 用法
def evaluate_ce_system():
    """评估一个 CE 系统实现。"""
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

## 基准测试方案

### 基准测试 1：压缩策略对比

在标准数据集上比较不同压缩策略。

```python
def benchmark_compression_strategies():
    """比较压缩策略。"""
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

    # 打印对比表
    print_comparison_table(results)
    return results
```

### 基准测试 2：多轮对话压力测试

```python
def benchmark_multi_turn():
    """测试多轮上下文管理。"""
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
                # 测试状态跟踪
                accuracy = verify_state(manager.state, conv.ground_truth[turn_idx])
                results["state_tracking_accuracy"].append(accuracy)

                # 测试漂移
                drift = compute_drift(manager)
                results["context_drift"].append(drift)

        # 测试跨轮一致性
        consistency = test_consistency(manager, conv.test_questions)
        results["consistency"].append(consistency)

    return {k: np.mean(v) for k, v in results.items()}
```

### 基准测试 3：大海捞针测试

上下文窗口压力测试。

```python
def benchmark_niah():
    """Needle in a Haystack test."""
    results = []

    for context_length in [1000, 5000, 10000, 50000]:
        for position in [0.0, 0.25, 0.5, 0.75, 1.0]:
            # 生成 haystack
            haystack = generate_haystack(context_length)

            # 在指定位置插入 needle
            needle = "The secret code is 7294"
            context = insert_at_position(haystack, needle, position)

            # 测试检索
            query = "What is the secret code?"
            answer = llm.generate(context + query)

            success = "7294" in answer
            results.append({
                "context_length": context_length,
                "position": position,
                "success": success
            })

    # 分析结果
    return analyze_niah_results(results)
```

---

## 评估最佳实践

### 1. 建立基线

始终对比以下对象：
- **朴素基线**：无 CE（完整上下文或简单截断）
- **简单基线**：固定大小分块
- **上一版本**：当你在迭代 CE 系统时

### 2. 使用多种指标

不要依赖单一指标：
- 结合效率（压缩）+ 质量（保留率）
- 平衡成本与性能
- 同时测试 token 级与任务级指标

### 3. 测试边界情况

- 超长文档（>50k tokens）
- 超短 query（<10 tokens）
- 超长多轮对话（>20 turns）
- 预算紧急情况（>95% 利用率）

### 4. 真实世界评估

除了合成 benchmark，还要：
- 与真实用户做 A/B 测试
- 监控生产指标
- 持续跟踪成本节省

### 5. 持续监控

```python
class CEProductionMonitor:
    """在生产环境中监控 CE 系统。"""

    def __init__(self):
        self.metrics_history = []

    def log_request(self, query: str, context: str, answer: str):
        """记录请求指标。"""
        self.metrics_history.append({
            "timestamp": time.time(),
            "tokens_used": count_tokens(context),
            "budget": self.budget,
            "utilization": count_tokens(context) / self.budget,
            # ... other metrics
        })

    def get_aggregate_metrics(self, window_hours: int = 24) -> dict:
        """获取某个时间窗口内的聚合指标。"""
        recent = filter_recent(self.metrics_history, window_hours)

        return {
            "avg_utilization": np.mean([m["utilization"] for m in recent]),
            "p95_utilization": np.percentile([m["utilization"] for m in recent], 95),
            "budget_violations": sum(1 for m in recent if m["utilization"] > 1.0),
            # ... other aggregates
        }
```

---

## 结论

CE 专属评估关注：

1. **效率（Efficiency）**：token 预算利用率、压缩效果
2. **编排（Composition）**：位置偏差、连贯性、冗余
3. **质量（Quality）**：事实一致性、实体保留
4. **管理（Management）**：状态跟踪、漂移检测
5. **成本（Cost）**：每单位质量成本、ROI

这些指标之所以能把 CE 与通用 LLM / RAG 评估区分开，是因为它们把“上下文”视为首要优化目标，而不是仅仅把它当作输入。
