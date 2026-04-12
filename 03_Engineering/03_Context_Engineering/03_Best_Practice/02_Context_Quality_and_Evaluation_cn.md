# 上下文质量与评估

*前置要求：[01_Context_Architecture_Patterns_cn.md](01_Context_Architecture_Patterns_cn.md)。*

---

上下文质量决定了模型是否在正确的顺序上、以正确的数量，看到了正确的信息。上下文质量差，是 LLM 应用失败最常见的根因。

## 1. 上下文质量的四个维度

| 维度 | 定义 | 失败模式 |
| :--- | :--- | :--- |
| **相关性（Relevance）** | 注入内容与当前 query 相关 | 不相关的 RAG chunks 分散模型注意力 |
| **完整性（Completeness）** | 回答所需信息都已存在 | 模型会对缺失事实产生幻觉 |
| **简洁性（Conciseness）** | 没有冗余或低价值内容 | 浪费 token、稀释注意力 |
| **排序（Ordering）** | 最重要内容位于最优位置 | 触发 Lost in the Middle 失效 |

这些维度彼此张力很强：追求最大完整性，往往会损害简洁性。上下文工程的艺术，就在于找到合适平衡。

---

## 2. 评估方法

### 2.1 Needle-in-a-Haystack（NIAH）

在长文档（haystack）中已知位置插入一个特定事实（needle），再要求模型取回它。

**设置**：
```python
def niah_test(model_fn, context_length: int, needle_depth: float) -> bool:
    """
    needle_depth: 0.0 = start, 0.5 = middle, 1.0 = end
    Returns True if model correctly retrieves the needle.
    """
    needle = "The secret project codename is AURORA."
    haystack = generate_filler_text(context_length)
    insert_pos = int(len(haystack) * needle_depth)
    context = haystack[:insert_pos] + needle + haystack[insert_pos:]
    response = model_fn(context, "What is the secret project codename?")
    return "AURORA" in response
```

**解释**：
- 所有深度得分都 > 95% → 该上下文长度下模型较可靠
- 在 40–60% 深度掉分 → 出现 Lost in the Middle；应使用 Sandwich Pattern
- 超过某个 token 长度后掉分 → 该长度就是你的有效上下文上限

### 2.2 上下文相关性评分（LLM-as-Judge）

使用一个便宜模型来评估检索上下文是否与 query 相关。

**Prompt 模板**：
```
You are evaluating whether retrieved context is relevant to a user query.

Query: {query}

Retrieved Context:
{context}

Score the relevance from 0 to 1:
- 1.0: Directly answers or strongly supports answering the query
- 0.5: Partially relevant, contains some useful information
- 0.0: Irrelevant, does not help answer the query

Output only a JSON object: {"score": <float>, "reason": "<one sentence>"}
```

**阈值**：注入前，丢弃相关性分数 < 0.3 的 chunks。

### 2.3 上下文利用率

衡量模型在回答中实际引用了注入上下文中的多少内容。

**方法**：生成结束后，使用归因模型（或 LLM judge）识别上下文中哪些部分被用到了。

```
Utilization Rate = (tokens from context cited in response) / (total injected context tokens)
```

**Benchmark**：调优良好的 RAG 系统利用率应 > 40%。低于 20% 通常意味着注入过量。

### 2.4 压缩质量（ROUGE-L）

在评估压缩策略时，应测量信息保留情况：

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = scorer.score(original_text, compressed_text)
rouge_l = scores['rougeL'].fmeasure
# Target: ROUGE-L > 0.6 for acceptable compression quality
```

---

## 3. 上下文排错清单

当模型给出错误或不完整答案时，按这个顺序系统诊断：

1. **答案是否根本不在上下文里？**
   - 在组装后的上下文中搜索期望答案。
   - 如果不存在 → 这是检索问题（RAG / memory），不是上下文工程问题。

2. **答案是否埋在长上下文的中间？**
   - 检查相关 chunk 的位置。
   - 如果是 → 应用 Sandwich Pattern，或把 chunk 移到结尾。

3. **上下文是否过满？**
   - 检查输入 token 数与窗口大小的比值。
   - 如果 >85% → 提高压缩力度。

4. **是否存在冲突信息？**
   - 在上下文中搜索自相矛盾的陈述。
   - 如果有 → 去重，或加入显式的冲突处理指令。

5. **System prompt 是否被遵守？**
   - 用最小上下文测试（仅 system prompt + query）。
   - 如果最小上下文时能遵守，而完整上下文时不能 → 说明上下文覆盖了 system prompt。

6. **压缩是否丢失了关键信息？**
   - 对比压缩版与未压缩版上下文下的模型输出。
   - 如果未压缩版本正确 → 说明压缩质量过低。

---

## 4. 上下文策略的 A/B 测试方法

为了验证上下文变更，应运行可控实验：

**要跟踪的指标**：
| 指标 | 测量方式 |
| :--- | :--- |
| Answer correctness | LLM-as-Judge 或人工评测测试集 |
| Context relevance score | 自动化（见 §2.2） |
| Input token count | API 响应元数据 |
| Cost per query | Token count × price |
| Latency (TTFT) | API timing |
| User satisfaction | 点赞/点踩、会话长度 |

**最小样本量**：每个变体 200–500 个 query，才能在正确性指标上获得统计显著性。

**发布策略**：先 shadow mode（记录两种方案，仅向用户展示控制组），再 10% → 50% → 100%。

---

## 关键参考文献

1. **Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation.** *arXiv:2309.15217*.
2. **Hsieh, C., et al. (2024). RULER: What's the Real Context Size of Your Long-Context Language Models?** *arXiv:2404.06654*.
3. **Gao, Y., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey.** *arXiv:2312.10997*.
