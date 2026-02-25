# Pre-training Evaluation: Monitoring, Diagnostics, and Probing

*Prerequisite: [02_Scaling_Laws.md](02_Scaling_Laws.md), [05_Data_Pipelines.md](05_Data_Pipelines.md).*

---

## 1. Real-time Training Diagnostics

Evaluating a model **during** pre-training is critical for detecting instabilities early and predicting final performance.

### 1.1 The Loss Curve: Reading the Tea Leaves
The training loss is the primary health indicator. However, raw loss is insufficient.

#### 1.1.1 Predicted vs. Actual Loss
Use the **Scaling Law** (Kaplan/Chinchilla) to establish a baseline.
- **Good**: $L_{actual}$ stays within 1-2% of $L_{predicted}$.
- **Warning**: Loss deviates upward (suggests data quality or architecture bottleneck).
- **Crisis**: Loss spikes (see [Training Stability](./08_Training_Stability.md)).

#### 1.1.2 Validation Split Strategy
Instead of one validation set, use **stratified val-sets**:
| Val Set | Purpose | Health Check |
|:--|:--|:--|
| **Held-out Web** | Generalization | Overfitting detector |
| **Cleaned Wiki/arXiv** | Knowledge acquisition | Efficiency of learning "fact-dense" data |
| **Synthetic Logic** | Reasoning capacity | Detecting emergence of logic |

### 1.2 Perplexity (PPL) Analysis
Perplexity is the exponentiated average negative log-likelihood:
$$PPL(X) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(x_i | x_{<i})\right)$$

**Key Milestone**: When PPL on code-only validation sets starts dropping faster than text-only, the model is entering the "Reasoning Phase".

---

## 2. Evaluation through Probing (Downstream Tasks)

Since we cannot run 500 benchmarks every hour, we use **zero-shot probing** on a subset of tasks.

### 2.1 The "Anchor" Benchmarks
OpenAI and Meta typically monitor these during training:

| Benchmark | Capability | Sensitivity |
|:--|:--|:--|
| **ARC-Challenge** | Common sense reasoning | High (emerges early) |
| **MMLU** | World knowledge | Medium (scales linearly with tokens) |
| **GSM8K** | Mathematical logic | Low (emerges late, "Step-wise" behavior) |
| **HumanEval** | Code generation | Variable (highly dependent on mixture) |

### 2.2 Probing with Linear Heads
If zero-shot performance is noisy, attach a **linear probe** to the frozen intermediate layers to check if the model "knows" facts (e.g., entity types) before it can articulate them.

## 3. Mechanistic Interpretability: Probing the "Why"

Beyond benchmarks, we evaluate if the model is learning *circuits* or just memorizing.

### 3.1 Induction Heads: The Engine of ICL
An **Induction Head** is a specific circuit that performs the operation: "If I see $[A][B] \dots [A]$, predict $[B]$."
- **Detection**: Check if attention patterns show a strong diagonal shift $(i \to j-1)$ in late pre-training.
- **Significance**: The emergence of induction heads directly correlates with the "phase change" where In-Context Learning (ICL) begins to work.

### 3.2 Feature Visualization (SAE)
Use **Sparse Autoencoders (SAEs)** to decompose hidden states into interpretable features (e.g., "Python syntax", "Legal tone").
- **Metric**: $L_0$ sparsity vs. Reconstruction MSE.

---

## 4. Quantifying Emergent Abilities

Emergence is often an artifact of non-linear metrics. Evaluation must account for this.

### 3.1 Measuring "Grokking" in Pre-training
Observe the gradient norm of specialized layers. A sudden spike followed by a drop in validation error on specific tasks (like modular arithmetic) indicates the model has switched from "memorization" to "generalization".

### 3.2 Probabilistic Logic Gates
Monitor the probability ratio $P(\text{correct\_logic}) / P(\text{hallucinated\_logic})$.
If the ratio $> 10$ consistently, the capability is "Unlocked".

---

## 4. Predicting Final Benchmark Performance

### 4.1 Log-Log Linear Extrapolation
Benchmark performance often follows a predictable path relative to training compute $C$:
$$\text{Metric}(C) \approx a \cdot \log(C) + b$$

By training a **1B model** to completion, one can predict the **70B model's** performance at 10% of its training duration with high accuracy.

### 4.2 Cross-Task Transfer Evaluation
Evaluate how learning Task A (e.g., LaTeX) accelerates learning Task B (e.g., Python). Positive transfer is the hallmark of a "Foundation" model.

---

## 5. Implementation: Evaluation Pipeline

```python
class PretrainingEvaluator:
    """Industrial pre-training evaluation suite."""

    def __init__(self, model, tokenizer):
        self.metrics = {
            "PPL": PerplexityMetric(),
            "ARC": ZeroShotARC(),
            "MMLU": ZeroShotMMLU(subset_size=100) # Fast probe
        }

    def run_diagnostics(self, step):
        # 1. Compute Loss and PPL
        stats = self.get_loss_stats()

        # 2. Check for "Capabality Jumps"
        if step % 5000 == 0:
            capabilities = self.probe_capabilities()

        # 3. Predict final score
        prediction = self.extrapolate_final_mmlu(stats)

        return stats, capabilities, prediction
```

---

## 6. Key References

1. **Wei et al. (2022)**: *Emergent Abilities of Large Language Models*.
2. **OpenAI (2023)**: *GPT-4 Technical Report* (Evaluation Section).
3. **Burnell et al. (2023)**: *Rethink Evaluation: On the Stability and Robustness of Foundation Model Evaluation*.
4. **Meta AI (2024)**: *Llama 3 Technical Report* (Probing during training).

---

*Document follows the evaluation standards of the Stanford Center for Research on Foundation Models (CRFM).*
