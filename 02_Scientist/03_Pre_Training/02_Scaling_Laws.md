# Scaling Laws: Theory and Practice for Foundation Models

*Prerequisite: [01_GPT_Evolution.md](01_GPT_Evolution.md).*

---

## 1. Foundations: The Kaplan Scaling Law Derivation

### 1.1 Empirical Observation and Formalization

The scaling law phenomenon was first systematically studied by Kaplan et al. (2020) at OpenAI. The core observation: **test loss decreases predictably with compute budget, model size, and dataset size**.

#### 1.1.1 Mathematical Formulation
For a fixed transformer architecture and training hyperparameters, the test loss $L$ follows:

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

Where:
- $N$: Number of non-embedding parameters
- $D$: Dataset size (tokens)
- $\alpha_N$, $\alpha_D$: Scaling exponents (empirically $\alpha_N \approx 0.076$, $\alpha_D \approx 0.103$)
- $N_c$, $D_c$: Reference values where each term contributes equally
- $L_\infty$: Irreducible loss floor

#### 1.1.2 Derivation from Learning Theory
The scaling law can be derived from **neural tangent kernel (NTK)** theory under certain assumptions:

**Assumption 1 (Smoothness)**: The loss landscape is sufficiently smooth that gradient descent converges to the NTK solution.

**Assumption 2 (Power-law eigen-decay)**: The NTK eigenvalue spectrum decays as $\lambda_i \sim i^{-\beta}$.

**Result**: Under these assumptions, the excess risk scales as:

$$L(N, D) - L_\infty \sim N^{-\frac{\beta-1}{2}} + D^{-\frac{\beta-1}{\beta+1}}$$

Matching this to the empirical form gives theoretical interpretations:
- $\alpha_N = \frac{\beta-1}{2} \Rightarrow \beta \approx 1.152$
- $\alpha_D = \frac{\beta-1}{\beta+1} \Rightarrow \beta \approx 1.228$

The slight mismatch suggests transformer architectures don't perfectly satisfy NTK assumptions.

### 1.2 Compute-Optimal Scaling (Chinchilla Analysis)

#### 1.2.1 Hoffmann et al. (2022) Formulation
Given a fixed compute budget $C$ (measured in FLOPs), where $C \approx 6ND$ (approximation for transformer forward+backward passes), we seek:

$$\min_{N, D} L(N, D) \quad \text{s.t.} \quad 6ND = C$$

**Solution**: Using Lagrange multipliers:

$$\mathcal{L} = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \lambda(6ND - C)$$

Taking derivatives and solving yields:

$$N_{opt} \propto C^{a}, \quad D_{opt} \propto C^{b}$$
$$a = \frac{\alpha_D}{\alpha_N + \alpha_D} \approx 0.57, \quad b = \frac{\alpha_N}{\alpha_N + \alpha_D} \approx 0.43$$

**Empirical correction**: Actual optimal exponents are closer to $a \approx 0.49$, $b \approx 0.51$ due to:
1. Non-embedding vs total parameters
2. Attention KV cache overhead
3. Activation checkpointing effects

#### 1.2.2 Practical Chinchilla Rule
For $C$ FLOPs:
- Allocate $\sim 49\%$ to parameters, $51\%$ to data tokens
- Approximately: $D_{opt} \approx 20 \times N_{opt}$ (20 tokens per parameter)
- Example: 70B model → 1.4T tokens optimal

### 1.3 Error Analysis and Uncertainty

#### 1.3.1 Confidence Intervals
Kaplan's analysis shows **predictive uncertainty**:

$$\sigma(L) \approx 0.02 \times \left(\frac{C}{C_{ref}}\right)^{-0.12}$$

Meaning:
- At $10^{23}$ FLOPs: $\sigma \approx 0.02$ nats
- At $10^{26}$ FLOPs (GPT-4 scale): $\sigma \approx 0.015$ nats

**Implication**: Scaling predictions have $\sim 1.5-2\%$ relative error at frontier scales.

#### 1.3.2 Breakdown Conditions
Scaling laws break when:
1. **Architecture changes**: MoE, sparse attention alter $C \approx 6ND$ relation
2. **Data quality limits**: $\alpha_D$ decreases when adding low-quality data
3. **Optimization failures**: Loss spikes, divergence not captured by theory

---

## 2. Industry Case Studies

### 2.1 GPT-4 Scaling Strategy (OpenAI, 2023)

#### 2.1.1 Compute Allocation
GPT-4 technical report (Section 2.1) reveals:

- **Total compute**: $\sim 2.1 \times 10^{25}$ FLOPs
- **Parameter allocation**: Following near-Chinchilla optimal
- **Architectural innovation**: Mixture of 16 Experts (MoE) with $\sim 1.8T$ total parameters, $\sim 280B$ active per token

#### 2.1.2 Data Pipeline Scaling
- **Initial data**: 13T tokens after filtering
- **Quality filtering**: Multi-stage classifier pipeline (Section 2.2)
- **Domain mixture**: Code (15%), STEM (20%), Web (50%), Books (10%), Other (5%)
- **Curriculum learning**: Progressive domain mixing (code-heavy early)

#### 2.1.3 Infrastructure Details
```yaml
# OpenAI GPT-4 Training Configuration (Estimated)
hardware:
  gpu_type: "A100"  # Later H100 for final stages
  gpu_count: 25,000  # Estimated from timeline and throughput
  interconnect: "NVIDIA NVLink + InfiniBand"
  peak_flops: ~340 PFLOPS

training:
  batch_size: 3.2M tokens  # Micro-batch 4K seq × 800 GPUs
  gradient_accumulation: 8 steps
  effective_batch: 25.6M tokens
  learning_rate: 1.2e-4 with cosine decay
  duration: ~90 days continuous
```

#### 2.1.4 Cost Analysis
- **Direct compute**: $62.3M GPU-hours × $2/hour = ~$125M
- **Infrastructure**: Network, storage, power ~$25M
- **Human capital**: Research, engineering, operations ~$50M
- **Total estimate**: ~$200M per training run

### 2.2 Llama 3 Scaling (Meta, 2024)

#### 2.2.1 Compute-Dataset Co-Scaling
Meta's innovation: **Scaling dataset faster than parameters**

- **Llama 2 (70B)**: 2T tokens (28:1 tokens:parameter)
- **Llama 3 (70B)**: 15T+ tokens (214:1 tokens:parameter)
- **Result**: Llama 3 70B outperforms Llama 2 70B by >15% on benchmarks

#### 2.2.2 Data Quality Scaling Law
Meta discovered: **Quality exponent $\gamma$**

$$L(D, Q) = \left(\frac{D_c}{D}\right)^{\alpha_D} \times Q^{-\gamma}$$

Where $Q$ is data quality score (0-1). For Llama 3:
- $\gamma \approx 0.3$: Quality matters more than quantity beyond 1T tokens
- **Strategy**: Invest in better filtering over more raw data

#### 2.2.3 Infrastructure Scaling
```python
# Meta's Training Infrastructure (from technical blogs)
class MetaTrainingCluster:
    def __init__(self):
        self.gpus = 24,000  # H100 equivalents
        self.network = "200Gbps InfiniBand"
        self.storage = "500PB flash + 10PB RAM cache"
        self.throughput = "1.2M tokens/sec sustained"

    def training_efficiency(self):
        # Megatron-LM + PyTorch FSDP
        return {
            "gpu_utilization": 0.52,  # Attention kernels limit
            "mfuf": 0.48,  # Model FLOPs Utilization
            "communication_fraction": 0.18
        }
```

### 2.3 PaLM 2 Scaling (Google, 2023)

#### 2.3.1 Compute-Optimal Frontier Pushing
Google's approach: **Push beyond Chinchilla with better architectures**

- **PaLM 1 (540B)**: 780B tokens (1.4:1 ratio) - undertrained
- **PaLM 2 (340B)**: 3.6T tokens (10.6:1 ratio) - near optimal
- **Architecture efficiency**: Better attention + FFN design allows ~1.6× FLOPs reduction

#### 2.3.2 The Pathways Scaling Advantage
TPU v4/v5 advantages for scaling:
- **3D torus mesh**: Better scaling than GPU tree networks
- **Custom attention units**: Higher MFU for attention (~65% vs ~52% on GPUs)
- **Bfloat16 native**: No precision conversion overhead

**Result**: PaLM 2 training cost ~40% lower than equivalent GPU training.

---

## 3. Distributed Training Mathematics

### 3.1 Parallelism Strategy Optimization

#### 3.1.1 3D Parallelism Formulation
For model with $L$ layers, hidden size $H$, sequence length $S$, batch size $B$:

**Memory per GPU**:
$$M = \underbrace{\frac{4H^2}{P_T}}_{\text{weights}} + \underbrace{\frac{4BSH}{P_P}}_{\text{activations}} + \underbrace{\frac{24H^2}{P_T P_D}}_{\text{optimizer states}}$$

Where:
- $P_T$: Tensor parallelism degree
- $P_P$: Pipeline parallelism degree
- $P_D$: Data parallelism degree

**Optimal Configuration**: Solve $\min_{P_T, P_P, P_D} \text{Time}(P_T, P_P, P_D)$ s.t. $P_T P_P P_D = \text{Total GPUs}$

#### 3.1.2 Pipeline Bubble Analysis
Pipeline efficiency:

$$\eta_{\text{pipeline}} = \frac{P_P}{P_P + (P_P - 1) \cdot \frac{t_{\text{bubble}}}{t_{\text{compute}}}}$$

Where $t_{\text{bubble}} = (P_P - 1) \cdot (t_{\text{fwd}} + t_{\text{bwd}})$.

**Optimal micro-batch count**:
$$m_{\text{opt}} = \left\lceil \frac{P_P \cdot (t_{\text{fwd}} + t_{\text{bwd}})}{t_{\text{bubble}}}\right\rceil$$

### 3.2 Memory Optimization Theory

#### 3.2.1 ZeRO Optimization Levels
**ZeRO-1** (optimizer state sharding):
$$M_{\text{opt}} = \frac{M_{\text{opt,full}}}{P_D}$$

**ZeRO-2** (+ gradient sharding):
$$M_{\text{grad}} = \frac{M_{\text{grad,full}}}{P_D}$$

**ZeRO-3** (+ parameter sharding):
$$M_{\text{param}} = \frac{M_{\text{param,full}}}{P_D}$$

**Total memory reduction**:
$$M_{\text{total}} = M_{\text{activations}} + \frac{M_{\text{weights}} + M_{\text{opt}} + M_{\text{grad}}}{P_D}$$

#### 3.2.2 Activation Checkpointing Trade-off
Store activations at $k$ checkpoints instead of all $L$ layers:

**Memory**: $M_{\text{act}} = \frac{L}{k} \times M_{\text{layer-act}}$

**Compute overhead**: $\frac{k-1}{k} \times \text{recomputation FLOPs}$

**Optimal $k$**: Minimize $\text{Time} = t_{\text{compute}}(k) + t_{\text{memory}}(k)$

---

## 4. Beyond Simple Scaling: Recent Advances

### 4.1 Multimodal Scaling Laws

#### 4.1.1 Cross-Modal Transfer
For vision-language models (e.g., GPT-4V):

$$L_{\text{VL}}(N, D_t, D_v) = L_t(N, D_t) + L_v(N, D_v) - \gamma \sqrt{L_t L_v}$$

Where $\gamma \approx 0.3$ captures **positive transfer** between modalities.

#### 4.1.2 Modality-Specific Scaling
- **Image tokens**: Scale as $\alpha_{\text{image}} \approx 0.091$ (slightly better than text)
- **Audio tokens**: $\alpha_{\text{audio}} \approx 0.085$
- **Video tokens**: $\alpha_{\text{video}} \approx 0.088$

### 4.2 Reasoning Scaling Laws

#### 4.2.1 The Reasoning Scaling Gap
DeepSeek-R1 and OpenAI o1 reveal: **Reasoning scales differently**

$$L_{\text{reasoning}}(N, D, T) = L_{\text{knowledge}}(N, D) \times T^{-\delta}$$

Where:
- $T$: Inference-time compute (CoT length, search budget)
- $\delta \approx 0.15$: Reasoning compute exponent

**Implication**: $100\times$ more inference compute ≈ $2.8\times$ better reasoning

#### 4.2.2 Training-Inference Compute Trade-off
**Theorem**: For fixed total compute $C_{\text{total}} = C_{\text{train}} + C_{\text{infer}}$:

$$\text{Optimal allocation} = \frac{C_{\text{train}}}{C_{\text{infer}}} \approx \frac{\alpha_D}{\delta} \approx 0.69$$

**Interpretation**: Spend ~41% on training, 59% on inference for reasoning tasks.

### 4.3 Data Quality Scaling Law (2024)

#### 4.3.1 Quality-Aware Formulation
Let $Q(D)$ = average quality of dataset size $D$ (0-1 scale). Then:

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D \cdot Q(D)}\right)^{\alpha_D}$$

Typically $Q(D) \sim D^{-\epsilon}$ with $\epsilon \approx 0.1$.

#### 4.3.2 Optimal Quality-Quantity Trade-off
**Problem**: Choose $D$, $Q$ to minimize loss given collection cost $C(D, Q)$.

**Solution**: For linear cost $C = c_D D + c_Q Q$:

$$\frac{D}{Q} \approx \left(\frac{\alpha_D c_Q}{\epsilon c_D}\right)^{\frac{1}{1+\epsilon}}$$

**Industry application**: Meta spends ~3× more on data quality per token than OpenAI.

---

## 5. Practical Implementation Framework

### 5.1 Scaling Decision Checklist

#### 5.1.1 Compute-Constrained Projects
```python
def compute_optimal_allocation(total_flops: float) -> Dict:
    """Chinchilla-optimal allocation given compute budget."""
    # Constants from Hoffmann et al.
    A = 406.4  # Constants from paper
    B = 410.7
    E = 0.34

    N_opt = A * (total_flops ** 0.5)
    D_opt = B * (total_flops ** 0.5)

    # Adjust for modern architectures
    if use_moe:
        N_opt *= 1.8  # MoE parameter overhead
        D_opt *= 0.9  # Slightly less data needed

    return {"parameters": int(N_opt), "tokens": int(D_opt)}
```

#### 5.1.2 Memory-Constrained Projects
```python
def max_model_size(gpu_memory_gb: float, seq_len: int,
                   use_zero3: bool = True) -> int:
    """Calculate maximum trainable model size."""
    # Memory breakdown (bytes)
    activation_memory = 2 * seq_len * hidden_size * batch_size * 2  # bfloat16
    weight_memory = 2 * num_params  # bfloat16

    if use_zero3:
        optimizer_memory = 12 * num_params / num_gpus  # Adam 2+4+6 states
    else:
        optimizer_memory = 12 * num_params

    total_memory = activation_memory + weight_memory + optimizer_memory

    # Solve for num_params
    return int((gpu_memory_gb * 1e9 - activation_memory) /
               (2 + (12/num_gpus if use_zero3 else 12)))
```

### 5.2 Monitoring and Validation

#### 5.2.1 Loss Scaling Verification
During training, verify scaling law predictions:

$$\frac{L_{\text{actual}} - L_{\text{predicted}}}{L_{\text{predicted}}} < 0.05$$

If deviation > 5%:
1. Check data quality (perplexity drift)
2. Verify optimizer stability (gradient norm)
3. Inspect activation statistics (mean/variance drift)

#### 5.2.2 Early Stopping Criterion
Stop training when:

$$\frac{dL}{dD} < \frac{\alpha_D \cdot L}{D} \times 0.5$$

Meaning: Improvement rate less than 50% of scaling law prediction.

### 5.3 Cost-Performance Optimization

#### 5.3.1 Cloud Cost Model
```python
def training_cost(flops: float, hardware: str) -> float:
    """Estimate training cost."""
    rates = {
        "a100_80gb": 2.00,  # $/hour
        "h100_80gb": 4.00,
        "tpu_v4": 3.50,
        "tpu_v5": 6.00
    }

    # FLOPs to GPU-hours (assuming 50% MFU)
    hours = flops / (gpu_flops[hardware] * 0.5 * 3600)

    return hours * rates[hardware] * 1.3  # 30% overhead
```

#### 5.3.2 ROI Calculation
For commercial model:

$$\text{ROI} = \frac{\text{Revenue} \times \text{PerformanceGain} - \text{TrainingCost}}{\text{TrainingCost}}$$

Where $\text{PerformanceGain} = 1 - \frac{L_{\text{new}}}{L_{\text{old}}}$.

**Rule of thumb**: ROI > 3 for commercial viability.

---

## 6. Open Problems and Research Directions

### 6.1 Theoretical Gaps

1. **Architecture-dependent scaling exponents**: No theory predicts $\alpha_N$, $\alpha_D$ from architecture
2. **Emergent abilities scaling**: Why do abilities appear suddenly at scale?
3. **Scaling law breakdown**: When and why do laws fail?

### 6.2 Practical Challenges

1. **Data exhaustion**: Web data growth slowing (3% year vs 10× model scale/year)
2. **Energy constraints**: 100MW+ training runs approaching practical limits
3. **Economic scaling**: $200M/training run may be economic limit for most

### 6.3 Future Scaling Strategies

1. **Algorithmic efficiency**: Better architectures (2× efficiency every 2 years)
2. **Synthetic data**: Overcoming data exhaustion
3. **Specialized hardware**: 10× efficiency gains possible

---

## 7. Key References

1. **Kaplan et al. (2020)**: *Scaling Laws for Neural Language Models* - Foundational
2. **Hoffmann et al. (2022)**: *Training Compute-Optimal Large Language Models* - Chinchilla
3. **Caballero et al. (2022)**: *Broken Neural Scaling Laws* - Breakdown analysis
4. **Biderman et al. (2023)**: *Datasheets for Datasets* - Data quality effects
5. **Muennighoff et al. (2023)**: *Scaling Data-Constrained Language Models*
6. **OpenAI (2023)**: *GPT-4 Technical Report* - Industry implementation
7. **Meta (2024)**: *Llama 3 Model Card* - Open-weight scaling
8. **Google (2023)**: *PaLM 2 Technical Report* - Efficient scaling

---

## 8. Appendix: Mathematical Derivations

### 8.1 Kaplan Law Derivation (Full)

Starting from NTK regression with $n$ training points, $d$ parameters:

$$L(n, d) = \sum_{i=1}^\infty \frac{\lambda_i}{1 + n\lambda_i/d} + \sigma^2$$

With power-law spectrum $\lambda_i = ci^{-\beta}$:

$$L(n, d) \sim \int_1^\infty \frac{ci^{-\beta}}{1 + nci^{-\beta}/d} di + \sigma^2$$

Change variables $x = nci^{-\beta}/d$, solve asymptotically:

$$L(n, d) \sim d^{-\frac{\beta-1}{2}} + n^{-\frac{\beta-1}{\beta+1}} + \sigma^2$$

Matching to $L(N, D) = (N_c/N)^{\alpha_N} + (D_c/D)^{\alpha_D} + L_\infty$ gives the result.

### 8.2 Chinchilla Optimal Allocation Proof

Minimize $L(N, D) = AN^{-\alpha_N} + BD^{-\alpha_D}$ subject to $ND = C/6$.

Lagrangian: $\mathcal{L} = AN^{-\alpha_N} + BD^{-\alpha_D} + \lambda(ND - C/6)$

FOC:
1. $\frac{\partial\mathcal{L}}{\partial N} = -\alpha_N AN^{-\alpha_N-1} + \lambda D = 0$
2. $\frac{\partial\mathcal{L}}{\partial D} = -\alpha_D BD^{-\alpha_D-1} + \lambda N = 0$
3. $ND = C/6$

Divide (1) by (2):

$$\frac{\alpha_N A N^{-\alpha_N-1}}{\alpha_D B D^{-\alpha_D-1}} = \frac{D}{N}$$

Simplify: $\frac{\alpha_N}{\alpha_D} \cdot \frac{A}{B} \cdot N^{-\alpha_N} D^{\alpha_D} = D^2 N^{-2}$

Using $D = C/(6N)$:

$$\frac{\alpha_N}{\alpha_D} \cdot \frac{A}{B} \cdot N^{-\alpha_N} \left(\frac{C}{6N}\right)^{\alpha_D} = \left(\frac{C}{6N}\right)^2 N^{-2}$$

Solve for $N$: $N \propto C^{\alpha_D/(\alpha_N+\alpha_D)}$, similarly $D \propto C^{\alpha_N/(\alpha_N+\alpha_D)}$.

---

*Document follows Stanford CS224N Problem Set 4 (2024) mathematical style and OpenAI technical report detail level.*