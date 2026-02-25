# Architecture Paradigms: Dense vs MoE vs Hybrid-MoE

*Prerequisite: [08_Dense_Arch.md](08_Dense_Arch.md), [09_MoE_Arch.md](09_MoE_Arch.md).*

---

## 1. Overview

Modern LLM architectures can be categorized into three paradigms based on how they utilize parameters during inference:

| Paradigm | Total Params | Active Params | Compute Efficiency | Communication Overhead |
|----------|-------------|---------------|-------------------|----------------------|
| **Dense** | All active | 100% | Low at scale | Minimal |
| **MoE** | Sparse activation | ~5-27% | High | High (All-to-All) |
| **Hybrid-MoE** | Mixed | Varies | Best tradeoff | Moderate |

## 2. Dense Transformer

In a Dense Transformer, every parameter participates in every forward pass. The compute cost scales linearly with parameter count.

**Characteristics:**
- All layers are fully connected — every token activates every parameter
- Scaling requires proportionally more compute (FLOPs ∝ Parameters)
- Simple to implement and train — no routing complexity
- Strong, stable representation learning across all layers

**When to choose Dense:**
- Smaller model scales where compute is not the bottleneck
- Tasks requiring consistent, uniform processing across all inputs
- When simplicity and training stability are prioritized

**Representative models:** GPT-3, Llama 2/3, Mistral 7B

## 3. MoE (Mixture of Experts) Transformer

MoE replaces the FFN layer with multiple expert sub-networks and a router. Only a subset of experts (top-K) are activated per token, decoupling total parameters from compute cost.

**Characteristics:**
- Sparse activation — each token only uses K out of N experts
- Massive parameter capacity with low inference FLOPs
- Requires load balancing to prevent expert collapse
- High All-to-All communication overhead in distributed training

**When to choose MoE:**
- Large-scale models where compute budget is constrained
- Scenarios requiring massive knowledge capacity (multilingual, multi-domain)
- When inference latency matters more than memory footprint

**Representative models:** Mixtral 8x7B, DeepSeek-V2/V3, Switch Transformer

## 4. Hybrid-MoE Transformer

Hybrid-MoE interleaves dense layers and MoE layers within the same model. This combines the stable representation learning of dense layers with the efficient capacity scaling of MoE layers.

**Characteristics:**
- Not every layer is MoE — dense and MoE layers alternate (e.g., every other layer)
- Dense layers provide a stable "backbone" for representation learning
- MoE layers add capacity efficiently where it matters most
- Reduces All-to-All communication overhead compared to pure MoE

**When to choose Hybrid-MoE:**
- Best balance of quality and efficiency for a given compute budget
- When pure MoE suffers from communication bottlenecks
- Large-scale production models targeting cost-effectiveness

**Representative models:** Snowflake Arctic (10B dense + 128 experts, 480B total, ~17B active)

## 5. Snowflake's Experimental Findings

Snowflake AI Research conducted systematic experiments comparing the three paradigms:

### 5.1 Number of Experts

Increasing the number of experts improves performance, but with **diminishing returns**. Beyond a certain point, adding more experts yields marginal gains while increasing routing complexity and communication cost.

### 5.2 MoE Layer Frequency

Not every layer benefits equally from being MoE. Experiments showed that **interleaving dense and MoE layers** (e.g., applying MoE to every other layer) can match or outperform full-MoE architectures while reducing overhead.

### 5.3 DBRX Comparison

Training Dense, MoE, and Hybrid-MoE models at comparable scale with the same data:
- **Hybrid-MoE consistently outperformed** both pure Dense and pure MoE for a given compute budget
- Pure MoE showed advantages in total capacity but suffered from communication overhead
- Pure Dense was simplest but least compute-efficient at scale

## 6. Architecture Decision Matrix

| Factor | Dense | MoE | Hybrid-MoE |
|--------|-------|-----|------------|
| Training simplicity | ★★★ | ★ | ★★ |
| Inference FLOPs | High | Low | Medium |
| Memory footprint | Proportional | High (all experts loaded) | High |
| Communication cost | Low | High (All-to-All) | Moderate |
| Scaling efficiency | Linear | Sub-linear | Best tradeoff |
| Quality per FLOP | Baseline | Good | Best |

## 7. Summary

The evolution from Dense → MoE → Hybrid-MoE reflects a progressive understanding of how to allocate parameters efficiently:

- **Dense** is the foundation — simple, stable, but expensive to scale
- **MoE** breaks the compute-parameter coupling but introduces routing and communication complexity
- **Hybrid-MoE** combines the best of both worlds, using dense layers for stable representations and MoE layers for efficient capacity expansion

For most large-scale production scenarios, Hybrid-MoE currently represents the optimal architecture choice.

## 8. Key References

1. **Fedus et al. (2022)**: *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*.
2. **Jiang et al. (2024)**: *Mixtral of Experts*.
3. **Snowflake AI Research (2024)**: *Snowflake Arctic: The Best LLM for Enterprise AI — Efficiently Intelligent, Truly Open*.
4. **Raschka (2024)**: *The Big LLM Architecture Comparison*.
