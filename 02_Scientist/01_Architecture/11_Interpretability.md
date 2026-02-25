# Mechanistic Interpretability: Understanding What Models Learn

*Prerequisite: [01_Transformer.md](01_Transformer.md), [02_Attention.md](02_Attention.md). Follows the research depth of Anthropic's Circuits work, Sparse Autoencoders (SAE), and the Superposition hypothesis.*

---

## 1. Why Interpretability Matters

| Motivation | Explanation |
|:--|:--|
| **Safety** | Understanding internal representations helps detect deceptive alignment, hidden goals, or dangerous capabilities before deployment |
| **Debugging** | Identify *why* a model hallucinates, refuses incorrectly, or fails on specific inputs |
| **Scientific understanding** | Reverse-engineer the algorithms learned by neural networks |
| **Alignment verification** | Verify that safety training actually changes internal representations, not just surface behavior |

---

## 2. Taxonomy of Interpretability Methods

```
Interpretability
├── Behavioral (black-box)
│   ├── Probing / Linear probes
│   └── Causal interventions (activation patching)
└── Mechanistic (white-box)
    ├── Circuit analysis (reverse-engineering computation graphs)
    ├── Sparse Autoencoders (decomposing representations)
    └── Logit lens / Tuned lens (reading intermediate predictions)
```

---

## 3. Probing: What Do Representations Encode?

### 3.1 Linear Probes

Train a simple linear classifier on frozen intermediate activations to test whether a concept is linearly represented:

$$\hat{y} = \sigma(W_{probe} \cdot h_l + b)$$

Where $h_l$ is the hidden state at layer $l$.

**Key findings**:
- Syntax (POS tags, dependency relations) is encoded in early-to-mid layers.
- Semantics and world knowledge concentrate in mid-to-late layers.
- Factual associations (e.g., "Eiffel Tower → Paris") are localized to specific layers and positions.

### 3.2 Limitations

- A successful probe doesn't prove the model *uses* that information — it may be a byproduct.
- **Causal interventions** (Section 4) are needed to establish that a representation is functionally important.

---

## 4. Causal Methods: Activation Patching

### 4.1 Core Idea

Replace (patch) an activation from one forward pass into another and observe the effect on output. If patching activation $h_l^{(clean)}$ into a corrupted run restores the correct answer, that activation is **causally important**.

### 4.2 Variants

| Method | Procedure | Granularity |
|:--|:--|:--|
| **Activation patching** | Patch entire layer output | Layer-level |
| **Path patching** | Patch along specific computational paths (head → MLP → head) | Circuit-level |
| **Causal tracing** (ROME) | Corrupt subject tokens, restore at each layer, measure recovery | Token × Layer |

### 4.3 Key Result: Factual Recall

Meng et al. (2022) showed that factual associations ("The Eiffel Tower is in [Paris]") are stored in **mid-layer MLP modules** at the **last subject token position**. This led to ROME (Rank-One Model Editing) — directly editing facts by modifying MLP weight matrices.

---

## 5. The Superposition Hypothesis

### 5.1 The Problem

A model with $d$-dimensional residual stream can represent at most $d$ orthogonal features. But models appear to represent **far more** concepts than they have dimensions. How?

### 5.2 The Hypothesis (Elhage et al., 2022)

Models represent $m \gg d$ features by encoding them as **nearly-orthogonal directions** in $d$-dimensional space. This works because:
- Most features are **sparse** (rarely active simultaneously).
- With sparsity, interference between non-orthogonal features is tolerable.

**Analogy**: Compressed sensing — you can recover sparse signals from fewer measurements than the signal dimension.

### 5.3 Implications

- Individual neurons are **polysemantic** (respond to multiple unrelated concepts) because features are distributed across neurons.
- Standard neuron-level analysis is fundamentally limited.
- We need methods to decompose superposed representations → **Sparse Autoencoders**.

---

## 6. Sparse Autoencoders (SAE)

### 6.1 Architecture

An SAE learns to decompose a model's activation $h \in \mathbb{R}^d$ into a sparse set of interpretable features $f \in \mathbb{R}^m$ where $m \gg d$:

$$f = \text{ReLU}(W_{enc}(h - b_{dec}) + b_{enc})$$
$$\hat{h} = W_{dec} \cdot f + b_{dec}$$

**Training objective**:
$$\mathcal{L} = \|h - \hat{h}\|_2^2 + \lambda \|f\|_1$$

The $L_1$ penalty enforces sparsity — only a few features activate for any given input.

### 6.2 Key Results (Anthropic, 2024)

Anthropic trained SAEs on Claude 3 Sonnet's residual stream and discovered:
- **Millions of interpretable features** — each corresponding to a human-understandable concept.
- Features for: Golden Gate Bridge, code bugs, deception, sycophancy, specific languages, safety-relevant concepts.
- **Feature steering**: Clamping a feature's activation to a high value reliably steers model behavior (e.g., making the model talk about the Golden Gate Bridge in every response).

### 6.3 SAE Variants

| Variant | Innovation | Reference |
|:--|:--|:--|
| **Vanilla SAE** | Standard L1-penalized autoencoder | Cunningham et al. (2023) |
| **TopK SAE** | Replace L1 with hard top-k sparsity constraint | Gao et al. (2024) |
| **Gated SAE** | Separate gate network decides which features activate | Rajamanoharan et al. (2024) |
| **Transcoders** | Map from one layer's activations to the next, decomposing MLP computation | Anthropic (2024) |

### 6.4 Open Challenges

- **Completeness**: Do SAE features capture *all* important computation, or do they miss some?
- **Feature splitting**: As SAE width increases, features split into finer sub-features — when to stop?
- **Scaling**: Training SAEs on frontier models (100B+) is computationally expensive.
- **Validation**: How to rigorously verify that discovered features are "real" and not artifacts?

---

## 7. Circuit Analysis

### 7.1 What is a Circuit?

A **circuit** is a minimal subgraph of the model's computational graph that implements a specific behavior. It consists of:
- Specific attention heads and MLP layers
- The connections (residual stream paths) between them

### 7.2 Notable Circuits Discovered

| Circuit | Behavior | Key Components |
|:--|:--|:--|
| **Induction heads** | In-context learning (copy patterns seen earlier) | Two attention heads: "previous token head" + "induction head" |
| **Indirect Object Identification (IOI)** | "Mary gave the book to [John]" → identify indirect object | ~26 heads in 7 classes across GPT-2 small |
| **Greater-Than** | Compare two numbers | Specific MLP neurons + attention heads |
| **Factual recall** | "Eiffel Tower is in [Paris]" | Mid-layer MLPs at last subject token |

### 7.3 Induction Heads: The Foundation of In-Context Learning

**Mechanism** (Olsson et al., 2022):
1. **Head A** (previous token head): Attends from token $B$ to the token before $B$ in a previous occurrence (i.e., to $A$ in the pattern $[A][B]...[A][B]$).
2. **Head B** (induction head): Copies the token that followed $A$ previously.

Together: if the model has seen "$[A][B]$" before, and now sees "$[A]$", it predicts "$[B]$" will follow.

**Significance**: This is believed to be the core mechanism behind in-context learning, and it emerges as a **phase transition** during training.

---

## 8. Logit Lens and Tuned Lens

### 8.1 Logit Lens (nostalgebraist, 2020)

Apply the final unembedding matrix $W_U$ to intermediate residual stream states to "read" what the model is predicting at each layer:

$$p_l = \text{softmax}(W_U \cdot h_l)$$

**Finding**: Early layers predict broad semantic categories; later layers refine to specific tokens.

### 8.2 Tuned Lens (Belrose et al., 2023)

Train a learned affine transformation per layer (instead of using $W_U$ directly), giving more accurate intermediate predictions:

$$p_l = \text{softmax}(A_l \cdot h_l + b_l)$$

---

## 9. Practical Applications

### 9.1 Model Editing (ROME / MEMIT)

Directly edit factual associations by modifying MLP weights:
- **ROME** (Meng et al., 2022): Rank-one update to a single MLP layer.
- **MEMIT** (Meng et al., 2023): Batch editing across multiple layers.

### 9.2 Representation Engineering

Use identified feature directions to steer model behavior without fine-tuning (see [04_Alignment_Frontiers.md](../07_Paper_Tracking/04_Alignment_Frontiers.md) — RepE, Circuit Breakers).

### 9.3 Safety Auditing

- Detect whether safety training creates genuine internal changes or merely surface-level refusal patterns.
- Identify "sleeper agent" features that activate only under specific conditions.

---

## 10. Key References

1. **Elhage et al. (2022)**: *Toy Models of Superposition* — Anthropic.
2. **Olsson et al. (2022)**: *In-context Learning and Induction Heads* — Anthropic.
3. **Meng et al. (2022)**: *Locating and Editing Factual Associations in GPT* (ROME).
4. **Cunningham et al. (2023)**: *Sparse Autoencoders Find Highly Interpretable Features in Language Models*.
5. **Templeton et al. (2024)**: *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* — Anthropic.
6. **Conmy et al. (2023)**: *Towards Automated Circuit Discovery for Mechanistic Interpretability* (ACDC).
7. **Belrose et al. (2023)**: *Eliciting Latent Predictions from Transformers with the Tuned Lens*.
