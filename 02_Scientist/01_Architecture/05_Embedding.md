# Embedding Strategies for Large Language Models

*Prerequisite: [04_Tokenizer.md](04_Tokenizer.md). For continual pre-training context, see [../03_Pre_Training/11_Continual_Pre_Training.md](../03_Pre_Training/11_Continual_Pre_Training.md).*

---

## 1. From Discrete IDs to Continuous Vectors

Tokenization handles segmentation; the Embedding layer assigns meaning. Each Token ID maps to a $d$-dimensional continuous vector.

- **Dimensionality**: Common choices include 4096 (Llama 2/3) or higher, providing sufficient capacity for subtle semantic distinctions.
- **Weight Tying**: Many modern architectures share weights between the input Embedding layer and the output LM Head, reducing parameters by ~30% and accelerating convergence.

---

## 2. Tokenizer Fertility

**Fertility** measures the coupling efficiency between a tokenizer and the Embedding space.

| Aspect | Description |
|:--|:--|
| **Definition** | Ratio of tokens generated per word or character |
| **Efficiency Impact** | Lower fertility → each token carries more information → faster inference, lower memory |
| **Multilingual Challenge** | English-centric tokenizers produce high fertility for CJK languages, degrading efficiency and consuming context window rapidly |

---

## 3. Vocabulary Expansion for Vertical Domains

When general-purpose models are applied to specialized domains (healthcare, legal, finance), the default vocabulary's fertility rises significantly. Vocabulary expansion becomes necessary.

### 3.1 The Random Initialization Trap

Directly assigning random vectors to new tokens introduces "semantic noise" that disrupts the model's stable latent space — causing a sharp loss spike at the start of training.

### 3.2 Subword-Based Averaging (Cold Start Solution)

Initialize the embedding of a new token (e.g., `myocardial`) with the average of its original subword components' embeddings (e.g., `myo` + `car` + `dial`).

$$\mathbf{e}_{\text{new}} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{e}_{\text{subword}_k}$$

This positions the new token near the relevant semantic cluster before any training begins.

---

## 4. Two-Stage Alignment Training

### Stage 1: Freeze Backbone (Embedding Alignment)

- Freeze all Transformer layer weights; only train the Embedding layer and LM Head.
- Forces new tokens to align into the existing semantic space without disturbing pretrained representations.

### Stage 2: Differential Learning Rate Fine-Tuning

- Unfreeze all parameters, but assign a much lower learning rate to the Embedding layer (e.g., $0.1 \times \eta_{\text{backbone}}$).
- Prevents "general knowledge collapse" caused by large Embedding fluctuations.

---

## 5. Data Replay

During Embedding expansion, mix in **10–15% general pre-training data** (e.g., Wikipedia, C4) as anchors.

- **Anchor Effect**: Prevents new tokens from overwriting the semantic boundaries of existing tokens, preserving the model's reasoning and language understanding baseline.
- See also: Data Replay in [../03_Pre_Training/11_Continual_Pre_Training.md](../03_Pre_Training/11_Continual_Pre_Training.md).

---

## 6. Glitch Tokens

Tokens like `SolidGoldMagikarp` exist in the vocabulary but disappeared during data cleaning, leaving their Embedding weights never updated from initialization.

- **Detection**: Before deployment, perform a norm check on the Embedding matrix — tokens with abnormally small or large norms are likely glitch tokens.
- **Mitigation**: Replace their embeddings with the mean embedding or remove them from the vocabulary.

---

## 7. Key References

1. **Press & Wolf (2017)**: *Using the Output Embedding to Improve Language Models* (Weight Tying).
2. **Hewitt & Manning (2019)**: *A Structural Probe for Finding Syntax in Word Representations*.
3. **Cui et al. (2023)**: *Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca* (Vocabulary expansion case study).
4. **Rumbelow & Watkins (2023)**: *SolidGoldMagikarp: Glitch Tokens in LLMs*.
