# Statistical Models for NLP

*Prerequisite: [02_Feature_Engineering.md](02_Feature_Engineering.md).*

---

Before deep learning became mainstream, these classical models were the core tools of NLP. Their ideas continue to influence modern architectures (e.g., CRF layers are still appended to BERT for NER). Understanding them reveals the arc of NLP's technical evolution.

## Contents

- [1. Naive Bayes](#1-naive-bayes)
- [2. Hidden Markov Model (HMM)](#2-hidden-markov-model-hmm)
- [3. Conditional Random Field (CRF)](#3-conditional-random-field-crf)
- [4. Comparison & Legacy](#4-comparison--legacy)

## 1. Naive Bayes

The simplest probabilistic classifier — based on Bayes' theorem, assuming all features (words) are mutually independent.

### Core Formula

$$P(c | d) \propto P(c) \prod_{i=1}^n P(w_i | c)$$

- $P(c)$: Class prior (e.g., 30% of emails are spam)
- $P(w_i | c)$: Probability of word $w_i$ given class $c$
- **The "Naive" assumption**: Words are conditionally independent given the class (clearly unrealistic, but works surprisingly well in practice)

### Classification Process

```
Input: "Free iPhone giveaway prize"

P(spam | text)  ∝ P(spam) × P(Free|spam) × P(iPhone|spam) × P(giveaway|spam) × P(prize|spam)
P(legit | text) ∝ P(legit) × P(Free|legit) × P(iPhone|legit) × P(giveaway|legit) × P(prize|legit)

→ Compare probabilities, pick the larger one
```

### Applications

- **Spam filtering**: The classic Naive Bayes application
- **Sentiment analysis**: Positive / negative binary classification
- **Document classification**: News categorization, topic labeling

### Pros and Cons

- **Pros**: Extremely fast training, requires little data, highly interpretable
- **Cons**: Independence assumption is too strong; cannot capture word order or word interactions

## 2. Hidden Markov Model (HMM)

A **Hidden Markov Model** is a sequence labeling model — given an observed sequence (words), infer the hidden state sequence (labels).

### Core Idea

```
Hidden states (Tags):    DET    NOUN    VERB    DET    NOUN
                          ↓      ↓       ↓       ↓      ↓
Observations (Words):    The    cat     sat     the    mat
```

The model is defined by two sets of probabilities:

- **Transition probabilities**: Probability of state-to-state transitions
  - $P(\text{NOUN} | \text{DET}) = 0.8$ (a determiner is very likely followed by a noun)
- **Emission probabilities**: Probability of a state generating an observation
  - $P(\text{"cat"} | \text{NOUN}) = 0.02$

### Three Core Problems and Algorithms

| Problem | Description | Algorithm |
|:--------|:-----------|:----------|
| **Evaluation** | Given a model, compute the probability of an observation sequence | Forward Algorithm |
| **Decoding** | Given observations, find the most likely state sequence | **Viterbi Algorithm** |
| **Learning** | Learn model parameters from data | Baum-Welch (EM) |

### Viterbi Algorithm (Dynamic Programming)

Finds the globally optimal label path, rather than greedily choosing at each position:

$$v_t(j) = \max_{i} \left[ v_{t-1}(i) \cdot a_{ij} \cdot b_j(o_t) \right]$$

### Applications

- **POS Tagging**: The classic method for part-of-speech tagging
- **Speech Recognition**: Sound signal → phonemes → words (HMM dominated ASR for decades)
- **Chinese Word Segmentation**: jieba's underlying engine uses HMM

### Limitations

- **Markov assumption**: Current state depends only on the previous state — cannot model long-range dependencies
- **Generative model**: Models joint $P(\text{observations, states})$, not directly $P(\text{states} | \text{observations})$
- **Limited features**: Can only use the current observed word as a feature

## 3. Conditional Random Field (CRF)

**CRF** directly addresses HMM's limitations — it is a **discriminative** sequence labeling model that directly models $P(\mathbf{y} | \mathbf{x})$.

### Key Differences from HMM

| Aspect | HMM | CRF |
|:-------|:----|:----|
| Model type | Generative $P(\mathbf{x}, \mathbf{y})$ | **Discriminative** $P(\mathbf{y} \| \mathbf{x})$ |
| Features | Current word only | **Arbitrary feature functions** (surrounding words, affixes, capitalization, dictionary matches, etc.) |
| Independence assumption | Observations are independent | **No independence assumption required** |

### Core Formula

$$P(\mathbf{y} | \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\left(\sum_{t=1}^T \sum_k \lambda_k f_k(y_{t-1}, y_t, \mathbf{x}, t)\right)$$

- $f_k$: Feature functions (e.g., "current word is capitalized AND label is PERSON")
- $\lambda_k$: Feature weights (learned during training)
- $Z(\mathbf{x})$: Normalization constant

### Feature Function Examples

```python
# CRF can use rich hand-crafted features
features = {
    "current_word":        word,
    "lowercase":           word.lower(),
    "prefix_3":            word[:3],
    "suffix_3":            word[-3:],
    "is_capitalized":      word[0].isupper(),
    "is_all_caps":         word.isupper(),
    "previous_word":       prev_word,
    "next_word":           next_word,
    "pos_tag":             pos_tag,
}
```

### Applications

- **Named Entity Recognition (NER)**: CRF was the state-of-the-art algorithm for NER
- **Chinese Word Segmentation**: Character-level CRF segmenters
- **Modern usage: BERT + CRF**: BERT provides contextual representations, the CRF layer ensures global label consistency (e.g., preventing "B-PER" from being followed directly by "I-LOC")

## 4. Comparison & Legacy

### Model Comparison

| Model | Type | Task | Core Strength | Core Limitation |
|:------|:-----|:-----|:-------------|:----------------|
| **Naive Bayes** | Classification | Text classification | Simple and fast | Independence assumption |
| **HMM** | Sequence labeling | POS / Segmentation | Unsupervised training | Limited features |
| **CRF** | Sequence labeling | NER / Segmentation | Rich features | Requires feature engineering |

### Legacy in Modern NLP

Though no longer mainstream, their ideas profoundly shaped modern architectures:

- **Naive Bayes' Bayesian thinking** → Bayesian Deep Learning
- **HMM's sequence modeling** → The hidden state concept in RNNs
- **CRF's global sequence optimization** → BERT-CRF remains a strong NER baseline; Transformer self-attention can be viewed as a "soft" global dependency model
- **Viterbi decoding** → Precursor to Beam Search

---

_Traditional NLP established the "feature engineering + statistical model" paradigm. Next, we enter the deep learning era — replacing hand-crafted features with continuous vectors, fundamentally changing the rules of the game._

_Next: [Word Embeddings](../03_Deep_Learning/01_Word_Embeddings.md)_
