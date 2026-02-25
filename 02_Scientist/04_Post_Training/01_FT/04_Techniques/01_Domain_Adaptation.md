# Domain Adaptation: Vocab Expansion & Training Strategies

*Prerequisite: [../01_Theory/02_PEFT_Strategies.md](../01_Theory/02_PEFT_Strategies.md), [../../../01_Architecture/05_Embedding.md](../../../01_Architecture/05_Embedding.md).*

---

How do you take a general model (like Llama-3-English) and turn it into a high-performance specialist (like Chinese-LLaMA)?

## 1. The Challenge of Domain Shift

When a model enters a new domain (e.g., Medical, Legal, or new language), it faces:

- **Missing Concepts**: Special terms are often fragmented by the tokenizer.
- **Shallow Knowledge**: The model knows the words but lacks the logical domain expert "common sense".

## 2. Case Study: Chinese-LLaMA (arXiv:2304.08177)

The HFL team solved this using a **Two-Stage** approach to ensure stability and efficiency.

### Stage 1: Embedding Alignment (Cold Start Fix)

- **Problem**: New tokens (like Chinese characters) are initialized as "noise".
- **Action**: Freeze the Transformer logic layer. Only train the **Embedding layer** and **LM Head**.
- **Math Strategy**: Use **Mean Initialization** (average of subword embeddings) to give new tokens a semantic "head start".
- **Result**: The new tokens are "aligned" to the existing vector space without confusing the model's logic.

### Stage 2: Joint Logic & Adaptation

- **Action**: Unlock the Transformer layers via **LoRA**.
- **Synergy**: Continue updating the Embeddings while training the LoRA adapters.
- **Outcome**: The model learns Chinese grammar/syntax (via LoRA) while refining the specific meaning of Chinese words (via Embeddings).

## 3. Architect's Checklist for Expansion

1. **Tokenizer Fertility Check**: Does the current tokenizer split domain terms too much? If yes, expand the vocabulary.
2. **Cold Start Mitigation**: Never use random initialization. Use **Average Subword Vectors**.
3. **Selective Tuning**: Don't update the whole model if your data is small (<10GB). Use Stage 1 Alignment followed by PEFT.
4. **Learning Rate Management**: The Embedding Layer often needs a different (usually higher) learning rate than the Transformer layers during alignment.

---

## 4. Key References

1. **Cui et al. (2023)**: *Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca*.
2. **Gururangan et al. (2020)**: *Don't Stop Pretraining: Adapt Pretrained Language Models to Domains and Tasks*.
