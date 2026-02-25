# Tokenizers and Token Economics

*Prerequisite: [01_Intelligence_Landscape.md](01_Intelligence_Landscape.md).*

For an LLM engineer, **Tokens** are the unit of currency. Understanding how text becomes tokens and how they impact cost and performance is essential.

---

## 1. How Tokenizers Work

LLMs do not process text character by character or word by word. They use **Subword Tokenization**.

### 1.1 Byte-Pair Encoding (BPE)
The most common algorithm (used by GPT, Llama, Mistral).
1. Start with individual characters as tokens.
2. Iteratively merge the most frequently occurring adjacent pair of tokens into a new token.
3. Stop when a target vocabulary size (e.g., 32k or 128k) is reached.

**Result**: Common words (e.g., " the") become single tokens. Rare words or misspellings are broken into multiple pieces (e.g., " antidisestablishmentarianism" → " anti", " dis", " establishment", ...).

### 1.2 The Language Bias
Tokenizers are trained on specific datasets (usually heavily English-weighted).
- **English**: High compression. 1 token $\approx$ 0.75 words.
- **Chinese/Japanese**: Low compression. 1 character can be 2-3 tokens in older tokenizers. Modern models (Qwen, Llama 3) have much better multilingual tokenizers.

## 2. Why Tokens Matter

### 2.1 The "Fixed Vocabulary" Problem
A model's vocabulary size (e.g., 128,256 tokens for Llama 3) is a fixed dimension in the embedding layer. Larger vocabularies:
- Pros: Better compression (fewer tokens per sentence), faster inference (fewer steps).
- Cons: Larger model size, more VRAM required for the final softmax layer.

### 2.2 Cost Management & Billing Nuances
API providers charge per 1M tokens. Input tokens are usually cheaper than Output tokens.

**The `max_tokens` Trap**:
- In most APIs (OpenAI, Anthropic), the `max_tokens` parameter applies **only to the completion (output)**, not the total request.
- If `Prompt_Tokens + max_tokens > Context_Window`, the request will fail with a 400 error.
- **Engineering Tip**: Always calculate `available_tokens = Model_Limit - Prompt_Tokens` before setting `max_tokens` dynamically.

### 2.3 Tokenization Artifacts
- **Billing discrepancies**: Some providers bill based on their own tokenizer, which might differ slightly from the public one (e.g., handling of special control tokens).
- **The "Over-billing" risk**: Long strings of non-Latin characters or repeated symbols can explode token counts unexpectedly.

## 3. Engineering Challenges with Tokenizers

| Problem | Explanation | Mitigation |
|---|---|---|
| **Whitespace Sensitivity** | Some tokenizers handle `word` differently from ` word`. | Always be consistent with spacing in prompt templates. |
| **Number Handling** | Many tokenizers break numbers into inconsistent chunks (e.g., "100" vs "10", "0"). | For math tasks, sometimes adding spaces between digits helps. |
| **Special Tokens** | Tokens like `<|endoftext|>` control the model's flow. | Never let user input contain these tokens (injection risk). |

## 4. Measuring Token Usage

Always use the specific tokenizer for the model you are calling.

```python
import tiktoken

# For GPT-4o
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, how are you?")
print(f"Token count: {len(tokens)}")
```

For open models, use the `AutoTokenizer` from the `transformers` library.


---

## Key References

1. **Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units.** *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, 1*, 1715–1725.
2. **Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.** *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, 66–71.
3. **Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.** *OpenAI Blog*.
4. **OpenAI (2023). tiktoken: OpenAI's Fast BPE Tokenizer.** *GitHub Repository*.
