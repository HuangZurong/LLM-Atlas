# Multilingual Engineering Best Practices

*Prerequisite: [../01_Theory/01_Foundations_and_Anatomy.md](../01_Theory/01_Foundations_and_Anatomy.md).*

---

Scaling LLM applications to multiple languages involves more than just translation. It requires an understanding of **Token Economics**, **Cultural Nuance**, and **Prompt Reliability**.

## 1. The "Source Language" Dilemma

### Option A: English Prompts for All Languages
- **Pros**: Models (like GPT-4o, Claude) are best at following instructions in English. Higher reasoning consistency.
- **Cons**: Small bias toward English-centric logic.

### Option B: Native Language Prompts
- **Pros**: May capture cultural nuances better for specific creative tasks.
- **Cons**: Smaller models (7B/14B) often have significantly worse instruction-following in non-English languages.

**Industrial Standard**: Use **English for the System Prompt** (Instructions/Rules) and provide **Native examples** in the few-shot section.

## 2. Token Economics across Languages

Different tokenizers (Tiktoken, SentencePiece) have different efficiencies for non-Latin scripts.

| Language | Token Multiplier (vs English) | Cost/Context Impact |
| :--- | :--- | :--- |
| **English** | 1x | Baseline |
| **Chinese (Simplified)** | 1.5x - 2.5x | Moderate impact. GPT-4o is efficient; older models are not. |
| **Japanese/Korean** | 2x - 3x | Significant impact. |
| **Hindi/Arabic** | 3x - 5x | Severe impact. Context window fills up quickly. |

**Optimization**: For high-token languages, use **shorthand** or **summarization** of the context before processing.

## 3. Dealing with Translation Hallucinations

When an LLM translates idioms or technical terms, it may hallucinate.
- **Best Practice**: Use **Glossary-Enforced Translation**.
- Provide a list of "Must-translate-as" terms in the System Prompt:
  > "GLOSSARY: 'User Interface' -> '用户界面'; 'Cloud Computing' -> '云计算'."

## 4. Few-Shot Calibration for Global Apps

If your app is global, your few-shot examples should be:
1. **Diverse**: Include examples in the target languages.
2. **Culturally Valid**: Ensure address forms (polite vs informal) are correct for the target locale (e.g., German "Sie" vs "du").

## 5. Automated Multilingual Evaluation

Don't rely on "Vibe checks" by native speakers alone.
- Use **Back-Translation**: Translate LLM Output -> English (via a different model) and check for semantic drift.
- Use **Metric-based Eval**: BLEU, ROUGE, or METEOR for translation quality.
