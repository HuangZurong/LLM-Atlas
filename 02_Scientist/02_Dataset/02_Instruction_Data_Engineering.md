# Instruction Data Engineering: Aligning with Precision

*Prerequisite: [01_Pre_Training_Data_at_Scale.md](01_Pre_Training_Data_at_Scale.md). For SFT training details, see [../04_Post_Training/01_FT/03_Decoder_NLG/01_LLM_SFT_Guide.md](../04_Post_Training/01_FT/03_Decoder_NLG/01_LLM_SFT_Guide.md).*

---

## 1. The Core Philosophy: LIMA Principle

"Less Is More for Alignment" (LIMA) posits that a model's knowledge is almost entirely learned during pre-training, while alignment is about learning the **surface form** or **style** of interaction.
- **Key finding**: 1,000 carefully curated, high-quality examples can outperform 50,000+ low-quality or synthesized samples.
- **Diversity > Volume**: Ensuring the instruction set covers distinct "skills" (reasoning, creative writing, code, math, safety).

---

## 2. Instruction Evolution (Evol-Instruct)

To increase the complexity and variety of instruction data without manual labeling, we use the **Evol-Instruct** approach (WizardLM):

### 2.1 Depth Evolution (Deepening)
Transform a simple instruction into a more complex one by adding constraints or multi-step logic.
- **Base**: "Write a Python function to sort a list."
- **Evolved**: "Write a Python function to sort a list of dictionaries by a specific key, handle missing keys gracefully, and ensure $O(N \log N)$ complexity."

### 2.2 Breadth Evolution (Widening)
Create new instructions that are related but in different domains or contexts to increase diversity.

---

## 3. Data Complexity Modeling: IFD

**IFD (Instruction-Following Difficulty)** is a metric from Li et al. (2023) used to select the most challenging and informative samples:
- **Calculation**: $IFD = \frac{\text{Loss}(Response \mid Instruction)}{\text{Loss}(Response \mid \varnothing)}$
- **Logic**: We **select samples with HIGH IFD**. A high IFD means the response is still difficult to generate even *given* the instruction — indicating a genuinely challenging, non-trivial task. Low IFD samples (where the instruction makes the response trivially predictable) are discarded as uninformative templates.
- **Citation**: Li et al. (2023), *From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning*.

---

## 4. Production Engineering: Data Packing

During Supervised Fine-Tuning (SFT), we must maximize GPU efficiency.

### 4.1 Sequence Packing (Constant Length Mapping)
Instead of padding each sample to the max sequence length (wasting 60-80% of tokens), we concatenate multiple samples into a single sequence, separated by an `<EOS>` or `<SEP>` token.
- **Efficiency**: Increases throughput by 2x-4x.
- **Masking**: Careful attention masking is required to ensure the model doesn't attend to the *next* instruction during the current response.

---

## 5. Formatting Standards

Modern industry standards for instruction formatting:
- **ChatML (OpenAI)**: Uses `<|im_start|>system...` tags.
- **Llama 3 Header**: Uses specific header tokens for role identification (`user`, `assistant`).

---

## 6. Key References

1.  **Zhou et al. (2023)**: *LIMA: Less Is More for Alignment*.
2.  **Xu et al. (2023)**: *WizardLM: Empowering Large Language Models to Follow Complex Instructions*.
3.  **HuggingFace (2024)**: *Alignment Handbook*.
