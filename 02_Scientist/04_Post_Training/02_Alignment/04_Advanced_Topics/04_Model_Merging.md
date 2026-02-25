# Model Merging (Weight-Space Alignment)

*Prerequisite: [../02_Preference_Alignment/03_DPO.md](../02_Preference_Alignment/03_DPO.md).*

Model merging is a set of techniques to combine the weights of two or more pre-trained or fine-tuned models into a single model. This is done **without any additional training (zero-shot)**, making it a highly efficient way to combine capabilities from different models.

---

## 1. Why Merge Models?

1. **Combine Capabilities**: Merge a "Coding Expert" model with a "Poetry Expert" model to create a "Multimodal Expert."
2. **Mitigate Catastrophic Forgetting**: If a model was fine-tuned on a new task but lost its general knowledge, merging it back with its base model can recover the general capabilities.
3. **De-biasing/Safety**: Merge a helpful (but risky) model with a safe (but boring) model to find a better balance.
4. **Efficiency**: Zero-cost improvement compared to expensive fine-tuning.

## 2. Key Techniques

| Technique | Description | Pros/Cons |
|:--|:--|:--|
| **Linear Averaging** | Simple $W_{new} = (1-\alpha)W_A + \alpha W_B$. | Simple; often performs poorly if weights aren't aligned. |
| **SLERP (Spherical Linear Interpolation)** | Interpolates weights along a curve on a sphere. | Better for maintaining the geometric properties of high-dimensional weights. |
| **TIES (Trim, Elect, and Sign)** | Trims small changes, elects the dominant sign for each parameter, and averages. | Reduces interference between conflicting updates; very popular for merging multi-task models. |
| **DARE (Drop And REscale)** | Randomly drops a large percentage of weight updates and rescales the remaining ones. | Surprisingly effective at removing "noise" from fine-tuning. |
| **Model Soups** | Averaging weights of multiple models fine-tuned with different hyperparameters. | Highly robust and often improves generalization. |

## 3. The "MergeKit" Ecosystem

**MergeKit** is the standard open-source library used by the community (especially on HuggingFace) to create "Franken-models." It supports various merging recipes and has led to a Cambrian explosion of highly-rated models like **Llama-3-Instruct-v1-Merge**.

## 4. Model Merging vs. Distillation

| Aspect | Model Merging | Distillation |
|:--|:--|:--|
| **Compute Cost** | Near Zero (CPU-only) | High (GPU training) |
| **Architecture** | Models must share the same architecture | Any architecture (Teacher can be different from Student) |
| **Complexity** | Simple mixing of weights | Complex training pipeline |
| **Goal** | Combine distinct weights | Transfer knowledge to a smaller model |

## 5. Challenges and Limitations

1. **Architecture Constraint**: You can generally only merge models with the same architecture and parameter count (e.g., you can merge two Llama-3-8B variants, but not a Llama-3-8B and a Mistral-7B).
2. **Interference**: Sometimes, the weights of two models conflict (e.g., one model learns to use a certain neuron for math, while another uses it for code). Merging them might break both capabilities.
3. **Evaluation**: Merged models can sometimes exhibit "glitches" or unstable behavior in specific edge cases that aren't caught by standard benchmarks.

## 6. Practical Workflow

1. **Identify Parents**: Find two models with desirable, complementary traits.
2. **Choose a Method**: Start with SLERP or TIES-Merging.
3. **Run MergeKit**: Use a YAML configuration to define the recipe.
4. **Evaluate**: Test the merged model on both parent tasks to ensure no regression.
5. **Iteration**: Adjust the merging weights (e.g., 70% model A, 30% model B) to fine-tune the balance.

---

## 7. Key References

1. **Wortsman et al. (2022)**: *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy without Increasing Inference Time*.
2. **Yadav et al. (2023)**: *TIES-Merging: Resolving Interference When Merging Models*.
3. **Goddard et al. (2024)**: *Arcee's MergeKit: A Toolkit for Merging Large Language Models*.
