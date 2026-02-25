# Introduction to Fine-Tuning

*Prerequisite: None (Entry point for Post-Training). For PEFT methods, see [02_PEFT_Strategies.md](02_PEFT_Strategies.md).*
*See Also: [../../../../04_Solutions/06_Finetuning_Playbook.md](../../../../04_Solutions/06_Finetuning_Playbook.md) (project-level execution guide), [../../../../03_Engineering/01_LLMs/03_Best_Practice/01_Architecture_Decision_Matrix.md](../../../../03_Engineering/01_LLMs/03_Best_Practice/01_Architecture_Decision_Matrix.md) (when to fine-tune vs RAG).*

---

Fine-tuning is the process of specializing a general pre-trained model for specific tasks or domains.

### Why Fine-Tune? (Benefits)

1.  **Performance**: Significantly reduces hallucinations, increases output consistency, and filters out unused information.
2.  **Privacy**: Enables on-prem or VPC deployment, preventing data leaks and ensuring no breaches from external API calls.
3.  **Cost Efficiency**: Lower cost per request in the long run, increased throughput, and greater parameter control.
4.  **Reliability**: Full control over uptime, lower latency for specific workloads, and deep customization.

### Prompt Engineering vs. Fine-tuning

| Feature          | Prompt Engineering        | Fine-tuning                            |
| :--------------- | :------------------------ | :------------------------------------- |
| **Effort**       | Easy/Fast to start        | Requires data & technical knowledge    |
| **Upfront Cost** | Low                       | Higher (Compute & Data prep)           |
| **Knowledge**    | No training needed        | Requires understanding of optimization |
| **Data Limit**   | Limited by Context Window | Nearly unlimited data filtering        |
| **Update Cycle** | Real-time (RAG)           | Static until next training run         |

### Instruction Fine-tuning (IFT)

IFT is a critical subset of Fine-tuning that focuses on making the model follow specific user intents (Instructions).

- **Relationship**: `Instruction Fine-Tuning` $\subset$ `Fine-Tuning`.
- **Sourcing**: FAQ, Customer Support conversations, Slack/Internal communications transformed into Alpaca-like pairs.

## 2. Advanced Challenges in Fine-tuning

### Catastrophic Forgetting (灾难性遗忘)

When a model learns new knowledge, it often "overwrites" or degrades its previous capabilities.

- **Research Trends**:
  - Empirical studies of LLM forgetting ([arXiv:2308.08747](https://arxiv.org/abs/2308.08747)).
  - Scaling Laws of forgetting ([arXiv:2401.05605](https://arxiv.org/html/2401.05605v1)).
  - Functional preservation methods like **FIP** ([arXiv:2205.00334](https://arxiv.org/pdf/2205.00334)) and **CURLoRA** ([arXiv:2408.14572](https://arxiv.org/abs/2408.14572)).
- **Mitigation**: Using "Recall and Learn" strategies or multi-task training frames.

### Model Hallucination & Knowledge Pruning

- **Hallucination**: Fine-tuning can inadvertently increase hallucinations if the training data contains false premises.
- **Knowledge Pruning**: Using protective experience training ([arXiv:2310.10158](https://arxiv.org/abs/2310.10158v2)) to selectively trim or shield specific knowledge during the tuning process.

---

## Key References

1. **Howard & Ruder (2018)**: *Universal Language Model Fine-tuning for Text Classification* (ULMFiT).
2. **Hu et al. (2021)**: *LoRA: Low-Rank Adaptation of Large Language Models*.
3. **Ouyang et al. (2022)**: *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT).
