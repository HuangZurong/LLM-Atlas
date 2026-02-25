# Track 4: Solutions (Architect & Ship)

Architectural decision frameworks and implementation roadmaps for domain LLM applications.

Four-phase progression: **Strategy → Infrastructure → Build → Ship**.

---

## Module Map

### Strategy

| File | Topic |
|:--|:--|
| [01_Technology_Selection.md](./01_Technology_Selection.md) | Prompt Eng vs RAG vs Fine-tuning decision framework |
| [02_Cost_ROI_Analysis_Model.md](./02_Cost_ROI_Analysis_Model.md) | TCO modeling, API vs self-host break-even, ROI formula |

### Infrastructure

| File | Topic |
|:--|:--|
| [03_Domain_Data_Strategy.md](./03_Domain_Data_Strategy.md) | Data sourcing, cleaning pipeline, synthetic data generation |
| [04_Evaluation_Loop.md](./04_Evaluation_Loop.md) | Offline/online evaluation, A/B testing, continuous monitoring |

### Build

| File | Topic |
|:--|:--|
| [05_RAG_Architecture.md](./05_RAG_Architecture.md) | Multi-source RAG, Agentic RAG, GraphRAG patterns |
| [06_Finetuning_Playbook.md](./06_Finetuning_Playbook.md) | CPT → SFT → DPO execution guide |
| [07_Knowledge_Graph_Integration.md](./07_Knowledge_Graph_Integration.md) | Hybrid structured + unstructured retrieval |
| [08_Agent_Workflow_Design.md](./08_Agent_Workflow_Design.md) | 5 workflow patterns, state management, HITL gates |

### Ship

| File | Topic |
|:--|:--|
| [09_Vertical_Scenario_Templates.md](./09_Vertical_Scenario_Templates.md) | Legal, Finance, Manufacturing, Medical blueprints |
| [10_Implementation_Roadmap.md](./10_Implementation_Roadmap.md) | PoC → MVP → Production → Scale with Go/No-Go criteria |

### Interactive Tools

| File | Description |
|:--|:--|
| [technology_selector.py](./technology_selector.py) | CLI tool: 7-question survey → approach + model tier + cost recommendation |
| [rag_pipeline_demo.py](./rag_pipeline_demo.py) | End-to-end RAG: chunking → hybrid retrieval → RRF → generation → faithfulness check |
| [finetune_pipeline_demo.py](./finetune_pipeline_demo.py) | QLoRA SFT pipeline: data prep → 4-bit loading → LoRA → training → evaluation |

---

*Theory foundations are in [Track 2: Scientist](../02_Scientist/). Engineering implementation details are in [Track 3: Engineering](../03_Engineering/).*
