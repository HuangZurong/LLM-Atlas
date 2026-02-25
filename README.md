# LLM Knowledge Base

A professional, four-track curriculum covering the full lifecycle of Large Language Models — from linguistic foundations to production deployment.

```
Track 1                Track 2                Track 3                  Track 4
Fundamentals           Scientist              Engineering              Solutions
(NLP → Transformer)    (Research & Training)  (Build & Operate)        (Architect & Ship)
       │                      │                      │                        │
       ▼                      ▼                      ▼                        ▼
  Prerequisite ──────→ Deep Theory ──────→ Industrial Practice ──────→ Business Delivery
```

---

## Track 1: Fundamentals

*From Classical NLP to the Attention Revolution.*

| Module | Core Topics | Entry Point |
| :--- | :--- | :--- |
| 01 Linguistics | NLP hierarchy, morphology, syntax, semantics | [Linguistic Foundations](01_Fundamentals/01_Linguistics/01_Linguistic_Foundations.md) |
| 02 Classical NLP | Preprocessing, BoW/TF-IDF, HMM/CRF | [Text Preprocessing](01_Fundamentals/02_Classical_NLP/01_Text_Preprocessing.md) |
| 03 Deep Learning | Word2Vec/GloVe, RNN/LSTM/GRU, Seq2Seq | [Word Embeddings](01_Fundamentals/03_Deep_Learning/01_Word_Embeddings.md), [Word2Vec Demo](01_Fundamentals/03_Deep_Learning/01_Word2Vec_Demo.py) |
| 04 Transformer Era | Attention mechanism, Transformer architecture, Pre-train paradigms | [Attention](01_Fundamentals/04_Transformer_Era/01_Attention.md), [Attention Viz](01_Fundamentals/04_Transformer_Era/01_Attention_Visualization.py) |
| 05 Applications | Classification, NER, MT, Summarization, Dialogue, Search | [LLM Disruption Map](01_Fundamentals/05_Applications/09_LLM_Disruption_Map.md) |

---

## Track 2: Scientist

*State-of-the-art model architecture, training, alignment, and frontier research.*

| Module | Core Topics | Entry Point |
| :--- | :--- | :--- |
| 01 Architecture (12) | Transformer, MHA/MQA/GQA/MLA, Efficient Attention, Tokenizer, Embedding, RoPE, Dense vs MoE, Decoding, Interpretability, Long Context | [Transformer](02_Scientist/01_Architecture/01_Transformer.md) |
| 02 Dataset (5) | Pre-training data at scale, Instruction data, Preference data, Synthetic data, PII management | [Data at Scale](02_Scientist/02_Dataset/01_Pre_Training_Data_at_Scale.md) |
| 03 Pre-Training (11) | GPT evolution, Scaling Laws, Attention optimizations, Data pipelines, Distributed training, Stability, Continual pre-training | [Scaling Laws](02_Scientist/03_Pre_Training/02_Scaling_Laws.md) |
| 04 Post-Training | **FT**: PEFT/LoRA/QLoRA, Domain adaptation | [PEFT Strategies](02_Scientist/04_Post_Training/01_FT/01_Theory/02_PEFT_Strategies.md) |
| | **Alignment**: PPO, DPO, KTO, RLAIF, Constitutional AI, RLVR, GRPO | [Alignment Overview](02_Scientist/04_Post_Training/02_Alignment/01_Overview.md) |
| | **Advanced**: Rejection Sampling, Iterative Training, Inference-Time Compute, Model Merging | [Inference-Time Compute](02_Scientist/04_Post_Training/02_Alignment/04_Advanced_Topics/03_Inference_Time_Compute.md) |
| | **Distillation** | [Distillation Overview](02_Scientist/04_Post_Training/03_Distillation/01_Overview.md) |
| 05 Evaluation (5) | Benchmarks taxonomy, Methodology, LLM-as-Judge, Safety eval, Contamination detection | [Benchmarks](02_Scientist/05_Evaluation/01_Benchmarks_Taxonomy.md) |
| 06 Multimodal (4) | Vision-Language, Audio/Speech, Video understanding, Multimodal eval | [VLM](02_Scientist/06_Multimodal/01_Vision_Language_Models.md) |
| 07 Paper Tracking (5) | Tracking methodology, Architecture/Training/Alignment/Multimodal frontiers | [Methodology](02_Scientist/07_Paper_Tracking/01_Tracking_Methodology.md) |

---

## Track 3: Engineering

*Building, deploying, and operating production-grade LLM applications.*

Every module follows a strict 3-layer structure: **Theory → Practical (.py) → Best Practice (.md)**.

| Module | Theory | Practical | Best Practice |
| :--- | :--- | :--- | :--- |
| 01 LLMs | Intelligence landscape, Tokenization & cost, API mechanics, Context optimization | [Async Gateway](03_Engineering/01_LLMs/02_Practical/01_Production_Async_Gateway.py), [Batch API](03_Engineering/01_LLMs/02_Practical/05_Batch_API_Pipeline.py), [Guardrails](03_Engineering/01_LLMs/02_Practical/08_Safety_Guardrails_Middleware.py) | [Architecture Matrix](03_Engineering/01_LLMs/03_Best_Practice/01_Architecture_Decision_Matrix.md), [Model Routing](03_Engineering/01_LLMs/03_Best_Practice/03_Model_Routing_Patterns.md) |
| 02 Prompting | Foundations, Programmatic prompting, Reasoning strategies, **Structured Output & Function Calling**, Prompt Template Architecture, **Data-Driven Prompt Design** | [DSPy](03_Engineering/02_Prompting/02_Practical/01_DSPy_Optimization_Basics.ipynb), [Self-Correction](03_Engineering/02_Prompting/02_Practical/02_Self_Correction_Chain.py), [Structured Output](03_Engineering/02_Prompting/02_Practical/05_Structured_Output_Patterns.py) | [Prompt CI/CD](03_Engineering/02_Prompting/03_Best_Practice/01_Automated_Prompt_CI_CD.md), [Defensive Design](03_Engineering/02_Prompting/03_Best_Practice/03_Defensive_Prompt_Design.md) |
| 03 Memory | Memory systems, Context window engineering | [Sliding Window](03_Engineering/03_Memory/02_Practical/01_Sliding_Window_and_Summary.py), [Vector Memory](03_Engineering/03_Memory/02_Practical/02_Vector_Memory_Store.py) | [Architecture Patterns](03_Engineering/03_Memory/03_Best_Practice/01_Memory_Architecture_Patterns.md), [Token Budget](03_Engineering/03_Memory/03_Best_Practice/03_Token_Budget_Management.md) |
| 04 RAG | Architecture, Advanced RAG, Data ingestion, GraphRAG | [Query Routing](03_Engineering/04_RAG/02_Practical/04_Query_Routing.py), [Hybrid Indexing](03_Engineering/04_RAG/02_Practical/05_Hybrid_Indexing.py), [Reranking](03_Engineering/04_RAG/02_Practical/06_Reranking_Pipeline.py) | [RAG Eval Framework](03_Engineering/04_RAG/03_Best_Practice/01_RAG_Evaluation_Framework.md), [Embedding Selection](03_Engineering/04_RAG/03_Best_Practice/02_Embedding_Selection_Matrix.md) |
| 05 Agent | Theory, Architecture, Workflow patterns, Multi-agent, MCP Protocol | [ReAct Agent](03_Engineering/05_Agent/02_Practical/01_ReAct_Tool_Agent.py), [Multi-Agent](03_Engineering/05_Agent/02_Practical/03_Multi_Agent_Coordination.py), [MCP Server](03_Engineering/05_Agent/02_Practical/04_MCP_Tool_Server.py) | [Agent Eval](03_Engineering/05_Agent/03_Best_Practice/01_Agent_Evaluation_and_Benchmarking.md), [Production Guardrails](03_Engineering/05_Agent/03_Best_Practice/02_Production_Guardrails_for_Agents.md) |
| | **Frameworks** (9): ADK, CrewAI, CamelAI, Agno, LangGraph, AutoGPT, BabyAGI, Semantic Kernel, OpenAI Swarm | [ADK Agent](03_Engineering/05_Agent/04_Frameworks/01_ADK/01_Single_Agent/agent.py), [Agno Agent](03_Engineering/05_Agent/04_Frameworks/04_Agno/01_Basic_Agent/agent.py) | |
| 06 Deployment | Optimization, Architecture, Quantization, Cloud comparison | [vLLM](03_Engineering/06_Deployment/02_Practical/01_vLLM_Deployment.py), [Continuous Batching](03_Engineering/06_Deployment/02_Practical/05_Continuous_Batching.py) | [Production Checklist](03_Engineering/06_Deployment/03_Best_Practice/01_Production_Readiness_Checklist.md), [SLOs & Monitoring](03_Engineering/06_Deployment/03_Best_Practice/03_SLOs_and_Monitoring.md) |
| 07 Security | LLM threats, Advanced threat modeling, Privacy/Compliance, Secure architecture | [Injection Detection](03_Engineering/07_Security/02_Practical/01_Prompt_Injection_Detection.py), [PII Redaction](03_Engineering/07_Security/02_Practical/02_PII_Redaction_and_Masking.py), [Agent Sandbox](03_Engineering/07_Security/02_Practical/03_Secure_Agent_Sandbox.py) | [Compliance Checklist](03_Engineering/07_Security/03_Best_Practice/01_Security_Compliance_Checklist.md), [Incident Response](03_Engineering/07_Security/03_Best_Practice/02_Incident_Response_Playbook.md) |
| 08 LLMOps | Maintenance, Observability, CI/CD for LLMs | [Eval Runner](03_Engineering/08_LLMOps/02_Practical/01_Automated_Evaluation_Runner.py), [Observability Collector](03_Engineering/08_LLMOps/02_Practical/02_Observability_Collector.py) | [Production Checklist](03_Engineering/08_LLMOps/03_Best_Practice/01_LLMOps_Production_Checklist.md), [On-Call Runbook](03_Engineering/08_LLMOps/03_Best_Practice/02_On_Call_Runbook.md) |

---

## Track 4: Solutions

*Architectural decision frameworks and implementation roadmaps for domain LLM applications.*

Four-phase progression: **Strategy → Infrastructure → Build → Ship**.

| Phase | Document | Key Question |
| :--- | :--- | :--- |
| **Strategy** | [01 Technology Selection](04_Solutions/01_Technology_Selection.md) | Prompt Eng vs RAG vs Fine-tuning? |
| | [02 Cost & ROI Analysis](04_Solutions/02_Cost_ROI_Analysis_Model.md) | Is it worth building? |
| **Infrastructure** | [03 Domain Data Strategy](04_Solutions/03_Domain_Data_Strategy.md) | Where does the data come from? |
| | [04 Evaluation Loop](04_Solutions/04_Evaluation_Loop.md) | How do we measure success? |
| **Build** | [05 RAG Architecture](04_Solutions/05_RAG_Architecture.md) | Multi-source, Agentic RAG patterns. |
| | [06 Finetuning Playbook](04_Solutions/06_Finetuning_Playbook.md) | CPT → SFT → DPO execution guide. |
| | [07 Knowledge Graph Integration](04_Solutions/07_Knowledge_Graph_Integration.md) | Hybrid structured + unstructured. |
| | [08 Agent Workflow Design](04_Solutions/08_Agent_Workflow_Design.md) | Business process orchestration. |
| **Ship** | [09 Vertical Scenario Templates](04_Solutions/09_Vertical_Scenario_Templates.md) | Legal, Finance, Manufacturing, Medical blueprints. |
| | [10 Implementation Roadmap](04_Solutions/10_Implementation_Roadmap.md) | PoC → MVP → Production → Scale. |

---

## Reading Paths

| Goal | Recommended Path |
| :--- | :--- |
| **"I'm new to NLP/LLM"** | Track 1 (all) → Track 3 (01-02) → Track 4 (01) |
| **"I want to build LLM apps"** | Track 3 (01→08) → Track 4 (01→10) |
| **"I want to train/align models"** | Track 1 (04) → Track 2 (01→05) |
| **"I need to deploy to production"** | Track 3 (06→08) → Track 4 (02, 10) |
| **"I'm evaluating LLM for my business"** | Track 4 (01→02) → Track 3 (01) → Track 4 (10) |
