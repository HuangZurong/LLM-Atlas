# Technology Selection: From Problem to Architecture

*Prerequisite: None (Solutions Track entry point). Cross-references [03_Engineering/](../03_Engineering/) modules.*

---

When facing a domain-specific LLM application requirement, the first question is never "how to fine-tune" — it's "what combination of techniques best fits this problem?" This document provides a systematic decision framework.

## 1. The Three Pillars: Prompt Engineering, RAG, and Fine-tuning

These are the three fundamental approaches to adapting an LLM to a specific domain. They are not mutually exclusive — most production systems combine two or all three. But understanding when each shines (and fails) is the foundation of all design decisions.

### 1.1 Prompt Engineering

**What it is**: Crafting instructions, examples, and context within the prompt to steer model behavior, without modifying the model itself.

**When to use**:
- The task can be solved with the model's existing knowledge
- You need rapid iteration and experimentation
- Data is scarce or unavailable for training
- The domain is broad rather than deeply specialized

**When it fails**:
- The model lacks the required domain knowledge entirely (e.g., proprietary internal processes)
- Consistent, structured output formats are critical and the model keeps deviating
- The task requires reasoning over large volumes of domain data that won't fit in context

**Cost**: Lowest. No training, no infrastructure beyond API calls.

### 1.2 Retrieval-Augmented Generation (RAG)

**What it is**: At inference time, retrieve relevant documents from an external knowledge base and inject them into the prompt as context, then let the model generate answers grounded in that context.

**When to use**:
- Domain knowledge is stored in documents, databases, or structured records
- Knowledge updates frequently (new regulations, project data, market changes)
- Factual accuracy and traceability (citing sources) are critical
- You need the model to answer questions about specific, private data

**When it fails**:
- The task requires deep reasoning or synthesis across many documents simultaneously
- The knowledge is implicit (e.g., "how experienced engineers think about risk") rather than explicit in documents
- Retrieval quality is poor — garbage in, garbage out
- The domain vocabulary is so specialized that embedding models can't capture semantic similarity

**Cost**: Medium. Requires building and maintaining a vector database, embedding pipeline, and retrieval infrastructure.

### 1.3 Fine-tuning

**What it is**: Further training a pre-trained model on domain-specific data to internalize new knowledge, behavior patterns, or output styles.

**When to use**:
- You need the model to adopt a specific "voice," style, or reasoning pattern
- Domain-specific terminology and concepts are poorly handled by the base model
- You have sufficient high-quality labeled data (thousands to tens of thousands of examples)
- Latency matters — fine-tuned models don't need retrieval overhead
- The task is narrow and well-defined (e.g., classifying construction defects, generating standardized reports)

**When it fails**:
- Knowledge changes frequently — you'd need to retrain constantly
- Training data is insufficient or low quality
- You need the model to cite specific sources (fine-tuned knowledge is "baked in" without attribution)
- Compute budget is severely limited

**Cost**: Highest. Requires GPU resources, training data preparation, evaluation pipelines, and ongoing maintenance.

### 1.4 Decision Matrix

| Factor | Prompt Engineering | RAG | Fine-tuning |
| :--- | :--- | :--- | :--- |
| **Domain data required** | None | Documents/records | Labeled training pairs |
| **Knowledge freshness** | Always current (model's cutoff) | Real-time updatable | Frozen at training time |
| **Setup cost** | Hours | Days to weeks | Weeks to months |
| **Inference latency** | Low | Medium (retrieval overhead) | Low |
| **Factual grounding** | Weak (may hallucinate) | Strong (cites sources) | Medium (no attribution) |
| **Behavioral control** | Limited | Limited | Strong |
| **Scalability to new domains** | Easy (change prompt) | Medium (rebuild index) | Hard (retrain) |

### 1.5 The Combination Principle

In practice, the most effective systems layer these approaches:

1. **Prompt Engineering + RAG** (most common): Use RAG to inject relevant context, then use carefully designed prompts to control how the model reasons over that context. This is the default starting point for most domain applications.

2. **Fine-tuning + RAG**: Fine-tune the model to understand domain terminology and reasoning patterns, then use RAG for factual grounding. This is the gold standard for high-stakes domain applications (medical, legal, engineering).

3. **Fine-tuning + Prompt Engineering**: Fine-tune for domain adaptation, then use prompts for task-specific steering. Useful when all knowledge is already internalized and retrieval isn't needed.

**Rule of thumb**: Start with Prompt Engineering. Add RAG when the model lacks knowledge. Add Fine-tuning when the model lacks behavior.

## 2. Base Model Selection

### 2.1 Architecture Choice: Encoder vs Decoder vs Encoder-Decoder

| Architecture | Representative Models | Strengths | Best For |
| :--- | :--- | :--- | :--- |
| **Encoder-only** | BERT, RoBERTa, DeBERTa | Bidirectional understanding | Classification, NER, semantic similarity, information extraction |
| **Decoder-only** | GPT-4, Llama, Qwen, Mistral | Autoregressive generation | Text generation, dialogue, code, general-purpose assistants |
| **Encoder-Decoder** | T5, BART, mT5 | Sequence-to-sequence mapping | Translation, summarization, structured output generation |

**Decision guide**:
- If your task is primarily **understanding** (classify this defect, extract entities from this report, match similar cases) → Encoder-only
- If your task is primarily **generation** (answer questions, write reports, have conversations) → Decoder-only
- If your task is **transformation** (summarize this document, translate this specification, convert unstructured to structured) → Encoder-Decoder or Decoder-only

For most modern domain LLM applications, **Decoder-only** models dominate because they handle both understanding and generation reasonably well, and the ecosystem (tooling, fine-tuning frameworks, deployment infrastructure) is most mature.

### 2.2 Open-source vs Closed-source

| Factor | Open-source (Llama, Qwen, Mistral) | Closed-source (GPT-4, Claude) |
| :--- | :--- | :--- |
| **Data privacy** | Full control — data stays on your infrastructure | Data sent to third-party API |
| **Customization** | Full fine-tuning, architecture modification possible | Limited to API-level prompt engineering and (sometimes) fine-tuning |
| **Cost structure** | High upfront (GPU), low marginal | Low upfront, usage-based (can get expensive at scale) |
| **Performance ceiling** | Slightly lower for general tasks | Currently highest for general reasoning |
| **Deployment flexibility** | On-premise, edge, air-gapped environments | Cloud-only, internet required |
| **Regulatory compliance** | Easier to meet data residency requirements | May conflict with data sovereignty regulations |

**Decision guide**:
- **Sensitive/regulated domains** (government, military, healthcare with PII) → Open-source, on-premise
- **Rapid prototyping, general tasks** → Closed-source API
- **Production at scale with domain specificity** → Open-source with fine-tuning
- **Budget-constrained research** → Open-source smaller models (7B-14B)

### 2.3 Parameter Scale vs Resource Constraints

| Model Scale | Typical Parameters | GPU Requirement (Inference) | GPU Requirement (Fine-tuning) | Sweet Spot |
| :--- | :--- | :--- | :--- | :--- |
| **Small** | 1B-3B | Single consumer GPU (8GB) | Single GPU (16GB) | Edge deployment, specific narrow tasks |
| **Medium** | 7B-14B | Single GPU (24GB) | 1-2 GPUs (48GB+) | Best cost-performance ratio for domain tasks |
| **Large** | 32B-72B | Multi-GPU or quantized | Multi-GPU cluster | Complex reasoning, multilingual |
| **Frontier** | 100B+ | GPU cluster | Impractical for most | Use via API only |

**The 7B-14B sweet spot**: For most domain-specific applications, 7B-14B parameter models offer the best trade-off. They are:
- Small enough to fine-tune on a single machine with 2-4 GPUs
- Large enough to capture complex domain knowledge after fine-tuning
- Fast enough for real-time inference
- Well-supported by quantization techniques (GPTQ, AWQ, GGUF) for further compression

### 2.4 Fine-tuning Method Selection

| Method | Parameters Modified | GPU Memory | Data Required | When to Use |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-tuning** | All | Very high (full model × 2-3) | 10K+ examples | Maximum performance, sufficient resources |
| **LoRA / QLoRA** | Low-rank adapters only (0.1-1%) | Low (quantized base + small adapters) | 1K-10K examples | Most practical choice for domain adaptation |
| **Prefix Tuning** | Prepended virtual tokens | Low | 500-5K examples | Lightweight task adaptation |
| **Prompt Tuning** | Soft prompt embeddings only | Minimal | 100-1K examples | When compute is extremely limited |

**Default recommendation**: **QLoRA** on a 7B-14B model. It achieves 90-95% of full fine-tuning performance at a fraction of the cost, and is the most battle-tested approach in production.

## 3. Knowledge Injection Strategies

When domain knowledge needs to be made available to the model, there are three fundamentally different approaches. The choice depends on the nature of the knowledge itself.

### 3.1 Vector Retrieval (RAG)

**Best for**: Explicit, document-based knowledge that can be chunked and retrieved.

- Construction specifications, regulatory documents, project reports
- FAQ databases, historical case records
- Any knowledge that exists as text and needs to be cited

**Architecture**: Documents → Chunking → Embedding → Vector DB → Retrieval → LLM

**Key design decisions**:
- Chunk size (too small loses context, too large dilutes relevance)
- Embedding model choice (domain-specific vs general-purpose)
- Retrieval strategy (dense, sparse, hybrid)
- Re-ranking (cross-encoder for precision)

### 3.2 Knowledge Graph

**Best for**: Structured, relational knowledge with clear entities and relationships.

- Organizational hierarchies, equipment taxonomies
- Causal chains (defect → cause → remedy)
- Regulatory cross-references, standard dependencies
- Temporal sequences (construction phases, milestone dependencies)

**Architecture**: Domain data → Entity extraction → Relation extraction → Graph DB (Neo4j) → Graph query → LLM

**When KG adds value over RAG**:
- Questions require multi-hop reasoning ("What equipment is affected if Supplier X delays delivery?")
- The knowledge has inherent graph structure that flat text retrieval would miss
- Consistency and completeness of knowledge matter (KG can be validated, documents can't)

**When KG is overkill**:
- Knowledge is primarily unstructured narrative
- The domain doesn't have clear entity-relationship patterns
- You lack domain experts to validate the graph

### 3.3 Domain Fine-tuning

**Best for**: Implicit knowledge, behavioral patterns, and domain "intuition."

- How experienced engineers reason about risk
- Domain-specific writing style and terminology
- Task-specific output formats and structures
- Judgment calls that can't be reduced to retrievable facts

### 3.4 Knowledge Strategy Decision Tree

```
Is the knowledge explicit and document-based?
├── Yes → Is it frequently updated?
│   ├── Yes → RAG (vector retrieval)
│   └── No → RAG or Fine-tuning (either works)
└── No → Is it structured with clear entities/relations?
    ├── Yes → Knowledge Graph + LLM
    └── No → Is it behavioral/stylistic?
        ├── Yes → Fine-tuning
        └── No → Likely needs domain expert consultation
                  to formalize the knowledge first
```

## 4. Typical Architecture Patterns

### 4.1 Pattern A: RAG-Centric (Most Common)

```
User Query → Query Rewriting → Retrieval → Re-ranking → Prompt Assembly → LLM → Response
                                   ↑
                            Vector Database
                          (domain documents)
```

**Use case**: Domain Q&A, document-grounded assistants, compliance checking.
**Pros**: Fast to build, knowledge is updatable, answers are traceable.
**Cons**: Limited by retrieval quality, struggles with complex reasoning.

### 4.2 Pattern B: Fine-tuned Domain Expert

```
User Query → Prompt Template → Fine-tuned LLM → Response
                                      ↑
                              Domain training data
                            (pre-baked into weights)
```

**Use case**: Specialized report generation, domain-specific classification, style-consistent output.
**Pros**: Fast inference, consistent behavior, no retrieval infrastructure.
**Cons**: Knowledge is frozen, no source attribution, requires retraining for updates.

### 4.3 Pattern C: RAG + Fine-tuned Model (Gold Standard)

```
User Query → Query Rewriting → Retrieval → Re-ranking → Prompt Assembly → Fine-tuned LLM → Response
                                   ↑                                              ↑
                            Vector Database                               Domain adaptation
                          (factual knowledge)                          (reasoning patterns)
```

**Use case**: High-stakes domain applications requiring both accuracy and domain fluency.
**Pros**: Best of both worlds — factual grounding + domain expertise.
**Cons**: Most complex to build and maintain.

### 4.4 Pattern D: Agent + Tools (Autonomous)

```
User Query → Agent LLM → [Plan] → Tool Call (DB query / API / Calculator / Search) → [Observe] → ... → Response
                  ↑                        ↑
           Agent framework            External tools
          (ReAct, Function Calling)   (domain-specific)
```

**Use case**: Complex tasks requiring multi-step reasoning, data lookup, and computation.
**Pros**: Can handle open-ended, multi-step problems.
**Cons**: Harder to control, higher latency, failure modes are complex.

## 5. From Requirements to Architecture: A Practical Checklist

When evaluating a domain LLM project, work through these questions in order:

1. **What is the core task?** Understanding, generation, or transformation?
2. **What knowledge does the model need?** Already in the model, in documents, in databases, or in people's heads?
3. **How often does the knowledge change?** Static (fine-tune), dynamic (RAG), or real-time (tools/APIs)?
4. **What are the accuracy requirements?** Approximate is OK (prompt engineering) vs must be factually grounded (RAG) vs must be precise (fine-tuning + evaluation)?
5. **What are the privacy/compliance constraints?** Can data leave the organization?
6. **What is the compute budget?** API-only, single GPU, or GPU cluster?
7. **What is the latency requirement?** Batch processing (seconds OK) vs real-time (sub-second)?
8. **Who are the users?** Technical (can tolerate imperfection) vs non-technical (must be polished)?

The answers to these questions will naturally converge on one of the architecture patterns above — or a hybrid of them.

---

## Key References

1. **Shanahan (2024)**: *Talking About Large Language Models*.
2. **Bommasani et al. (2021)**: *On the Opportunities and Risks of Foundation Models*.
