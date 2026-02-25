# Information Extraction

*Prerequisite: [02_Sequence_Labeling.md](02_Sequence_Labeling.md).*

---

**Task**: Extract structured information (entities, relations, events) from unstructured text — turning documents into queryable databases.

## 1. Sub-tasks

| Task | Description | Example |
|:-----|:-----------|:--------|
| **Relation Extraction** | Identify relationships between entities | "Elon Musk founded SpaceX" → (Elon Musk, founded, SpaceX) |
| **Event Extraction** | Detect events and their arguments | "Apple acquired Beats for $3B" → {event: acquisition, buyer: Apple, target: Beats, price: $3B} |
| **Knowledge Graph Construction** | Build entity-relation graphs from text | Wikipedia → (entities, relations, attributes) |
| **Entity Linking** | Map text mentions to unique KG entries | "Paris" → [Paris, France] (not Paris Hilton) |

## 2. Technical Evolution

```
Rule-based (Regex, Dependency patterns)
    ↓
Feature-based ML (SVM/CRF with linguistic features)
    ↓
Deep Pipeline (NER → R.E. → Event detection)
    ↓
Joint Extraction (Multi-task learning for entities and relations)
    ↓
LLM-based (Generative IE via structured prompting)
```

## 3. Industrial Systems

| System | Company | Year | Venue | Description |
|:-------|:--------|:-----|:------|:------------|
| **Knowledge Vault** | Google | 2014 | KDD | Web-scale probabilistic knowledge fusion using 16 extraction systems; produced 1.6B triples with 324M at >0.7 confidence |
| **AliCoCo** | Alibaba | 2020 | SIGMOD | E-commerce cognitive concept net defining user needs; deployed on Taobao connecting users, items, and shopping scenarios |
| **AliCG** | Alibaba | 2021 | KDD | Fine-grained conceptual graph for semantic search; deployed at UC Browser |
| **AliMe KG** | Alibaba | 2020 | arXiv | Domain KG for shopping guide and QA; BERT-based relation extraction; millions of shopping-related triples |
| **OpenTag** | Amazon | 2018 | KDD | Active learning for product attribute value extraction; achieves 83% F1 with minimal annotated samples |
| **Industry-Scale KGs** | Google, Microsoft, Meta, eBay, IBM | 2019 | VLDB | Joint paper comparing KG systems: Google Knowledge Graph, Bing KG, Facebook Social Graph |
| **DeepIE** | Baidu | 2022 | arXiv | Industrial information extraction framework using pre-trained models for zero-shot and few-shot relation extraction |

## 4. The LLM Shift

LLMs have revolutionized IE by enabling **Zero-shot Information Extraction**. Complex schemas that previously required multi-stage pipelines can now be handled with structured prompting (e.g., JSON output).

However, in production:
- **Consistency**: LLMs may hallucinate relations; specialized models are more deterministic.
- **Cost**: For extracting triples from billions of web pages, specialized BERT-class models are still 100x more cost-effective.
- **Hybrid Approach**: Using LLMs to generate "silver" training data for smaller, faster student models.

## 5. Industrial Challenges

### 5.1 Schema Evolution
In industry, the "schema" (what entities/relations to extract) changes frequently.
- **Solution**: Open Information Extraction (OpenIE) to discover new relations without a fixed schema, followed by human-in-the-loop schema mapping.

### 5.2 Conflict Resolution (Fusion)
When multiple documents provide conflicting information (e.g., "Apple has 150k employees" vs. "160k").
- **Solution**: Probabilistic fusion and source reliability scoring (as used in Knowledge Vault).

## Key References

- Dong et al., "[Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion](https://dl.acm.org/doi/10.1145/2623330.2623623)", KDD 2014
- Luo et al., "[AliCoCo: Alibaba E-commerce Cognitive Concept Net](https://arxiv.org/abs/2003.13230)", SIGMOD 2020
- Li et al., "[AliCG: Fine-grained and Evolvable Conceptual Graph Construction](https://arxiv.org/abs/2106.01686)", KDD 2021
- Noy et al., "[Industry-Scale Knowledge Graphs: Lessons and Challenges](https://dl.acm.org/doi/10.14778/3352063.3352101)", VLDB 2019
- Zheng et al., "[OpenTag: Open Attribute Value Extraction from Product Profiles](https://arxiv.org/abs/1806.01264)", KDD 2018

---

_Next: [Machine Translation](./04_Machine_Translation.md) — Converting text across languages at industrial scale._
