# Knowledge Graph Integration: Structured Knowledge Meets LLMs

*Prerequisite: [05_RAG_Architecture.md](05_RAG_Architecture.md).*

---

Knowledge Graphs (KG) and LLMs are complementary: KGs provide structured, verifiable facts and relationships; LLMs provide natural language understanding and generation. This document covers when and how to combine them.

## 1. When Does a Knowledge Graph Add Value?

A KG is not always necessary. It adds value when your domain knowledge has these characteristics:

### 1.1 KG is Worth Building When:

- **Rich entity relationships**: The domain has clear entities (equipment, personnel, phases, regulations) with meaningful relationships between them
- **Multi-hop reasoning is needed**: "Which subcontractors are affected if steel delivery is delayed?" requires traversing: Steel → Supplier → Delivery Schedule → Dependent Tasks → Assigned Subcontractors
- **Consistency matters**: KG can be validated for contradictions; a document corpus cannot
- **Knowledge is reusable across tasks**: The same entity-relationship structure serves Q&A, risk analysis, and planning
- **Provenance tracking**: Every fact in a KG has a traceable source

### 1.2 KG is Overkill When:

- Knowledge is primarily narrative/unstructured (opinions, analysis, descriptions)
- The domain doesn't have clear entity types or relationship patterns
- You lack domain experts to define the schema and validate the graph
- The knowledge changes so rapidly that maintaining the graph is impractical
- Simple vector retrieval already achieves sufficient accuracy

### 1.3 KG vs RAG: A Comparison

| Dimension | RAG (Vector Retrieval) | Knowledge Graph |
| :--- | :--- | :--- |
| **Knowledge form** | Unstructured text chunks | Structured triples (entity-relation-entity) |
| **Query type** | "Tell me about X" | "What is the relationship between X and Y?" |
| **Multi-hop reasoning** | Weak (retrieves independent chunks) | Strong (traverses relationships) |
| **Precision** | Approximate (semantic similarity) | Exact (structured query) |
| **Construction cost** | Low (chunk + embed) | High (schema design + entity extraction + validation) |
| **Maintenance** | Easy (re-index documents) | Hard (update entities and relations) |
| **Explainability** | Medium (can cite source documents) | High (can show reasoning path) |

## 2. Knowledge Graph Design for Domain LLMs

### 2.1 Schema Design

The schema defines what entity types and relationship types exist. This is the most critical decision — get it wrong and the entire graph is useless.

**Process**:
1. Collect 20-30 representative domain questions that require structured knowledge
2. For each question, identify what entities and relationships are needed to answer it
3. Generalize into entity types and relationship types
4. Validate with domain experts
5. Start small (5-10 entity types, 10-15 relationship types), expand later

**Example schema for infrastructure construction-operations**:

```
Entity Types:
- Project, Phase, Milestone
- Organization, Person, Role
- Equipment, Material, Supplier
- Risk, Issue, Decision
- Regulation, Standard, Specification
- Location, Facility, System

Relationship Types:
- Project --[has_phase]--> Phase
- Phase --[depends_on]--> Phase
- Phase --[assigned_to]--> Organization
- Equipment --[supplied_by]--> Supplier
- Equipment --[installed_in]--> Facility
- Risk --[affects]--> Phase
- Risk --[mitigated_by]--> Decision
- Decision --[references]--> Regulation
- Person --[has_role]--> Role
- Issue --[caused_by]--> Equipment
```

### 2.2 Entity and Relation Extraction

How to populate the graph from domain documents:

| Method | Accuracy | Throughput | Cost | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Manual expert annotation** | Highest | Very low | High | Gold-standard seed data, validation |
| **LLM-based extraction** | High | High | Medium | Bulk extraction from documents |
| **NER + RE models** | Medium-high | Very high | Low (after training) | Large-scale automated extraction |
| **Rule-based extraction** | Variable | High | Low | Highly structured documents (tables, forms) |

**Recommended approach**: Hybrid pipeline

```
Documents → LLM extraction (GPT-4/Claude) → Candidate triples → Rule-based filtering → Expert review (sample) → Knowledge Graph
```

LLM extraction prompt example:
```
Given the following text from a construction project report, extract all entities and relationships.

Entity types: Project, Phase, Organization, Equipment, Risk, Regulation
Relationship types: has_phase, depends_on, assigned_to, supplied_by, affects, references

Text: """
The runway rehabilitation project entered the asphalt paving phase in March 2024.
This phase depends on the completion of base course preparation by ABC Construction.
The main risk is weather delays, which could affect the October deadline.
All work must comply with CAAC MH5001-2021 standards.
"""

Output as JSON triples:
[
  {"head": "Runway Rehabilitation Project", "relation": "has_phase", "tail": "Asphalt Paving Phase"},
  {"head": "Asphalt Paving Phase", "relation": "depends_on", "tail": "Base Course Preparation"},
  {"head": "Base Course Preparation", "relation": "assigned_to", "tail": "ABC Construction"},
  {"head": "Weather Delays", "relation": "affects", "tail": "Asphalt Paving Phase"},
  {"head": "Asphalt Paving Phase", "relation": "references", "tail": "CAAC MH5001-2021"}
]
```

### 2.3 Graph Storage

| Database | Type | Strengths | Best For |
| :--- | :--- | :--- | :--- |
| **Neo4j** | Native graph | Cypher query language, mature ecosystem, visualization | Most domain KG applications |
| **Amazon Neptune** | Managed graph | Scalable, supports both property graph and RDF | Cloud-native deployments |
| **NebulaGraph** | Distributed graph | High performance at scale | Very large graphs (100M+ edges) |
| **NetworkX** | In-memory (Python) | Simple, no infrastructure | Prototyping, small graphs (<100K nodes) |

**Default recommendation**: Neo4j Community Edition for most domain projects. Free, well-documented, excellent visualization tools.

## 3. Integration Patterns: KG + LLM

### 3.1 Pattern A: KG-Enhanced RAG

The simplest integration. Use the KG to improve retrieval, not replace it.

```
User Query → Entity Recognition → KG Lookup (expand context) → Enhanced Query → Vector Retrieval → LLM
```

**How it works**:
1. Extract entities from the user query ("What risks affect the paving phase?")
2. Query the KG for related entities (paving phase → depends_on → base preparation; paving phase → affected_by → weather delays)
3. Use the KG results to expand the retrieval query or add structured context
4. Retrieve relevant documents with the enriched query
5. Pass both KG context and retrieved documents to the LLM

**Advantage**: Minimal changes to existing RAG pipeline. KG acts as a "knowledge booster."

### 3.2 Pattern B: KG as Primary Knowledge Source

For highly structured domains where most questions can be answered by graph traversal.

```
User Query → Intent Classification → ┬→ Graph Query (Cypher/SPARQL) → Structured Answer → LLM (natural language generation)
                                      └→ Fallback to RAG if query can't be mapped to graph
```

**How it works**:
1. Classify the query intent (entity lookup, relationship query, path finding, aggregation)
2. Convert natural language to graph query (Text-to-Cypher)
3. Execute the graph query
4. Pass structured results to LLM for natural language response generation

**Text-to-Cypher example**:

User: "Which suppliers are involved in the runway project?"
```cypher
MATCH (p:Project {name: "Runway Project"})-[:has_phase]->(ph:Phase)-[:uses]->(e:Equipment)-[:supplied_by]->(s:Supplier)
RETURN DISTINCT s.name, e.name, ph.name
```

**Challenge**: Text-to-Cypher is hard. Current LLMs achieve ~70-80% accuracy on complex queries. Mitigate with:
- Few-shot examples in the prompt (show 5-10 query→Cypher pairs)
- Schema description in the prompt (tell the LLM what entity/relation types exist)
- Validation layer (check Cypher syntax before execution)
- Fallback to RAG when confidence is low

### 3.3 Pattern C: KG-Grounded Reasoning

The most sophisticated pattern. The LLM uses the KG as a reasoning tool, traversing it step by step.

```
User Query → LLM (reasoning agent) → [Think: I need to find...] → KG Query → [Observe: results] → [Think: Now I need...] → KG Query → ... → Final Answer
```

**How it works**: The LLM is given access to KG query tools and uses ReAct-style reasoning to decompose complex questions into a series of graph lookups.

**Example**:

User: "If Supplier X goes bankrupt, what project milestones are at risk?"

```
Think: I need to find what equipment Supplier X provides.
Action: KG Query → MATCH (s:Supplier {name: "X"})-[:supplies]->(e:Equipment) RETURN e
Observe: [Crane Model A, Steel Beam Type B]

Think: Now I need to find which phases use this equipment.
Action: KG Query → MATCH (e:Equipment)-[:used_in]->(ph:Phase) WHERE e.name IN ["Crane Model A", "Steel Beam Type B"] RETURN ph
Observe: [Foundation Phase, Steel Structure Phase]

Think: Now I need to find which milestones depend on these phases.
Action: KG Query → MATCH (ph:Phase)-[:required_for]->(m:Milestone) WHERE ph.name IN [...] RETURN m
Observe: [Structural Completion Milestone, Handover Milestone]

Answer: If Supplier X goes bankrupt, the Structural Completion Milestone and Handover Milestone are at risk, because...
```

## 4. Graph Quality Assurance

### 4.1 Validation Rules

Implement automated checks:

- **Schema compliance**: Every triple must use defined entity types and relationship types
- **Cardinality constraints**: e.g., each Phase must have exactly one assigned Organization
- **Temporal consistency**: Start dates must precede end dates; dependency chains must be acyclic
- **Referential integrity**: No dangling references (entity mentioned in a relationship must exist as a node)
- **Duplicate detection**: Merge "ABC Construction Co." and "ABC Construction Company" into one entity

### 4.2 Entity Resolution

The hardest practical problem in KG construction. The same real-world entity appears with different names:

- "ABC Construction" / "ABC Construction Co., Ltd." / "ABC建设集团"
- "Phase 2" / "Second Phase" / "施工二期"

**Approaches**:
1. **String similarity**: Fuzzy matching (Levenshtein, Jaro-Winkler). Simple but brittle.
2. **Embedding similarity**: Embed entity names, cluster similar ones. Better for multilingual.
3. **LLM-based**: Ask an LLM "Are these the same entity?" with context. Most accurate but expensive.
4. **Canonical naming**: Define a naming convention and normalize all entities during extraction.

### 4.3 Ongoing Maintenance

A KG is not a one-time build. Plan for:

- **Incremental updates**: New documents → extract new triples → merge into existing graph
- **Conflict resolution**: New information contradicts existing triples. Which is authoritative?
- **Staleness detection**: Flag entities/relations that haven't been updated beyond a threshold
- **Version control**: Track graph changes over time (who added what, when, from which source)

## 5. Practical Considerations

### 5.1 Start Small

Don't try to build a comprehensive domain KG from day one. Start with:
1. One sub-domain (e.g., "equipment and suppliers" rather than "everything about construction")
2. 3-5 entity types, 5-8 relationship types
3. 1000-5000 triples
4. One integration pattern (Pattern A: KG-Enhanced RAG)

Validate that this small KG actually improves answer quality before scaling up.

### 5.2 Cost-Benefit Reality Check

| KG Size | Construction Effort | Maintenance Effort | Typical Value Add |
| :--- | :--- | :--- | :--- |
| **Small** (1K-10K triples) | 1-2 weeks | Low | Improves specific query types by 10-20% |
| **Medium** (10K-100K triples) | 1-3 months | Medium | Enables multi-hop reasoning, significant quality improvement |
| **Large** (100K+ triples) | 3-12 months | High | Comprehensive domain reasoning, but diminishing returns |

For most domain LLM projects, a medium-sized KG focused on the most important entity types provides the best ROI.

### 5.3 When to Skip KG Entirely

If after reading this document you feel the effort is disproportionate to the benefit, that's a valid conclusion. Many successful domain LLM systems use only RAG + fine-tuning without any knowledge graph. KG is a powerful tool, but it's not mandatory.

The key question: "Do my users frequently ask questions that require traversing relationships between entities?" If yes, invest in a KG. If most questions are "tell me about X" rather than "how does X relate to Y through Z," RAG alone is sufficient.

---

## Key References

1. **Pan et al. (2024)**: *Unifying Large Language Models and Knowledge Graphs: A Roadmap*.
