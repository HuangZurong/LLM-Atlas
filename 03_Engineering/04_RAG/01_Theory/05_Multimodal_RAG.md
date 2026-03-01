# Multimodal RAG

*Prerequisite: [02_Advanced_RAG.md](02_Advanced_RAG.md).*
*See Also: [../../../02_Scientist/06_Multimodal/01_Vision_Language.md](../../../02_Scientist/06_Multimodal/01_Vision_Language.md) (VLM foundations).*

---

Standard RAG operates on text only. **Multimodal RAG** extends the pipeline to ingest, index, retrieve, and reason over **images, tables, charts, diagrams, audio, and video** alongside text — reflecting how real-world knowledge is actually stored.

## 1. The Problem

Enterprise documents are inherently multimodal:

- **PDF reports** contain charts, tables, and diagrams that carry critical information not present in surrounding text.
- **Technical manuals** rely on schematics and flow diagrams.
- **Meeting recordings** combine speech, slides, and screen shares.
- **Product catalogs** are image-first, with text as metadata.

A text-only RAG pipeline silently drops 30–60% of the information in a typical enterprise document.

## 2. Architecture Patterns

There are three primary strategies for handling multimodal content, each with different trade-offs:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multimodal RAG Strategies                        │
│                                                                     │
│  Strategy A            Strategy B              Strategy C           │
│  Text Summaries        Multimodal Embeddings   Native Multimodal    │
│                                                                     │
│  Image ─► LLM ─► Text  Image ─► CLIP ─► Vec   Image ─► Store raw  │
│  Table ─► LLM ─► Text  Text  ─► CLIP ─► Vec   Text  ─► Store raw  │
│  Text  ─► Store text   Audio ─► CLAP ─► Vec   Audio ─► Store raw  │
│       │                      │                      │               │
│       ▼                      ▼                      ▼               │
│  Text Vector DB         Unified Vector DB      Multimodal LLM      │
│  (standard RAG)         (cross-modal search)   (direct reasoning)  │
└─────────────────────────────────────────────────────────────────────┘
```

### Strategy A: Multimodal → Text Summaries

- **Approach**: Use a Vision-Language Model (VLM) to convert images, tables, and charts into **text descriptions** during ingestion. Then run standard text RAG.
- **Flow**: `PDF → Parse → Extract images/tables → VLM summarizes each → Text chunks + summaries → Embed → Vector DB`
- **Pros**: Reuses the entire existing text RAG stack; simple to implement.
- **Cons**: Lossy — the text summary may miss visual details (spatial layout, color coding, precise data points in charts).
- **Best For**: Quick wins; documents where images are supplementary, not primary.

### Strategy B: Multimodal Embeddings

- **Approach**: Use models that embed **both** text and images into the **same vector space** (e.g., CLIP, CLAP for audio, ImageBind for universal).
- **Flow**: `Query (text) → CLIP embed → Search across text chunks AND image embeddings → Return mixed results`
- **Key Models**:
  - **CLIP** (OpenAI): Text ↔ Image alignment. Query "a cat on a roof" retrieves matching images.
  - **CLAP**: Text ↔ Audio alignment.
  - **ImageBind** (Meta): Aligns 6 modalities (text, image, audio, depth, thermal, IMU) in a shared space.
- **Pros**: True cross-modal retrieval; can find images from text queries and vice versa.
- **Cons**: Embedding alignment is imperfect — text-to-text and image-to-image retrieval is stronger than cross-modal.
- **Best For**: Image-heavy corpora (product catalogs, design systems, medical imaging).

### Strategy C: Native Multimodal Generation

- **Approach**: Store raw multimodal content. At generation time, pass both retrieved text **and** raw images/tables directly to a multimodal LLM (GPT-4o, Gemini, Claude).
- **Flow**: `Query → Retrieve text chunks + linked raw images → Feed all to multimodal LLM → Answer`
- **Pros**: No information loss; the LLM reasons directly over visual content.
- **Cons**: Higher token cost; requires multimodal LLM; context window limits on image count.
- **Best For**: High-stakes applications where visual detail matters (medical imaging, engineering diagrams, financial charts).

### Hybrid: The Production Pattern

In practice, production systems combine strategies:

1. **Ingest**: Extract images/tables → Generate text summaries (Strategy A) **AND** store raw assets with chunk linkage.
2. **Index**: Embed text summaries for retrieval (text vector DB).
3. **Retrieve**: Standard text retrieval returns chunks + linked raw images.
4. **Generate**: Pass text context + raw images to multimodal LLM (Strategy C).

This gives the **retrieval precision** of text search with the **generation quality** of native multimodal reasoning.

## 3. Ingestion Pipelines by Modality

### 3.1 Document Ingestion (Images, Tables, Charts)

```
Raw Document (PDF / PPTX / HTML)
    │
    ├── Text Extraction ──► Text chunks (standard pipeline)
    │
    ├── Table Extraction
    │   ├── Rule-based (Camelot, Tabula) ──► Structured CSV/JSON
    │   └── VLM-based (screenshot → GPT-4o) ──► Markdown table + summary
    │
    ├── Image Extraction
    │   ├── Classify: chart vs. diagram vs. photo vs. logo
    │   ├── Charts ──► VLM ──► Data description + key takeaways
    │   ├── Diagrams ──► VLM ──► Component/flow description
    │   └── Photos ──► VLM ──► Caption + CLIP embedding
    │
    └── Link: each extracted element maintains a pointer to its source chunk/page
```

**Key Tools**:
- **Unstructured.io**: Open-source document parser with multimodal element classification.
- **LlamaParse**: LLM-powered PDF parsing that preserves table and image structure.
- **Docling** (IBM): Document understanding with layout analysis and table extraction.

#### Table RAG: A Special Case

Tables are the hardest modality because they are **structured** (rows × columns) but stored as **unstructured** text in most RAG pipelines.

- **Problem**: Flattening a table into text ("Row 1: Apple, Revenue: $100B, Growth: 5%...") destroys the relational structure that makes tables useful.
- **Solutions**:
  - **Markdown Preservation**: Store tables as Markdown, which LLMs can parse natively.
  - **Text-to-SQL over extracted tables**: Parse tables into a SQLite DB, then use Text-to-SQL for precise queries (see Advanced RAG Section 1.1).
  - **Table Summarization**: VLM generates a natural language summary of the table's key insights for embedding, while the raw table is stored for generation.

### 3.2 Audio Ingestion

```
Audio Source (podcast, meeting, call center)
    │
    ├── Speech-to-Text (Whisper, Conformer)
    │   └── Timestamped transcript ──► Text chunks (standard pipeline)
    │
    ├── Speaker Diarization (who spoke when)
    │   └── Speaker-labeled segments ──► Metadata per chunk
    │
    └── Optional: CLAP embedding of raw audio segments
        └── Enables cross-modal retrieval (text query → audio clip)
```

- **Primary strategy**: Convert to text via ASR, then standard text RAG. This is Strategy A applied to audio.
- **Rich metadata**: Timestamps + speaker labels make audio RAG uniquely valuable for meeting search ("What did Alice say about the Q3 budget?").
- **Key Tools**: OpenAI Whisper, Assembly AI, Deepgram, pyannote (diarization).

### 3.3 Video Ingestion

```
Video Source (training video, presentation, surveillance)
    │
    ├── Audio Track ──► ASR ──► Timestamped transcript
    │
    ├── Visual Track
    │   ├── Keyframe Extraction (scene change detection / fixed interval)
    │   ├── Keyframes ──► VLM ──► Frame descriptions
    │   └── Keyframes ──► CLIP embedding
    │
    ├── Slide Detection (for presentations)
    │   └── OCR + VLM ──► Slide content as text
    │
    └── Alignment: merge transcript + visual descriptions by timestamp
        └── Unified multimodal chunks (text + linked keyframes)
```

- **Core challenge**: Video = audio + visual + temporal. The alignment across these streams is the hard problem.
- **Chunking strategy**: Chunk by **scene** (visual change detection) or by **topic** (transcript segmentation), not by fixed time intervals.
- **Retrieval**: Text search on transcript + descriptions returns chunks with timestamp ranges. The UI can jump to the exact moment in the video.
- **Key Tools**: FFmpeg (extraction), SceneDetect (scene boundaries), Twelve Labs (video understanding API).

## 4. Multimodal Retrieval & Re-Ranking

Standard text retrieval techniques (Advanced RAG Sections 6.1–6.3) need adaptation when results are a mix of text, images, and audio.

### 4.1 Cross-Modal Retrieval

- **Same-space search** (Strategy B): A text query embedded via CLIP retrieves both text and images from a unified vector space. Works out-of-the-box but cross-modal similarity scores are noisier than within-modality scores.
- **Parallel retrieval + merge**: Run separate retrievals per modality (text→text, text→image via CLIP) and merge using RRF (Advanced RAG Section 6.1). More robust than single-space search.

### 4.2 Cross-Modal Re-Ranking

- **Problem**: A cross-encoder re-ranker trained on (text, text) pairs cannot score (text, image) pairs.
- **Solutions**:
  - **VLM-as-Reranker**: Pass (query, image) to a VLM and ask "Is this image relevant to the query? Score 1–5." Expensive but accurate.
  - **Caption → text reranker**: Use the image's text summary as a proxy, then apply standard cross-encoder re-ranking on (query, caption). Cheaper but lossy.
  - **Late fusion**: Re-rank text and image results independently within their modalities, then interleave by normalized score.

### 4.3 Vector DB Support

| Vector DB | Multimodal Support |
|---|---|
| **Weaviate** | Native multi-modal modules (img2vec, multi2vec-clip). Store and search images, text, and audio in one collection. |
| **Qdrant** | Named vectors — store multiple embeddings (text, CLIP, CLAP) per point; search any subset. |
| **Milvus** | Multi-vector fields. Dynamic schema supports heterogeneous embeddings. |
| **Chroma** | Text-focused. Multimodal requires external embedding + metadata linkage. |
| **Pinecone** | Single-vector per record. Multimodal via metadata linking or CLIP embeddings as primary vector. |

## 5. Interaction with Advanced RAG Techniques

Multimodal content changes how upstream RAG stages operate:

| Advanced RAG Stage | Multimodal Impact |
|---|---|
| **Query Translation** (Multi-query, HyDE) | HyDE can generate hypothetical text descriptions of images. Multi-query can generate modality-specific variants ("find a chart showing..." vs "find text about..."). |
| **Routing** | Route to different indexes by modality: text queries → text index, image-description queries → CLIP index, structured queries → SQL over extracted tables. |
| **Indexing** | Multi-representation indexing is essential: index text summary for retrieval, link to raw image for generation. |
| **Re-Ranking** | Requires cross-modal re-ranking (Section 4.2). |
| **Generation** | Multimodal LLM receives mixed context (text + images). Prompt must instruct the LLM to reference visual evidence. |

## 6. Evaluation

Multimodal RAG is harder to evaluate than text-only RAG because standard metrics assume text-only pipelines.

### 6.1 Retrieval Metrics

| Metric | Text RAG | Multimodal RAG |
|---|---|---|
| Precision@K | Standard | Need cross-modal relevance judgments — is this image relevant to the query? |
| Recall | Standard | Must account for relevant content across modalities |
| MRR (Mean Reciprocal Rank) | Standard | Applicable but requires multimodal ground truth |

**Practical approach**: Human-annotated (query, relevant_asset) pairs where assets can be text chunks, images, or tables.

### 6.2 Generation Metrics

| Metric | What it measures | Multimodal adaptation |
|---|---|---|
| **Faithfulness** | Is the answer supported by retrieved context? | Use VLM-as-a-Judge: "Given this chart, is the claim 'revenue grew 20%' supported?" |
| **CLIPScore** | Alignment between generated text and source image | Higher score = answer is semantically consistent with the visual content |
| **Table Accuracy** | Are numbers from tables extracted correctly? | Compare extracted values against ground truth structured data |

### 6.3 End-to-End

- **Modality Coverage**: Did the system retrieve from the right modality? If the answer depends on a chart but only text was retrieved, this is a modality-level retrieval failure.
- **Cross-modal Grounding**: Does the answer correctly reference visual evidence? ("As shown in Figure 3..." — is Figure 3 actually relevant?)
