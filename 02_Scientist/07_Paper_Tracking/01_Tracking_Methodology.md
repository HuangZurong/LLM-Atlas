# Paper Tracking Methodology

*Prerequisite: None (Entry point for Paper Tracking module).*
*See Also: [../../02_Scientist/03_Pre_Training/09_Research_Trends.md](../03_Pre_Training/09_Research_Trends.md) (industry trends & roadmap).*

---

## 1. Information Sources (by Priority)

### 1.1 Primary Sources (First-Party)

| Source | Frequency | Value |
|:--|:--|:--|
| **arXiv cs.CL / cs.LG / cs.AI** | Daily | Latest preprints; first-publication venue for all major work |
| **Top Conferences** (NeurIPS, ICML, ICLR, ACL, EMNLP) | Annual | Peer-reviewed, high-quality work |
| **Company Technical Reports** (OpenAI, Google, Meta, DeepSeek, Anthropic) | Irregular | Industrial frontier; typically rich in engineering details |

### 1.2 Curated Sources (Second-Party)

| Source | Type | Characteristics |
|:--|:--|:--|
| **Hugging Face Daily Papers** | Daily picks | Community-voted, broad coverage |
| **Papers With Code** | Paper + code + leaderboard | Strong reproducibility |
| **Semantic Scholar / Connected Papers** | Citation graph | Track paper influence and connections |
| **Twitter/X Academic Circle** | Real-time discussion | Author first-hand commentary; high noise |
| **AI Podcasts** (Latent Space, Gradient Dissent) | In-depth interviews | Authors share motivation and unpublished details |

---

## 2. Filtering Criteria

Signals that a paper is worth deep reading:

### 2.1 Strong Signals (Must-Read)
- Technical reports from **top labs** (OpenAI, Google DeepMind, Meta FAIR, Anthropic, DeepSeek)
- **Oral/Spotlight** papers at top conferences
- Achieves **SOTA on multiple leaderboards** simultaneously
- Discussed and shared by **multiple prominent researchers**

### 2.2 Medium Signals (Worth Tracking)
- Proposes a **new paradigm/framework** (not incremental improvement)
- Has **open-source code and weights**
- Solves a **known important problem** (e.g., KV Cache compression, long context, hallucination)

### 2.3 Weak Signals (Quick Skim)
- Incremental improvement on a single benchmark only
- No code, no reproduction details
- Over-reliance on prompt engineering "tricks"

---

## 3. Paper Reading Framework (Three-Pass Method)

### Pass 1: 5-Minute Skim
- Read **Title + Abstract + Figures/Tables**
- Answer: What problem does this paper solve? What method? What results?
- Decide: Worth a second pass?

### Pass 2: 30-Minute Focused Read
- Read **Introduction + Method + Key Experiments**
- Skip proofs and appendices
- Answer:
  - What is the core innovation? (one sentence)
  - Key difference from existing methods?
  - Is the experimental setup fair? Are baselines reasonable?
  - What limitations did the authors not mention?

### Pass 3: Deep Analysis (important papers only)
- Read Method line-by-line, derive the math
- Reproduce key experiments or read source code
- Write up in the classic paper analysis module

---

## 4. Tracking Record Template

Each paper entry should follow this format:

```markdown
### [Short Name] Paper Title (Year)
- **Authors/Institution**:
- **Link**: arXiv / official blog
- **One-line Summary**:
- **Core Innovation**:
- **Key Results**: (SOTA numbers, efficiency gains, etc.)
- **Relation to Existing Work**: (What does it improve? What does it replace?)
- **Limitations**:
- **Implications for Our Knowledge Base**: (Does it affect our modules? Which ones need updating?)
```

---

## 5. Tracking Cadence

| Frequency | Action |
|:--|:--|
| **Daily** | Browse arXiv / HF Daily Papers titles, bookmark interesting ones |
| **Weekly** | Deep-read 2-3 bookmarked papers, update Frontiers files |
| **Monthly** | Review monthly trends, decide if main 02_Scientist modules need updates |
| **Quarterly** | Full knowledge base audit against top conference accepted papers |
