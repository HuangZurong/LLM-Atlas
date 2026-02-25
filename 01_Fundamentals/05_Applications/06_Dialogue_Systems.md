# Dialogue Systems

*Prerequisite: [../04_Transformer_Era/03_Pre_Training_Paradigms.md](../04_Transformer_Era/03_Pre_Training_Paradigms.md).*

---

**Task**: Enable natural language conversation between humans and machines — the ultimate interface for AI assistants and customer service.

## 1. Taxonomy

| Category | Goal | Example |
|:---------|:-----|:--------|
| **Task-Oriented (TOD)** | Complete specific tasks via slots | Booking flights, tech support, checking weather |
| **Open-Domain (Chitchat)** | Engaging, long-term conversation | Microsoft XiaoIce, Replika |
| **Question Answering** | Providing factual answers | Siri/Alexa fact-check, Google Assistant |
| **Hybrid (LLM-based)** | Multi-purpose reasoning + tools | ChatGPT, Claude, Gemini |

## 2. Technical Evolution

```
Rule-based (Regex, AIML, State Graphs)
    ↓
Modular Pipeline (NLU → DST → Policy → NLG)
    ↓
End-to-End Neural (Seq2Seq, Memory Networks)
    ↓
Pre-trained Generative (DialoGPT, Meena, LaMDA)
    ↓
LLM + Tools + RAG (Instruction-following, tool-use via API)
```

## 3. Industrial Systems

| System | Company | Year | Venue | Description |
|:-------|:--------|:-----|:------|:------------|
| **AliMe Chat** | Alibaba | 2017 | ACL | Hybrid retrieval + Seq2Seq chatbot for e-commerce; handles millions of queries daily on Taobao |
| **XiaoIce** | Microsoft | 2020 | MIT Press | Social chatbot focused on emotional bond (EQ) and engagement (CPS: Chat-turns Per Session) |
| **Google Duplex** | Google | 2018 | Production | AI for making real-world phone calls; handles natural disfluencies ("um", "ah") and complex scheduling |
| **FoodGPT** | Meituan | 2025 | KDD | Domain-specific LLM for food delivery; uses FoodInstruct dataset to handle complex ordering and logistics queries |
| **Alexa Architecture** | Amazon | 2021 | Blog | Massive modular pipeline serving millions of Echo devices; handles wake-word, ASR, NLU, and Skill routing |
| **Siri Semantic Parser** | Apple | 2021 | arXiv | On-device semantic parsing for privacy-preserving voice commands; uses distilled Transformers for low-latency |

## 4. Production Reality: The Pipeline

Industrial dialogue systems (especially voice) are complex pipelines:

```
User Voice  →  ASR (Speech-to-Text)
                   ↓
Input Text  →  NLU (Intent + Slot Extraction)
                   ↓
Dialogue State Tracking (DST)  ←  Maintains context over turns
                   ↓
Dialogue Policy (Action Selection)  ←  Call API? Ask clarifying question?
                   ↓
NLG (Natural Language Generation)  ←  Template-based or Neural
                   ↓
Output Text →  TTS (Text-to-Speech)  →  User
```

## 5. The LLM Disruption in Dialogue

LLMs are collapsing this modular pipeline into a single model, but production still faces hurdles:
- **Controllability**: LLMs might promise discounts they shouldn't.
- **Latency**: Voice interfaces require **< 500ms** response time; LLM generation can be too slow.
- **Tool Integration**: Using "Function Calling" to let LLMs interact with databases and APIs (e.g., checking order status).

## 6. Practical Engineering Insights

### 6.1 Multi-turn Context Management
How to remember the user said "it" referred to "the blue shoes" 3 turns ago?
- **DST**: Explicitly tracking slot values in a state object.
- **Context Window**: Appending previous turns to the LLM prompt (expensive).

### 6.2 Error Recovery
What to do when the model doesn't understand?
- **Clarification Strategies**: "Did you mean Tokyo or Kyoto?"
- **Fallbacks**: Routing to a human agent when intent confidence is low.

## Key References

- Qiu et al., "[AliMe Chat: A Sequence to Sequence and Rerank based Chatbot Engine](https://aclanthology.org/P17-2079/)", ACL 2017
- Zhou et al., "[The Design and Implementation of XiaoIce](https://arxiv.org/abs/1812.08989)", Computational Linguistics (MIT Press) 2020
- Leviathan et al., "[Google Duplex: An AI System for Accomplishing Real-World Tasks Over the Phone](https://ai.googleblog.com/2018/05/duplex-ai-system-for-natural-conversation.html)", Google AI Blog 2018
- Chen et al., "[G-Eval: NLG Evaluation using GPT-4 with Better Chain-of-Thought Alignment](https://arxiv.org/abs/2303.16634)", arXiv 2023
- Gao et al., "[Neural Approaches to Conversational AI](https://arxiv.org/abs/1809.08269)", Foundations and Trends in IR 2019

---

_Next: [Search and Recommendation](./07_Search_and_Recommendation.md) — NLP powering the understanding layer of search and recommendation._
