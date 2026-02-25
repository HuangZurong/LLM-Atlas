# Best Practice: LLM Open Source Ecosystem (2025)

*Prerequisite: [../01_Theory/01_Intelligence_Landscape.md](../01_Theory/01_Intelligence_Landscape.md).*

The LLM engineering landscape is defined by its open-source tools. This document categorizes the must-know projects for building production-grade systems.

---

## 1. Inference & Serving
*The engine that runs the model.*

| Project | Description | Key Tech |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | The industry standard for high-throughput serving. | PagedAttention, Continuous Batching |
| [SGLang](https://github.com/sgl-project/sglang) | Fastest for complex generation and structured output. | RadixAttention (Automatic Prefix Caching) |
| [Ollama](https://github.com/ollama/ollama) | Simplest way to run LLMs locally on macOS, Linux, and Windows. | llama.cpp under the hood |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | The foundation for CPU inference and 4-bit GGUF quantization. | Metal/CUDA/OpenCL backends |

## 2. Abstraction & Orchestration
*The glue between your application and models.*

| Project | Description | Use Case |
|---|---|---|
| [LiteLLM](https://github.com/BerriAI/litellm) | Unified interface for 100+ LLM APIs. | Proxy, load balancing, cost tracking |
| [LangChain](https://github.com/langchain-ai/langchain) | The most popular framework for building LLM apps. | RAG, Agents, standard interfaces |
| [LlamaIndex](https://github.com/run-llama/llama_index) | Best-in-class data framework for LLM applications. | Complex RAG, index structures |
| [DSPy](https://github.com/stanfordnlp/dspy) | Programmatic prompt optimization (no more manual prompt tuning). | Compiling prompts into models |

## 3. Evaluation & Observability
*Measuring and seeing what's happening.*

| Project | Description | Metrics |
|---|---|---|
| [Langfuse](https://github.com/langfuse/langfuse) | Open-source tracing and observability. | Cost, latency, user feedback |
| [Arize Phoenix](https://github.com/Arize-ai/phoenix) | Evaluation and tracing for RAG and Agents. | Hallucination, relevance, precision |
| [RAGAS](https://github.com/explodinggradients/ragas) | Framework for automated evaluation of RAG. | Faithfulness, context recall |
| [DeepEval](https://github.com/confident-ai/deepeval) | Unit testing framework for LLMs. | Toxicity, bias, answer correctness |

## 4. Structured Generation & Security
*Ensuring outputs are safe and follow schema.*

| Project | Description | Technique |
|---|---|---|
| [Outlines](https://github.com/outlines-dev/outlines) | Guaranteed structured generation (JSON, Regex). | Constrained Decoding |
| [Llama Guard](https://huggingface.co/meta-llama/Llama-Guard-3-8B) | Meta's safety model for input/output filtering. | Safety Classification |
| [Garak](https://github.com/leondz/garak) | Vulnerability scanner for LLMs (injection, jailbreak). | Red Teaming |

## 5. Industrial Trends 2025
1. **Vertical Integration**: Organizations are moving from generic APIs to self-hosted MoE models (DeepSeek-V3) using SGLang/vLLM for 50%+ cost reduction.
2. **Compound AI Systems**: Focus shifting from "better models" to "better systems" (routing + caching + multi-step verification).
3. **Small Model Dominance**: High-performance 1B-8B models (Llama 3, Qwen 2.5) are replacing larger models for routine extraction and classification.

---
**Sources**:
- [vLLM Documentation](https://docs.vllm.ai/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Langfuse Tracing Guide](https://docs.langfuse.com/tracing)
- [DeepSeek-V3 Technical Report](https://github.com/deepseek-ai/DeepSeek-V3)
