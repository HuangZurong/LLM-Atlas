# 08 · 进阶上下文范式

*注：本文从近期学术研究（例如 “A Survey of Context Engineering for Large Language Models”）中提炼出可落地的工程洞见。虽然学术界常把“Context Engineering”当作一个涵盖 RAG 与 Prompt Engineering 的宽泛总称，但我们会严格以生产导向的定义来过滤这些概念：**管理 token 预算、优先级与信息生命周期**。*

*在 CE 流水线中的位置：第 3 步（Compress & Degrade）和第 4 步（Assemble & Observe）的高级策略*

---

## 1. 信息论视角的上下文压缩

基础实现通常依赖启发式压缩（截断、滑动窗口、抽取式摘要），而学术研究则提出了高度优化、且考虑算力成本的压缩算法，能在不损失语义的前提下把 token 占用降低多达 50%。

### 1.1 基于困惑度的剪枝（例如 LLMLingua）

与其简单丢弃最旧消息，这种方法会使用一个更小、更便宜的语言模型（如 Llama-1B），计算上下文窗口中每个 token 或句子的信息熵（困惑度）。
- **机制**：保留高困惑度 token（出人意料、信息密集的事实），激进剪掉低困惑度 token（可预测语法、填充词、重复结构）。
- **工程启示**：在 `Context_Compression.py` 流水线中，为非关键上下文层（例如原始 RAG 检索 chunks）集成轻量级困惑度过滤器，在送入昂贵前沿模型前，最大化高信号 token 的密度。

### 1.2 注意力引导式剪枝（Attention-Guided Pruning）

利用 Transformer 内部的交叉注意力机制。
- **机制**：通过一次快速前向传播，系统分析哪些历史 token 相对当前用户 query 获得的注意力权重最低，再把这些“被忽略”的 token 从上下文中剔除。
- **工程启示**：这对动态记忆管理非常有效。与其做摘要，不如直接丢掉那些在数学意义上与当前任务图完全无关的对话轮次。

---

## 2. 结构与模态感知的预算分配

随着上下文窗口从纯文本扩展到数据库、知识图谱（Knowledge Graphs, KGs）和多模态输入（图像/音频），`TokenBudgetController` 必须适应非线性的 token 成本。

### 2.1 结构化数据的文本化（Verbalization）

把原始 JSON、SQL schema 或知识图谱三元组直接塞进上下文窗口，既极其低效，也常常会削弱推理表现。
- **Schema Pruning**：在注入上下文之前，先基于与用户 query 的 embedding 相似度，动态裁剪 SQL schema 中不相关的表和列。
- **Linearization（语言化）**：把图结构转换为优化过的自然语言叙述或紧凑的 Markdown 表格。
- **工程启示**：在上下文组装器中专门创建一个 `StructuredDataLayer`，自动把复杂数据结构序列化为 token 效率更高的格式，而不是直接做原始字符串 dump。

### 2.2 多模态预算降级策略（Multimodal Budget Degradation）

图像和音频消耗上下文容量的方式不同于文本（例如 OpenAI 按 512x512 图像 tile 计算 Vision tokens）。
- **机制**：当总 token 预算逼近危险上限时，上下文管理器不应只会删除文本，而应系统性降低多模态内容的保真度。
- **工程启示**：为上下文历史中的图像实现多级降级链：`High-Res (Multi-tile) -> Low-Res (Single 85-token tile) -> Text Description Only (Image caption) -> Dropped`。

---

## 3. 多代理上下文编排

在多代理系统（Multi-Agent Systems, MAS）中，如果把整个上下文窗口在代理之间来回传递，成本会呈二次爆炸，并引发 context rot。高级编排需要严格的通信协议。

### 3.1 以状态为上下文总线

代理不应直接传递原始对话历史。它们应该读写一个共享的、强类型的状态对象（Context Bus）。

### 3.2 有损与无损的上下文交接

- **无损交接（Lossless Handoff，Prefix Cache Optimized）**：向子代理交接任务时，传递静态上下文（system prompts、严格规则）的**完全相同 token 序列**。这可以保证高**Prefix Caching** 命中率，最多节省 90% 的计算成本。
- **有损交接（Lossy Handoff，Semantic Compression）**：当代理完成研究任务后，它绝不能把 scratchpad 或原始 web search 结果交给下一个代理，而必须先执行一次 “compaction” 步骤，只传递一个高密度摘要（有损交接），以便将全局上下文占用压到最低。

---

## 4. 高级评估框架

标准的 “Needle-In-A-Haystack” (NIAH) 测试只衡量检索能力。更高级的上下文工程，需要评估模型在长上下文中的*推理*与*利用程度*。

### 4.1 多针推理（Multi-Needle Reasoning）

测试模型能否从 100K+ token 上下文窗口中检索出多个分散事实，并综合出一个新答案，而不是仅仅定位某个孤立字符串。

### 4.2 上下文利用与参数记忆

一个关键指标是：模型究竟是否真的*信任*提供给它的上下文，而不是依赖自身预训练权重。
- **机制**：在上下文窗口中间注入反事实信息（假事实）。
- **工程启示**：如果模型忽略了注入内容，仍按自身预训练知识作答，那么你的上下文编排就失败了（通常是优先级摆放不佳，或出现 Lost-in-the-Middle 效应）。这个指标可以直接评估 `ContextComposer` 的有效性。
