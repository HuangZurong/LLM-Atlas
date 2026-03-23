# 检索子系统技术设计

## 1. 目标

快速验证「服装图片 + 结构化标签 → 检索匹配模特图」的检索链路。

## 2. 整体架构

### 2.1 全链路

```
全库 (~10w)
  │
  ▼  枚举硬过滤（用户指定了哪些就过滤哪些，未指定的跳过）
候选池 (~N)
  │
  ▼  多路召回（各自独立，带原始分数）
  │   ├── 图像路: fashionSigLIP FAISS ANN → Top-200 + cosine score
  │   ├── 文本关键词路: BM25 → Top-200 + BM25 score（有标签才走）
  │   └── 文本语义路: text embedding FAISS ANN → Top-200 + cosine score（有标签才走）
  │
  ▼  粗排（合并去重 → 分数归一化 → 动态加权融合）
~100 候选
  │
  ▼  精排（逐维度细粒度匹配 → 加权打分 → 各维度分数可解释）
~20 最终结果
```

### 2.2 每层职责

| 阶段 | 输入 → 输出 | 核心目标 | 可容忍 | 时间预算 |
|------|-------------|----------|--------|----------|
| 枚举过滤 | 全库 → 候选池 | 硬约束淘汰 | — | < 10ms |
| 多路召回 | 候选池 → ~500 | **不漏**（高召回） | 噪声多 | < 30ms |
| 粗排 | ~500 → ~100 | **快速去噪** | 排序不够精准 | < 10ms |
| 精排 | ~100 → ~20 | **排准**（高精度） | 可以慢 | < 200ms |

### 2.3 设计原则

1. **逐层收窄、逐层加重**：越往后候选越少，可投入越重的计算
2. **每层引入新信号**：避免同一个信号在多层重复使用而无新增信息
3. **优雅降级**：标签缺失时系统仍可工作，不崩塌


## 3. 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 语言 | Python 3.11+ | uv 管理项目和虚拟环境 |
| UI | **Gradio** | 快速验证界面 |
| 图像 Embedding | **marqo-fashionSigLIP** (`hf-hub:Marqo/marqo-fashionSigLIP`) | 时尚领域专用 |
| 向量检索 | **FAISS** (faiss-cpu / faiss-gpu) | 内存向量索引 |
| 关键词检索 | **rank_bm25** | 纯 Python BM25 |
| 数据存储 | **JSONL 文件** + **numpy .npy 文件** | 全文件化，启动时加载到内存 |
| VLM 打标（可选） | Qwen2-VL / InternVL 等 | 按 tag.md 生成结构化标签 |


## 4. 数据存储

### 4.1 目录结构

```
data/
├── images/                       # 图片文件
├── metadata.jsonl                # 每行一条：元数据 + tag.md 结构化标签
├── image_embeddings.npy          # (N, 768) 图像向量
├── text_embeddings.npy           # (N, 768) 全量标签文本向量（召回用）
├── field_embeddings/             # 字段级 embedding（精排用，入库时预计算）
│   ├── 整体氛围.npy
│   ├── 姿势_妆容.npy
│   ├── 场景.npy
│   ├── 主题风格.npy
│   ├── 脸部特征_妆容特征.npy
│   └── 特定活动.npy
├── faiss_image.index
└── faiss_text.index
```

### 4.2 metadata.jsonl 格式

每行一个 JSON，与 tag.md 一一对应：

```json
{
  "id": 0,
  "image_path": "images/001.jpg",
  "image_type": "model",
  "tags": {
    "模特": {
      "人种": "Asian",
      "性别": "Female",
      "年龄": "26-45",
      "发型": "中长直发，自然黑色，中分",
      "皮肤颜色": "米色",
      "皮肤特征": "无",
      "脸部特征": "自信的微笑，眼睛明亮",
      "妆容": "自然妆",
      "妆容特征": "淡粉色腮红，裸色唇彩",
      "身材": "Slim",
      "姿势": "站立，双手自然下垂",
      "头部姿态": "正面，直视镜头",
      "手持": ""
    },
    "服装": {
      "上装类型": "连衣裙",
      "上装风格": "A字版型，收腰设计",
      "上装长度": "Knee-length",
      "上装剪裁类型": "Slim fit",
      "上装袖长": "Short sleeve",
      "上装颜色": "主色白色，碎花点缀",
      "上装特征": "小碎花印花",
      "下装类型": "",
      "下装风格": "",
      "下装长度": "",
      "下装颜色": "",
      "下装特征": "",
      "搭配": "",
      "配饰": "珍珠耳环",
      "鞋袜": "白色凉鞋"
    },
    "摄影主题": {
      "光线": "柔光，侧面光源",
      "服装氛围": "清新甜美",
      "整体氛围": "清新自然",
      "目标人群": "年轻女性",
      "节日": "",
      "四季": "夏",
      "特定活动": ""
    },
    "镜头语言": {
      "模糊程度": "背景轻微模糊",
      "视角": "正面平视",
      "头部完整度": "完整",
      "模特位置": "中心"
    },
    "场景": "户外花园，绿色植物背景"
  }
}
```

### 4.3 枚举字段清单

| 字段路径 | 枚举值 |
|----------|--------|
| `模特.性别` | Female, Male |
| `模特.年龄` | 0-3, 3-5, 5-9, 10-14, 15-25, 26-45, 46-60, 60+ |
| `模特.人种` | European, South American, Asian, Indian, Middle Eastern, Maasai, African American, African Black |
| `模特.身材` | Slim, Standard, Plus_Size, Muscular |
| `服装.上装长度` | Crop top, Waist-length, Hip-length, Below-hip length, Mid-thigh length, Knee-length, Calf-length, Ankle-length, Full-length, Floor-length, Puddle hem |
| `服装.上装剪裁类型` | Skin-tight, Slim fit, Loose fit, Oversized |
| `服装.上装袖长` | Sleeveless, Cap sleeve, Short sleeve, Elbow-length sleeve, Three-quarter sleeve, Long sleeve, Extra-long sleeve |
| `摄影主题.四季` | 春, 夏, 秋, 冬 |
| `镜头语言.模特位置` | 左上, 中上, 右上, 左中, 中心, 右中, 左下, 中下, 右下 |


## 5. 召回层

### 5.1 枚举硬过滤

只过滤用户**实际指定的**枚举字段。用户未选的字段不参与过滤，不会误杀。内存遍历 metadata 返回匹配 ID 集合。

### 5.2 三路召回

每路独立检索，返回 `(doc_id, score)` 有序列表：

| 路 | 方法 | 返回分数类型 | 执行条件 |
|----|------|-------------|----------|
| 图像路 | fashionSigLIP → FAISS ANN Top-200 | cosine similarity | 始终执行 |
| 文本关键词路 | BM25 Top-200 | BM25 score | query 有文字标签时 |
| 文本语义路 | text embedding → FAISS ANN Top-200 | cosine similarity | query 有文字标签时 |

### 5.3 关于非对称性

query 是服装图，候选是模特图，图像类型不同。fashionSigLIP 对服装视觉特征（颜色、纹理、款式）有较好捕捉，在特征空间中可以找到跨类型关联，但这种关联是粗略的。

**召回阶段容忍粗糙，非对称匹配的精确处理交给精排。**


## 6. 粗排

### 6.1 定位

- **输入**：三路召回合并去重后的 ~500 候选（每个候选带各路原始分数）
- **输出**：~100 候选
- **时间预算**：< 10ms
- **核心目标**：快速去噪，保召回

### 6.2 为什么不用 RRF

RRF 只看排名、丢分数。两个候选在图像路的 cosine 分别是 0.95 和 0.50，若恰好排第1第2，RRF 赋予的分数差异极小，区分度不够。

粗排直接用原始分数做归一化加权，**保留分数的绝对差异信息**。

### 6.3 可用信号

**全部是召回阶段已产生的数据，零额外计算**：

| 信号 | 来源 | 说明 |
|------|------|------|
| 图像路 cosine score | FAISS 搜索返回值 | 不在此路结果中的候选，该分数为 0 |
| BM25 score | rank_bm25 返回值 | 同上 |
| 语义路 cosine score | FAISS 搜索返回值 | 同上 |

### 6.4 分数归一化

三路分数尺度不同（cosine 在 [-1, 1]，BM25 可能 0~30+），需归一化到同一尺度。

对每路使用 min-max 归一化，映射到 [0, 1]：

```
norm_score(d) = (score(d) - min) / (max - min)
```

只在**当前这批召回结果内**做归一化，不需要全局统计。若某路只有一个结果，该路 norm_score 设为 1.0。

### 6.5 动态加权

用户可能只传图片没传标签（MVP 阶段常见），此时只有图像路有结果。

**按实际有效路数动态调权**：

| 场景 | 有效路 | 基础权重 | 归一化后实际权重 |
|------|--------|----------|-----------------|
| 图片 + 标签 | 三路 | 图像 0.50, BM25 0.25, 语义 0.25 | 0.50, 0.25, 0.25 |
| 只有图片 | 仅图像路 | 图像 0.50 | 1.00 |
| 只有标签 | BM25 + 语义 | BM25 0.25, 语义 0.25 | 0.50, 0.50 |

归一化规则：`有效权重 = 该路基础权重 / Σ所有有效路基础权重`

### 6.6 融合打分

对合并去重后的每个候选：

```
coarse_score(d) = Σ (有效权重_i × norm_score_i(d))
```

若某候选只出现在一路中，其他路的 norm_score 为 0。按 coarse_score 降序取 Top-100。

### 6.7 粗排小结

| 特性 | 说明 |
|------|------|
| 速度 | numpy 批量计算 500 条，< 5ms |
| 召回安全性 | 无硬淘汰规则，纯分数排序截断 |
| 信号来源 | 全部复用召回阶段的已有分数 |
| 降级行为 | 只有一路有结果时，该路权重自动升至 100% |
| 留给精排的 | 可能包含"向量相似但语义不合理"的噪声 |


## 7. 精排

### 7.1 定位

- **输入**：粗排输出的 ~100 候选
- **输出**：~20 最终结果，附各维度分数
- **时间预算**：< 200ms
- **核心目标**：排准，可解释

### 7.2 精排要回答的核心问题

> 「脑补这个模特穿上 query 的衣服，整体看起来协调吗？」

这不是简单的相似度，而是**跨维度的兼容性评估**。需要拆解为独立维度分别打分，再加权合成。

### 7.3 与粗排的本质区别

| | 粗排 | 精排 |
|--|------|------|
| 信号 | 召回的粗粒度整体分数 | 逐字段、逐维度的细粒度匹配 |
| 匹配方式 | 对称（query 向量 vs candidate 向量） | **非对称**（query 服装字段 ↔ candidate 模特/场景字段） |
| 计算 | 零推理（复用已有分数） | 有少量推理（query 侧字段 encode） |
| 可解释性 | 弱（一个总分） | 强（每个维度有独立分数） |

### 7.4 非对称匹配设计

query 侧描述的是服装，candidate 侧描述的是模特+场景。两侧标签**不应对称比较**，而是按业务含义交叉匹配：

| Query 侧字段 | ↔ | Candidate 侧字段 | 匹配含义 |
|-------------|---|------------------|----------|
| 服装氛围 | ↔ | 整体氛围 | 衣服调性和照片整体感觉一致 |
| 上装风格 | ↔ | 姿势 + 妆容 | 衣服设计风格和模特表现力匹配 |
| 目标人群 | ↔ | 脸部特征 + 妆容特征 | 衣服面向人群和模特气质吻合 |
| 整体氛围 | ↔ | 场景 + 主题风格 | 照片环境和衣服氛围协调 |
| 上装剪裁类型 | ↔ | 身材 | 衣服剪裁适合模特体型 |
| 特定活动 | ↔ | 特定活动 | 活动主题一致 |

**关键洞察：candidate 图上原本穿的衣服标签不重要（会被换掉），精排不比较两侧的服装标签。**

### 7.5 精排四个维度

```
精排总分 = w1·身体兼容 + w2·风格匹配 + w3·场景匹配 + w4·视觉匹配
```

#### 维度 1：身体兼容 (score_body)

**含义**：服装剪裁是否适合模特体型。

**匹配逻辑**：基于预定义的兼容性矩阵查表。

剪裁 × 身材兼容性矩阵：

|              | Slim | Standard | Plus_Size | Muscular |
|--------------|------|----------|-----------|----------|
| Skin-tight   | 1.0  | 0.8      | 0.4       | 0.7      |
| Slim fit     | 0.9  | 1.0      | 0.6       | 0.8      |
| Loose fit    | 0.7  | 0.9      | 1.0       | 0.9      |
| Oversized    | 0.6  | 0.8      | 1.0       | 0.8      |

- 1.0 = 非常适合，0.4 = 不太合适但不是硬淘汰
- 矩阵可由业务侧调整，存为配置文件
- query 未指定剪裁或 candidate 无身材标签时，返回默认值 0.7（中性）
- **计算成本**：O(1) 查表

#### 维度 2：风格匹配 (score_style)

**含义**：服装的风格气质和模特的表现力是否协调。**最重要的维度。**

**匹配逻辑**：逐字段 embedding 余弦相似度取平均。

匹配对（非对称）：
- query「服装氛围」 ↔ candidate「整体氛围」
- query「上装风格」 ↔ candidate「姿势」+「妆容」
- query「目标人群」 ↔ candidate「脸部特征」+「妆容特征」

取所有非空匹配对的 cosine 均值。空字段跳过，不参与计算。

**计算成本**：candidate 侧 embedding 入库时预计算（field_embeddings/ 目录），在线只需 encode query 侧 3 个字段（一次 batch ~20ms），然后查表做 cosine（< 1ms）。

#### 维度 3：场景匹配 (score_scene)

**含义**：服装适合的场景和候选图的拍摄环境是否协调。

**匹配逻辑**：同风格匹配，逐字段 cosine 取平均。

匹配对（非对称）：
- query「服装氛围」 ↔ candidate「场景」
- query「整体氛围」 ↔ candidate「主题风格」
- query「特定活动」 ↔ candidate「特定活动」（两侧都非空时）

同样非空才参与计算。

**计算成本**：同上，candidate 侧预计算，query 侧复用维度 2 已 encode 的字段。

#### 维度 4：视觉匹配 (score_visual)

**含义**：两张图片在视觉特征空间的整体接近程度。

**匹配逻辑**：query 图像 embedding 与 candidate 图像 embedding 的精确余弦值。

**与粗排的区别**：粗排用的是 FAISS ANN 近似值经过归一化和加权后的结果；精排直接用精确 cosine，作为四个维度之一，不再是主导信号。

**计算成本**：向量已有，numpy 点积，< 0.1ms。

### 7.6 权重

| 维度 | 初始权重 | 理由 |
|------|----------|------|
| 身体兼容 | **0.15** | 基础条件，区分度较低（大部分剪裁对大部分体型都可接受） |
| 风格匹配 | **0.30** | 最核心维度，"穿上好不好看"主要取决于风格气质是否一致 |
| 场景匹配 | **0.25** | 场景和氛围是用户直观感受的重要部分 |
| 视觉匹配 | **0.30** | 图像级的整体视觉感受，作为兜底信号 |

权重调优方向：
- **短期**：人工看 case 调权重
- **中期**：收集 Gradio UI 上的用户选择偏好
- **长期**：learning to rank

### 7.7 计算成本控制

精排最关键的设计决策：**candidate 侧的字段级 embedding 在入库时预计算好，在线不计算**。

| 操作 | 在线/离线 | 数量 | 耗时 |
|------|----------|------|------|
| candidate 字段级 embedding | **离线**（入库时） | 一次性 | 在线零成本 |
| query 字段级 embedding | **在线** | 3~5 个字段 batch | ~20ms |
| cosine 计算 | 在线 | 100 × 3~4 个匹配对 | < 1ms |
| 兼容性矩阵查表 | 在线 | 100 × 1 次 | < 0.1ms |
| **精排总计** | | | **~25ms** |

### 7.8 标签缺失的降级策略

| 情况 | 处理 |
|------|------|
| candidate 缺某字段 embedding | 该匹配对跳过，mean 只算有值的 |
| candidate 完全没有标签 | 风格和场景维度返回默认分数 0.5 |
| query 缺某字段 | 该维度跳过，不参与加权 |
| 只有图片没标签 | 精排退化为纯 score_visual，其余维度不计算 |
| 全部候选都缺标签 | 精排等价于粗排的图像排序（系统仍可用） |

### 7.9 精排输出

最终为每个结果附带各维度分数，用于调试和前端展示：

```json
{
  "id": 42,
  "image_path": "images/042.jpg",
  "final_score": 0.78,
  "score_detail": {
    "body": 0.90,
    "style": 0.75,
    "scene": 0.82,
    "visual": 0.68
  }
}
```

### 7.10 后续可扩展的精排信号

当前精排只用了"轻量"信号。后续可按需增加更重的信号：

| 信号 | 计算方式 | 成本 | 适合时机 |
|------|----------|------|----------|
| 跨模态匹配 | cos(query_image_emb, candidate_tag_text_emb) | 低（已有向量） | Phase 2 |
| 标签覆盖度 | candidate 有值字段数 / 总字段数 | 极低 | Phase 2 |
| VLM rerank | VLM 同时看两张图，判断"穿上好不好看" | 很高（~2s/条） | Phase 4 |
| 用户行为 | 历史点击/选用率 | 需数据积累 | 长期 |


## 8. 入库预计算清单

| 预计算项 | 文件 | 用于阶段 |
|----------|------|----------|
| fashionSigLIP 图像向量 | image_embeddings.npy | 召回 + 精排(视觉) |
| 全量标签拼接文本向量 | text_embeddings.npy | 召回(语义路) |
| FAISS 图像索引 | faiss_image.index | 召回 |
| FAISS 文本索引 | faiss_text.index | 召回 |
| 字段「整体氛围」embedding | field_embeddings/整体氛围.npy | 精排(风格) |
| 字段「姿势+妆容」embedding | field_embeddings/姿势_妆容.npy | 精排(风格) |
| 字段「脸部特征+妆容特征」embedding | field_embeddings/脸部特征_妆容特征.npy | 精排(风格) |
| 字段「场景」embedding | field_embeddings/场景.npy | 精排(场景) |
| 字段「主题风格」embedding | field_embeddings/主题风格.npy | 精排(场景) |
| 字段「特定活动」embedding | field_embeddings/特定活动.npy | 精排(场景) |

字段 embedding 统一使用 fashionSigLIP text encoder，不引入额外模型。
若某条记录该字段为空，对应位置存零向量。


## 9. 延迟预估

| 阶段 | 操作 | GPU | CPU |
|------|------|-----|-----|
| 枚举过滤 | 遍历 metadata | 5ms | 5ms |
| Query encode image | fashionSigLIP | 30ms | 300ms |
| Query encode text（召回） | fashionSigLIP text | 10ms | 50ms |
| FAISS 图像路 | top-200 | 5ms | 10ms |
| FAISS 语义路 | top-200 | 5ms | 10ms |
| BM25 | get_scores + sort | 10ms | 10ms |
| 粗排 | 归一化 + 加权 | 2ms | 2ms |
| Query encode fields（精排） | 3~5 字段 batch | 10ms | 50ms |
| 精排打分 | cosine + 矩阵 | 1ms | 1ms |
| **总计** | | **~80ms** | **~440ms** |


## 10. 项目结构

```
conrain/
├── docs/
│   ├── retrieval-prd.md
│   ├── tag.md
│   └── retrieval-technical-design.md
├── retrieval/
│   ├── pyproject.toml                    # uv 项目配置
│   ├── main.py                           # Gradio UI 入口
│   ├── config.py                         # 配置
│   ├── core/
│   │   ├── encoder.py                    # fashionSigLIP 编码封装
│   │   ├── retriever.py                  # 召回 + 粗排 + 精排主流程
│   │   ├── coarse_ranker.py              # 粗排：分数归一化 + 动态加权
│   │   ├── fine_ranker.py                # 精排：逐维度匹配 + 兼容性矩阵
│   │   ├── bm25.py                       # BM25 关键词检索
│   │   └── tag_utils.py                  # 标签 flatten / 枚举提取
│   ├── indexing/
│   │   ├── build_index.py                # 生成 embedding + FAISS 索引 + 字段 embedding
│   │   └── vlm_tagger.py                # VLM 打标（可选）
│   ├── data/                             # 数据目录（gitignore）
│   │   ├── images/
│   │   ├── metadata.jsonl
│   │   ├── image_embeddings.npy
│   │   ├── text_embeddings.npy
│   │   ├── field_embeddings/
│   │   ├── faiss_image.index
│   │   └── faiss_text.index
│   └── scripts/
│       └── prepare_sample_data.py
```


## 11. 依赖

```toml
# pyproject.toml dependencies
[project]
dependencies = [
    "gradio",
    "open-clip-torch",
    "torch",
    "faiss-cpu",
    "rank-bm25",
    "numpy",
    "Pillow",
    "jieba",
]
```


## 12. 快速验证计划

### Phase 1：图像检索 + 粗排

- [ ] encoder.py（fashionSigLIP 加载和推理）
- [ ] 少量样例数据（~100 张图 + metadata.jsonl）
- [ ] build_index.py（图像 embedding + FAISS 索引）
- [ ] retriever.py（仅图像路召回 + 粗排单路模式）
- [ ] Gradio 界面：上传图片 → 返回相似图片

### Phase 2：三路召回 + 完整粗排

- [ ] BM25 索引 + 检索
- [ ] 文本语义向量检索
- [ ] 枚举字段过滤
- [ ] 粗排的三路分数归一化 + 动态加权
- [ ] Gradio 界面：图片 + 多标签组合输入

### Phase 3：精排

- [ ] build_index 增加字段级 embedding 预计算
- [ ] fine_ranker.py 兼容性矩阵
- [ ] fine_ranker.py 逐维度匹配打分
- [ ] 精排结果附各维度分数
- [ ] Gradio 展示分数详情

### Phase 4：VLM 打标 + 迭代

- [ ] 接入 VLM 批量打标
- [ ] 对比标签质量对检索效果的影响
- [ ] 收集用户反馈调优权重
