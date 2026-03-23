# Claude Code Skills 开发指南

## 目录

1. [核心概念](#核心概念)
2. [Skill结构](#skill结构)
3. [开发流程](#开发流程)
4. [最佳实践](#最佳实践)
5. [常见模式](#常见模式)
6. [测试与迭代](#测试与迭代)
7. [案例研究](#案例研究)

---

## 核心概念

### 什么是Agent Skill？

Agent Skill是**指导AI代理行为的文档**，而非代码。它告诉AI：
- **何时**使用某个能力
- **如何**正确使用
- **什么**是质量标准
- **如何**验证成功

### Skill vs 其他扩展功能

| 功能 | 核心作用 | 加载方式 | 上下文成本 |
|------|---------|---------|-----------|
| **Skill** | 可重用知识/工作流 | 按需加载 | 低（仅描述预加载） |
| CLAUDE.md | 全局持久上下文 | 会话开始全量加载 | 高 |
| MCP | 外部服务连接 | 工具定义加载 | 中 |
| Subagent | 隔离执行上下文 | 独立上下文 | 无（隔离） |

**关键区别**：
- Skill ≠ MCP工具（Skill指导如何使用MCP）
- Skill ≠ 代码（Skill是行为指南）
- Skill ≠ API文档（Skill包含上下文和最佳实践）

### 何时创建Skill？

✅ **应该创建**：
- 重复执行的标准化任务
- 需要参考文档的知识库
- 需要特定步骤的工作流
- 跨项目可重用的模式

❌ **不应创建**：
- 一次性任务（直接在对话中说明）
- 简单命令（CLAUDE.md规则即可）
- 纯技术接口（用MCP）

---

## Skill结构

### 文件组织

```
my-skill/
├── SKILL.md                    # 主文件（必需）
│   ├── YAML frontmatter        # 元数据
│   └── Markdown内容            # 指令、示例、检查清单
│
└── reference/                  # 参考文档（可选）
    ├── advanced-features.md    # 详细特性
    ├── examples.md             # 更多示例
    └── troubleshooting.md      # 故障排除
```

### YAML Frontmatter

**必需字段**：

```yaml
---
name: skill-name                # 最多64字符，小写字母/数字/连字符
description: "..."              # 最多1024字符，包含触发条件
---
```

**命名规范**：
- ✅ 动名词形式：`processing-pdfs`、`analyzing-data`
- ✅ 行动导向：`process-pdfs`、`analyze-data`
- ❌ 模糊名称：`helper`、`utils`
- ❌ 保留字：`anthropic-*`、`claude-*`

**描述规范**：
- 使用第三人称
- 包含功能描述
- 包含触发条件
- 具体明确

**好的示例**：
```yaml
---
name: render-visualization
description: "Create charts, tables, and reports. Use when visualizing
data or generating formatted outputs. Trigger when user wants to
visualize query results, create formatted tables, or generate reports."
---
```

**不好的示例**：
```yaml
---
name: visualization-helper
description: "Helps with visualization tasks"  # 太模糊
---
```

### 内容结构

推荐的SKILL.md结构（< 500行）：

```markdown
---
name: my-skill
description: "..."
---

# Skill Title

## Overview
简要说明技能的功能和价值（2-3句话）

## When to Use
具体的触发场景列表

## Quick Start
最常用的最小示例

## Common Workflows
常见使用流程（带步骤）

## Requirements / Standards
输出质量要求

## Verification Checklist
验证清单

## Best Practices
最佳实践建议

## Anti-Patterns
常见错误和陷阱

## Advanced Features
指向reference/的链接
```

---

## 开发流程

### 第一步：识别需求

**评估标准**：
1. 是否重复出现？
2. 是否需要特定知识？
3. 是否跨项目可用？
4. 是否有明确的工作流？

**示例观察**：
```
观察：用户多次要求生成报告，每次都需要：
- 相同的格式要求
- 相同的数据验证步骤
- 相同的输出结构

结论：创建 report-generation skill
```

### 第二步：创建最小版本

**原则**：解决当前问题，不过度设计

```markdown
---
name: report-generation
description: "Generate structured reports from data. Use when user needs formatted reports with tables and charts."
---

# Report Generation

## Quick Start
```python
generate_report(data, format="markdown")
```

## Requirements
- Include summary section
- Use consistent formatting
- Validate data before rendering
```

### 第三步：测试验证

**测试清单**：
- [ ] 描述触发是否正确？
- [ ] 内容是否清晰易懂？
- [ ] 示例是否可直接使用？
- [ ] 检查清单是否完整？

**多模型测试**：
- **Haiku**：内容是否足够详细？
- **Sonnet**：内容是否高效清晰？
- **Opus**：内容是否避免过度解释？

### 第四步：迭代优化

**基于使用反馈**：
1. 观察Claude如何使用skill
2. 识别困惑或错误点
3. 简化冗余内容
4. 补充缺失信息
5. 重新测试

---

## 最佳实践

### 1. 简洁是关键

❌ **过度解释**（假设Claude不懂）：
```markdown
## PDF Processing

PDF (Portable Document Format) is a file format developed by Adobe.
It's widely used for documents. To extract text from PDFs, you need
a library. We recommend pdfplumber because it's easy to use...
```

✅ **简洁直接**（假设Claude知道PDF）：
```markdown
## PDF Processing

Extract text using pdfplumber:
```python
import pdfplumber
with pdfplumber.open("file.pdf") as pdf:
    text = pdf.pages[0].extract_text()
```
```

### 2. 渐进式披露

**原则**：主文件精简，详情按需加载

```markdown
## Advanced Features

**OCR支持**: 参见 `reference/ocr-processing.md`
**批量处理**: 参见 `reference/batch-operations.md`
**错误处理**: 参见 `reference/error-handling.md`
```

**好处**：
- 降低初始上下文成本
- 保持主文件清晰
- 按需加载详情

### 3. 设置适当的自由度

**高自由度**（文本说明）：
```markdown
## Code Review Process
1. Analyze code structure
2. Check for potential bugs
3. Suggest improvements
4. Verify project conventions
```

**低自由度**（精确脚本）：
```markdown
## Database Migration
Run exactly this script:
```bash
python scripts/migrate.py --verify --backup
```
Do not modify flags or parameters.
```

### 4. 验证清单模式

**Before/After检查**：

```markdown
## Verification Checklist

Before rendering:
- [ ] Data is array of objects
- [ ] Required fields present
- [ ] Data types consistent

After rendering:
- [ ] result.success === true
- [ ] Output format correct
- [ ] No errors in logs
```

### 5. 避免时间敏感信息

❌ **会过时的信息**：
```markdown
If executing before August 2025, use old API.
After August 2025, use new API.
```

✅ **版本标记**：
```markdown
## Current Method
Use v2 API: `api.example.com/v2/endpoint`

## Legacy
<details>
<summary>v1 API (deprecated 2025-08)</summary>
Use `api.example.com/v1/endpoint` (no longer supported)
</details>
```

---

## 常见模式

### 模式1：模板模式

**适用场景**：需要特定输出格式

```markdown
## Report Structure

Use this exact template:
```markdown
# [Analysis Title]

## Executive Summary
[One paragraph overview]

## Key Findings
- Finding 1 with supporting data
- Finding 2 with supporting data

## Recommendations
1. Specific actionable recommendation
2. Specific actionable recommendation
```
```

### 模式2：工作流模式

**适用场景**：多步骤任务

```markdown
## PDF Form Filling Workflow

Copy this checklist and track progress:
```
Task Progress:
- [ ] Step 1: Analyze form
- [ ] Step 2: Create field mapping
- [ ] Step 3: Validate mapping
- [ ] Step 4: Fill form
- [ ] Step 5: Verify output
```

**Step 1: Analyze Form**
Run: `python scripts/analyze_form.py input.pdf`

**Step 2: Create Field Mapping**
Edit `fields.json` to add values for each field.
...
```

### 模式3：条件工作流

**适用场景**：决策分支

```markdown
## Document Modification

1. Determine modification type:
   **Creating new content?** → Follow "Creation Workflow"
   **Editing existing content?** → Follow "Edit Workflow"

2. Creation Workflow:
   - Use docx-js library
   - Build document from scratch
   - Export as .docx

3. Edit Workflow:
   - Unpack existing document
   - Modify XML directly
   - Validate after each change
   - Repack when done
```

### 模式4：Skill + MCP组合

**Skill指导** + **MCP执行**：

```markdown
## Quick Start

Use the `render_chart` MCP tool:
```json
{
  "chartType": "column",
  "axis": [...],
  "data": [...]
}
```

See `reference/chart-types.md` for available chart types.
```

---

## 测试与迭代

### 创建评估

**评估结构**：
```json
{
  "skills": ["my-skill"],
  "query": "测试问题",
  "files": ["test-files/input.txt"],
  "expected_behavior": [
    "预期行为1",
    "预期行为2",
    "预期行为3"
  ]
}
```

### 与Claude协作迭代

**迭代流程**：

```
┌─────────────────┐
│  Claude A       │  ← 帮助创建/改进skill
│  (专家)         │
└────────┬────────┘
         │ 创建/改进
         ↓
┌─────────────────┐
│  Skill          │
│  (SKILL.md)     │
└────────┬────────┘
         │ 被使用
         ↓
┌─────────────────┐
│  Claude B       │  ← 使用skill执行任务
│  (工作者)       │
└─────────────────┘
```

**具体步骤**：

1. **完成任务**：与Claude A一起完成任务，注意重复提供的上下文
2. **识别模式**：提取可重用的知识/工作流
3. **请求创建**："创建一个skill来捕获我们使用的模式"
4. **审查简洁性**："删除关于X的解释，Claude已经知道了"
5. **优化结构**："将详细内容移到reference/目录"
6. **测试验证**：用Claude B测试实际效果
7. **迭代改进**：基于观察结果调整

### 观察导航模式

**关键观察点**：
- Claude是否按预期路径读取文件？
- Claude是否遗漏重要引用？
- Claude是否过度依赖某些部分？
- Claude是否忽略某些内容？

**根据观察调整**：
- 引用未被跟随 → 使链接更明确
- 部分被重复读取 → 考虑移到主文件
- 内容被忽略 → 可能信号不足或不必要

---

## 案例研究

### 案例1：PDF处理Skill

**需求**：团队经常处理PDF文件，提取文本、表格，填充表单

**Skill结构**：
```
pdf-processing/
├── SKILL.md                 # 主文件（180行）
│   ├── Quick Start
│   ├── Text Extraction
│   ├── Table Extraction
│   └── Advanced Features → reference/
│
└── reference/
    ├── form-filling.md      # 表单填充详细指南
    ├── ocr-processing.md    # OCR处理
    └── batch-operations.md  # 批量操作
```

**YAML Frontmatter**：
```yaml
---
name: pdf-processing
description: "Extract text and tables from PDFs, fill forms, merge documents.
Use when processing PDF files or user mentions PDFs, forms, or document extraction."
---
```

**关键内容**：
```markdown
## Quick Start

Extract text with pdfplumber:
```python
import pdfplumber
with pdfplumber.open("file.pdf") as pdf:
    text = pdf.pages[0].extract_text()
```

## Table Extraction

For tables, use:
```python
tables = pdf.pages[0].extract_tables()
for table in tables:
    # table is list of lists
```

## Advanced Features

**Form Filling**: See `reference/form-filling.md`
**OCR for Scanned PDFs**: See `reference/ocr-processing.md`
**Batch Processing**: See `reference/batch-operations.md`
```

**成功因素**：
- 描述明确包含触发条件
- Quick Start立即可用
- 渐进式披露减少初始负载
- 测试验证所有路径可达

### 案例2：数据分析Skill

**需求**：标准化数据分析和可视化流程

**Skill结构**：
```
data-analysis/
├── SKILL.md                 # 主文件（220行）
│   ├── Workflow Overview
│   ├── Data Validation
│   ├── Analysis Patterns
│   └── Visualization Guide
│
└── reference/
    ├── statistical-methods.md
    ├── chart-selection.md
    └── report-templates.md
```

**工作流模式**：
```markdown
## Analysis Workflow

Follow this workflow for data analysis:

1. **Validate Data**
   ```python
   df.info()  # Check types and nulls
   df.describe()  # Summary statistics
   ```

2. **Clean Data**
   - Handle missing values
   - Remove duplicates
   - Fix data types

3. **Analyze**
   See `reference/statistical-methods.md` for common patterns

4. **Visualize**
   See `reference/chart-selection.md` for chart type guidance

5. **Report**
   Use templates from `reference/report-templates.md`
```

**验证清单**：
```markdown
## Verification Checklist

Before analysis:
- [ ] Data loaded correctly
- [ ] Data types verified
- [ ] Missing values identified

After analysis:
- [ ] Results make sense
- [ ] Outliers explained
- [ ] Conclusions supported by data

Before delivery:
- [ ] Visualizations clear
- [ ] Report formatted correctly
- [ ] Key insights highlighted
```

### 案例3：V5渲染服务Skill

**需求**：指导AI正确使用V5渲染服务

**Skill结构**：
```
render-visualization/
├── SKILL.md                 # 主文件（200行）
│   ├── Overview
│   ├── When to Use
│   ├── Quick Start (Chart/Table/Report)
│   ├── Common Workflows
│   ├── Verification Checklist
│   └── Advanced Features → reference/
│
└── reference/
    ├── percentage-handling.md
    ├── multi-metric.md
    ├── report-structure.md
    └── performance.md
```

**关键特点**：
- 清晰的触发条件描述
- 三种渲染类型的Quick Start
- Before/After验证清单
- 渐进式披露高级特性

**成功指标**：
- Claude能正确判断何时使用
- 配置错误率低
- 验证步骤被执行
- 输出质量符合要求

---

## 技巧与陷阱

### 技巧清单

✅ **始终做到**：
- [ ] 描述具体，包含触发条件
- [ ] 主文件 < 500行
- [ ] 使用一致的术语
- [ ] 提供可运行的示例
- [ ] 包含验证检查清单
- [ ] 使用Unix风格路径（正斜杠）
- [ ] 引用保持一级深度

✅ **优先考虑**：
- [ ] 简洁优于完整
- [ ] 示例优于解释
- [ ] 检查清单优于长篇说明
- [ ] 渐进披露优于一次性全量

### 常见陷阱

❌ **避免**：

1. **过度解释**：
   ```markdown
   # 不好的例子
   PDF (Portable Document Format) is a file format...
   ```

2. **模糊描述**：
   ```yaml
   description: "Helps with documents"  # 太模糊
   ```

3. **多个选项**：
   ```markdown
   # 不好的例子
   You can use pypdf, pdfplumber, PyMuPDF, pdf2image...
   ```

4. **Windows路径**：
   ```markdown
   # 不好的例子
   Run `scripts\helper.py`
   ```

5. **时间敏感信息**：
   ```markdown
   # 不好的例子
   Before August 2025, use old API...
   ```

6. **深层嵌套引用**：
   ```markdown
   # 不好的例子
   See advanced.md → details.md → implementation.md
   ```

---

## 资源与工具

### 官方文档
- [Claude Agent Skills Best Practices](https://platform.claude.com/docs/zh-CN/agents-and-tools/agent-skills/best-practices)
- [Skills Overview](https://platform.claude.com/docs/zh-CN/agents-and-tools/agent-skills/overview)

### 示例Skills
- V5 Render Service: `v5/skills/SKILL.md`
- PDF Processing: 参考 xlsx skill 示例

### 开发工具
- Claude Code CLI: 直接创建和管理skills
- 评估框架: 测试skill有效性

---

## 总结

### Skill开发核心原则

1. **简洁**：每个token都要有价值
2. **渐进**：主文件精简，详情按需
3. **验证**：提供检查清单确保质量
4. **迭代**：基于实际使用持续改进

### 开发检查清单

**创建前**：
- [ ] 确认需求是重复的
- [ ] 确认需要特定知识/工作流
- [ ] 确认不是简单命令或纯接口

**创建中**：
- [ ] 描述具体且包含触发条件
- [ ] 主文件 < 500行
- [ ] 示例可直接使用
- [ ] 包含验证清单
- [ ] 使用渐进式披露

**创建后**：
- [ ] 多模型测试（Haiku/Sonnet/Opus）
- [ ] 真实场景验证
- [ ] 观察Claude导航模式
- [ ] 收集团队反馈
- [ ] 持续迭代改进

---

**记住**：好的Skill不是写出来的，而是迭代出来的。从最小版本开始，在实际使用中观察、学习、改进。
