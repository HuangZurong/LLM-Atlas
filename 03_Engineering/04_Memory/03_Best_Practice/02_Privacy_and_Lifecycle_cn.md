# 隐私、生命周期与被遗忘权

*前置要求：[01_Memory_Architecture_Patterns.md](01_Memory_Architecture_Patterns.md)。*

---

LLM 应用中的记忆会带来显著的隐私与合规挑战。本文覆盖负责任地管理记忆所需的工程模式。

## 1. GDPR 问题

在 GDPR（以及类似的 CCPA、PIPL 等法规）下，用户拥有**删除权**。如果你的 LLM 将用户数据存成记忆，就必须能够：
1. **列出**与某位用户相关的全部记忆。
2. 按请求**删除**特定记忆。
3. **导出**全部已存储数据（数据可携带性）。

### 工程含义：
每一条记忆都必须带有如下标签：
```json
{
  "user_id": "user_123",
  "created_at": "2026-02-24T10:00:00Z",
  "source": "conversation",
  "content": "User prefers Python.",
  "ttl": 7776000
}
```

## 2. 记忆生命周期管理

### 2.1 四个阶段
```
CREATE → ACTIVE → DECAY → DELETE
```

| 阶段 | 触发条件 | 动作 |
| :--- | :--- | :--- |
| **Create** | 实体抽取或用户显式陈述 | 带时间戳与 TTL 存储 |
| **Active** | 记忆较新且经常被检索 | 在检索评分中给予完整权重 |
| **Decay** | 记忆年龄超过阈值（如 90 天） | 降低检索权重；标记为待审查 |
| **Delete** | TTL 到期、用户请求删除，或检测到矛盾 | 从所有存储中硬删除（数据库 + 向量索引） |

### 2.2 矛盾解决
当新信息与已存记忆冲突时：
- **Session 1**: "Budget is $10K"
- **Session 5**: "Budget is $50K"

**策略**：始终优先采用最新的记忆。将旧记忆标记为 `superseded`，随后最终清除。

## 3. 什么该记住，什么该忘记

### 应当记住：
- 明确的用户偏好（“我喜欢简洁的回答”）
- 项目决策（“我们选择 Qdrant 作为向量存储”）
- 事实性上下文（“用户是 Acme Corp 的 ML 工程师”）

### 绝不应记住：
- 密码、API 密钥或凭据
- 与任务无关的个人敏感信息（社会安全号、信用卡号）
- 情绪状态或健康信息（除非应用专门为此设计）

### 工程模式：存储前做 PII 过滤
```python
def sanitize_before_storage(text: str) -> str:
    # Use regex or Presidio to strip PII
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
    text = re.sub(r'\b\d{16}\b', '[CARD_REDACTED]', text)
    return text
```

## 4. 多租户与访问控制

在企业环境中，记忆必须被**划定作用域**：

| 作用域 | 可见性 | 示例 |
| :--- | :--- | :--- |
| **User** | 仅该用户本人 | 个人偏好 |
| **Project** | 项目全部成员 | 架构决策 |
| **Organization** | 组织内所有用户 | 公司政策、术语表 |
| **Global** | 所有用户（只读） | 产品文档 |

**规则**：任何记忆查询都必须始终包含用户的作用域。绝不可泄漏跨租户的记忆。

## 5. 审计轨迹

每一次记忆操作（create、read、update、delete）都应被记录：
```json
{
  "action": "DELETE",
  "memory_id": "mem_abc123",
  "user_id": "user_123",
  "reason": "user_request",
  "timestamp": "2026-02-24T12:00:00Z"
}
```
这对合规审计与问题排查都至关重要。
