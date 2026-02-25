# Privacy, Lifecycle & the Right to Forget

*Prerequisite: [01_Memory_Architecture_Patterns.md](01_Memory_Architecture_Patterns.md).*

---

Memory in LLM applications introduces significant privacy and compliance challenges. This guide covers the engineering patterns required for responsible memory management.

## 1. The GDPR Problem

Under GDPR (and similar regulations like CCPA, PIPL), users have the **Right to Erasure**. If your LLM stores user data as memories, you must be able to:
1. **List** all memories associated with a user.
2. **Delete** specific memories on request.
3. **Export** all stored data (data portability).

### Engineering Implication:
Every memory entry must be tagged with:
```json
{
  "user_id": "user_123",
  "created_at": "2026-02-24T10:00:00Z",
  "source": "conversation",
  "content": "User prefers Python.",
  "ttl": 7776000
}
```

## 2. Memory Lifecycle Management

### 2.1 The Four Stages
```
CREATE → ACTIVE → DECAY → DELETE
```

| Stage | Trigger | Action |
| :--- | :--- | :--- |
| **Create** | Entity extraction or explicit user statement | Store with timestamp and TTL |
| **Active** | Memory is recent and frequently retrieved | Full weight in retrieval scoring |
| **Decay** | Memory age exceeds threshold (e.g., 90 days) | Reduce retrieval weight; flag for review |
| **Delete** | TTL expires, user requests deletion, or contradiction detected | Hard delete from all stores (DB + vector index) |

### 2.2 Contradiction Resolution
When new information conflicts with stored memory:
- **Session 1**: "Budget is $10K"
- **Session 5**: "Budget is $50K"

**Strategy**: Always prefer the most recent memory. Mark the old one as `superseded` and eventually purge it.

## 3. What to Remember vs. What to Forget

### Remember:
- Explicit user preferences ("I prefer concise answers")
- Project decisions ("We chose Qdrant for vector storage")
- Factual context ("User is an ML engineer at Acme Corp")

### Never Remember:
- Passwords, API keys, or credentials
- PII that isn't necessary for the task (SSN, credit card numbers)
- Emotional states or health information (unless the app is specifically designed for this)

### Engineering Pattern: PII Filter Before Storage
```python
def sanitize_before_storage(text: str) -> str:
    # Use regex or Presidio to strip PII
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
    text = re.sub(r'\b\d{16}\b', '[CARD_REDACTED]', text)
    return text
```

## 4. Multi-Tenancy & Access Control

In enterprise environments, memories must be **scoped**:

| Scope | Visibility | Example |
| :--- | :--- | :--- |
| **User** | Only the individual user | Personal preferences |
| **Project** | All members of a project | Architecture decisions |
| **Organization** | All users in the org | Company policies, glossary |
| **Global** | All users (read-only) | Product documentation |

**Rule**: A memory query must always include the user's scope. Never leak cross-tenant memories.

## 5. Audit Trail

Every memory operation (create, read, update, delete) should be logged:
```json
{
  "action": "DELETE",
  "memory_id": "mem_abc123",
  "user_id": "user_123",
  "reason": "user_request",
  "timestamp": "2026-02-24T12:00:00Z"
}
```
This is essential for compliance audits and debugging.
