# AcmeCorp API 参考指南 v2.3

## 概览

AcmeCorp API 提供了以编程方式访问我们电商平台的能力。本文档涵盖认证、核心端点、速率限制和最佳实践。

**基础 URL**：`https://api.acmecorp.com/v2`

**认证方式**：所有请求都要求在 `X-API-Key` 请求头中携带 API key。

---

## 认证

### 获取你的 API 密钥

1. 登录 AcmeCorp Dashboard
2. 进入 Settings > API Keys
3. 点击 “Generate New Key”
4. 妥善保存该 key — 它只会显示一次

**重要**：API key 绝不能共享，也不能提交到版本控制中。请改用环境变量。

### 发起已认证请求

```bash
curl -H "X-API-Key: your_api_key_here" \
     https://api.acmecorp.com/v2/orders
```

无效或缺失的 API key 会返回 `401 Unauthorized`。

---

## 速率限制

| 套餐 | 请求数/分钟 | 请求数/天 |
|------|-------------|-----------|
| Free | 60 | 1,000 |
| Pro | 600 | 10,000 |
| Enterprise | 6,000 | Unlimited |

当你超过速率限制时，API 会返回 `429 Too Many Requests`，并通过 `Retry-After` 请求头告知需要等待的秒数。

**最佳实践**：在客户端代码中实现指数退避。先从 1 秒开始，每次重试翻倍，最大 60 秒。

---

## 核心端点

### 订单

#### 获取订单列表

**端点**：`GET /orders`

**参数**：
- `status`（string，可选）：按状态过滤。可选值：`pending`、`processing`、`shipped`、`delivered`、`cancelled`
- `limit`（integer，可选）：返回结果数（默认：20，最大：100）
- `offset`（integer，可选）：分页偏移量（默认：0）

**响应**：
```json
{
  "orders": [
    {
      "order_id": "AC-7823",
      "customer_id": "CUST-4521",
      "status": "shipped",
      "total": 89.00,
      "items": [
        {
          "product_id": "XK-500",
          "name": "Wireless Keyboard",
          "quantity": 1,
          "price": 89.00
        }
      ],
      "created_at": "2025-03-10T14:30:00Z",
      "shipped_at": "2025-03-12T09:15:00Z"
    }
  ],
  "pagination": {
    "total": 45,
    "limit": 20,
    "offset": 0,
    "has_more": true
  }
}
```

#### 获取单个订单

**端点**：`GET /orders/{order_id}`

**响应**：单个订单对象，包含完整详情和物流跟踪信息。

#### 创建订单

**端点**：`POST /orders`

**请求体**：
```json
{
  "customer_id": "CUST-4521",
  "items": [
    {
      "product_id": "XK-500",
      "quantity": 1
    }
  ],
  "shipping_address": {
    "name": "Jane Doe",
    "street": "123 Main St",
    "city": "Springfield",
    "state": "IL",
    "zip": "62701",
    "country": "US"
  }
}
```

**响应**：返回 `201 Created`，并附带创建完成的订单对象。

#### 更新订单状态

**端点**：`PATCH /orders/{order_id}/status`

**请求体**：
```json
{
  "status": "shipped",
  "tracking_number": "1Z999AA10123456784"
}
```

**说明**：只有管理员账号可以更新订单状态。普通 API key 会收到 `403 Forbidden`。

#### 取消订单

**端点**：`POST /orders/{order_id}/cancel`

**条件**：
- 只有状态为 `pending` 或 `processing` 的订单才可以取消
- 已发货订单需要改走退款申请流程

**响应**：`200 OK`，并返回更新后的订单对象。

---

### 商品

#### 获取商品列表

**端点**：`GET /products`

**参数**：
- `category`（string，可选）：按类别过滤
- `min_price`（number，可选）：最低价格
- `max_price`（number，可选）：最高价格
- `in_stock`（boolean，可选）：仅显示有库存商品

#### 获取单个商品

**端点**：`GET /products/{product_id}`

返回完整商品详情，包括库存水平、变体和图片。

---

### 客户

#### 获取客户

**端点**：`GET /customers/{customer_id}`

**响应**：
```json
{
  "customer_id": "CUST-4521",
  "name": "Jane Doe",
  "email": "jane.doe@email.com",
  "membership_tier": "premium",
  "member_since": "2022-03-15",
  "total_orders": 12,
  "total_spent": 1547.89
}
```

#### 更新客户

**端点**：`PATCH /customers/{customer_id}`

可更新字段：`email`、`phone`、`address`、`preferences`

---

## Webhook

当你的账户中发生事件时，AcmeCorp 可以实时推送通知。

### 支持的事件

- `order.created`
- `order.shipped`
- `order.delivered`
- `order.cancelled`
- `refund.processed`
- `customer.updated`

### 配置 Webhook

1. 进入 Dashboard > Settings > Webhooks
2. 输入你的 endpoint URL（必须为 HTTPS）
3. 选择要订阅的事件
4. 验证 endpoint（我们会发送一次测试请求）

### Webhook 负载

```json
{
  "event": "order.shipped",
  "timestamp": "2025-03-12T09:15:00Z",
  "data": {
    "order_id": "AC-7823",
    "tracking_number": "1Z999AA10123456784",
    "carrier": "FedEx"
  },
  "signature": "sha256=..."
}
```

**安全性**：始终使用你的 webhook secret 验证签名。

---

## 错误处理

所有错误都遵循以下格式：

```json
{
  "error": {
    "code": "INVALID_ORDER_ID",
    "message": "Order AC-9999 not found",
    "details": {
      "order_id": "AC-9999"
    }
  }
}
```

### 常见错误码

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | API key 无效或缺失 |
| `FORBIDDEN` | 403 | 权限不足 |
| `NOT_FOUND` | 404 | 资源不存在 |
| `VALIDATION_ERROR` | 400 | 请求参数无效 |
| `RATE_LIMIT_EXCEEDED` | 429 | 请求过多 |
| `INTERNAL_ERROR` | 500 | 服务端错误（稍后重试） |

---

## 最佳实践

### 1. 幂等性

对于关键操作（创建订单、处理支付），请在请求头中带上 `Idempotency-Key`。如果以相同 key 重试请求，我们会返回原始响应，而不会创建重复记录。

```bash
curl -X POST \
     -H "X-API-Key: your_key" \
     -H "Idempotency-Key: unique_key_123" \
     -d '{"customer_id":"CUST-4521",...}' \
     https://api.acmecorp.com/v2/orders
```

### 2. 分页

对所有列表端点都要使用分页。不要假设所有结果都能装进一次响应里。

**示例**：
```python
offset = 0
all_orders = []
while True:
    response = requests.get(f"/orders?offset={offset}&limit=100")
    orders = response.json()["orders"]
    all_orders.extend(orders)
    if not response.json()["pagination"]["has_more"]:
        break
    offset += 100
```

### 3. 错误处理

- 始终检查 HTTP 状态码
- 解析错误响应中的详细信息
- 对 5xx 错误实现指数退避重试
- 记录带请求 ID 的错误日志以便排查

### 4. 缓存

缓存那些变化较少的数据：
- 商品目录（缓存 1 小时）
- 客户信息（缓存 15 分钟）
- 永远不要缓存订单状态（它要求实时性）

---

## 版本控制

API 使用基于 URL 的版本控制（例如 `/v2/orders`）。在同一主版本内，我们保持向后兼容。

**弃用策略**：端点会在移除前 6 个月被标记弃用。被弃用的端点会返回 `Warning` 请求头。

**当前版本**：v2.3（发布于 2025-01-15）

---

## SDK 与库

官方 SDK：
- Python：`pip install acmecorp`
- JavaScript：`npm install @acmecorp/api`
- Ruby：`gem install acmecorp`

所有 SDK 都会自动处理认证、重试与分页。

---

## 支持

- **文档**：https://docs.acmecorp.com
- **状态页**：https://status.acmecorp.com
- **邮箱**：api-support@acmecorp.com
- **响应时间**：标准版 24 小时，Enterprise 版 4 小时

---

## 更新日志

### v2.3（2025-01-15）
- 新增退款事件的 webhook 支持
- 提高 Pro 套餐速率限制
- 新增 `PATCH /orders/{id}/status` 端点

### v2.2（2024-10-01）
- 新增幂等 key 支持
- 新增客户偏好端点
- 修复分页相关 bug

### v2.1（2024-07-01）
- 新增商品筛选
- 改进错误消息
- 性能优化

### v2.0（2024-01-15）
- API 全面重构
- 新认证系统
- 相对 v1 的破坏性变更（见迁移指南）
