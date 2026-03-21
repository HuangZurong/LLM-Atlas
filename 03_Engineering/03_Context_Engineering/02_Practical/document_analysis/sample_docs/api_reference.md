# AcmeCorp API Reference Guide v2.3

## Overview

The AcmeCorp API provides programmatic access to our e-commerce platform. This document covers authentication, core endpoints, rate limits, and best practices.

**Base URL**: `https://api.acmecorp.com/v2`

**Authentication**: All requests require an API key in the `X-API-Key` header.

---

## Authentication

### Getting Your API Key

1. Log into your AcmeCorp Dashboard
2. Navigate to Settings > API Keys
3. Click "Generate New Key"
4. Store the key securely — it will only be shown once

**Important**: API keys should never be shared or committed to version control. Use environment variables instead.

### Authenticating Requests

```bash
curl -H "X-API-Key: your_api_key_here" \
     https://api.acmecorp.com/v2/orders
```

Invalid or missing API keys will return `401 Unauthorized`.

---

## Rate Limits

| Tier | Requests/minute | Requests/day |
|------|----------------|--------------|
| Free | 60 | 1,000 |
| Pro | 600 | 10,000 |
| Enterprise | 6,000 | Unlimited |

When you exceed the rate limit, the API returns `429 Too Many Requests` with a `Retry-After` header indicating seconds to wait.

**Best Practice**: Implement exponential backoff in your client code. Start with 1 second, double on each retry, max 60 seconds.

---

## Core Endpoints

### Orders

#### List Orders

**Endpoint**: `GET /orders`

**Parameters**:
- `status` (string, optional): Filter by status. Values: `pending`, `processing`, `shipped`, `delivered`, `cancelled`
- `limit` (integer, optional): Number of results (default: 20, max: 100)
- `offset` (integer, optional): Pagination offset (default: 0)

**Response**:
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

#### Get Single Order

**Endpoint**: `GET /orders/{order_id}`

**Response**: Single order object with full details including tracking information.

#### Create Order

**Endpoint**: `POST /orders`

**Request Body**:
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

**Response**: `201 Created` with the created order object.

#### Update Order Status

**Endpoint**: `PATCH /orders/{order_id}/status`

**Request Body**:
```json
{
  "status": "shipped",
  "tracking_number": "1Z999AA10123456784"
}
```

**Note**: Only admin accounts can update order status. Regular API keys will receive `403 Forbidden`.

#### Cancel Order

**Endpoint**: `POST /orders/{order_id}/cancel`

**Conditions**:
- Orders can only be cancelled if status is `pending` or `processing`
- Shipped orders require a refund request instead

**Response**: `200 OK` with updated order object.

---

### Products

#### List Products

**Endpoint**: `GET /products`

**Parameters**:
- `category` (string, optional): Filter by category
- `min_price` (number, optional): Minimum price
- `max_price` (number, optional): Maximum price
- `in_stock` (boolean, optional): Only show in-stock items

#### Get Product

**Endpoint**: `GET /products/{product_id}`

Returns full product details including inventory levels, variants, and images.

---

### Customers

#### Get Customer

**Endpoint**: `GET /customers/{customer_id}`

**Response**:
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

#### Update Customer

**Endpoint**: `PATCH /customers/{customer_id}`

Updatable fields: `email`, `phone`, `address`, `preferences`

---

## Webhooks

AcmeCorp can send real-time notifications when events occur in your account.

### Supported Events

- `order.created`
- `order.shipped`
- `order.delivered`
- `order.cancelled`
- `refund.processed`
- `customer.updated`

### Setting Up Webhooks

1. Go to Dashboard > Settings > Webhooks
2. Enter your endpoint URL (must be HTTPS)
3. Select events to subscribe to
4. Verify the endpoint (we'll send a test request)

### Webhook Payload

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

**Security**: Always verify the signature using your webhook secret.

---

## Error Handling

All errors follow this format:

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

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource doesn't exist |
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server-side error (retry later) |

---

## Best Practices

### 1. Idempotency

For critical operations (creating orders, processing payments), include an `Idempotency-Key` header. If a request is retried with the same key, we'll return the original response instead of creating a duplicate.

```bash
curl -X POST \
     -H "X-API-Key: your_key" \
     -H "Idempotency-Key: unique_key_123" \
     -d '{"customer_id":"CUST-4521",...}' \
     https://api.acmecorp.com/v2/orders
```

### 2. Pagination

Always use pagination for list endpoints. Don't assume all results will fit in one response.

**Example**:
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

### 3. Error Handling

- Always check HTTP status codes
- Parse error responses for detailed messages
- Implement retry logic with exponential backoff for 5xx errors
- Log errors with request IDs for debugging

### 4. Caching

Cache rarely-changing data:
- Product catalog (cache for 1 hour)
- Customer info (cache for 15 minutes)
- Never cache order status (real-time)

---

## Versioning

The API uses URL-based versioning (e.g., `/v2/orders`). We maintain backward compatibility within a major version.

**Deprecation Policy**: Endpoints will be deprecated 6 months before removal. Deprecated endpoints return a `Warning` header.

**Current Version**: v2.3 (Released 2025-01-15)

---

## SDKs and Libraries

Official SDKs available:
- Python: `pip install acmecorp`
- JavaScript: `npm install @acmecorp/api`
- Ruby: `gem install acmecorp`

All SDKs handle authentication, retries, and pagination automatically.

---

## Support

- **Documentation**: https://docs.acmecorp.com
- **Status Page**: https://status.acmecorp.com
- **Email**: api-support@acmecorp.com
- **Response Time**: 24 hours for standard, 4 hours for Enterprise

---

## Changelog

### v2.3 (2025-01-15)
- Added webhook support for refund events
- Increased rate limits for Pro tier
- New `PATCH /orders/{id}/status` endpoint

### v2.2 (2024-10-01)
- Added idempotency key support
- New customer preferences endpoint
- Bug fixes for pagination

### v2.1 (2024-07-01)
- Added product filtering
- Improved error messages
- Performance optimizations

### v2.0 (2024-01-15)
- Complete API redesign
- New authentication system
- Breaking changes from v1 (see migration guide)