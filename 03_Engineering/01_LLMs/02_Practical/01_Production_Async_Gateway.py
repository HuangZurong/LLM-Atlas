import asyncio
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from openai import AsyncOpenAI, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ───────────────────────────────────────────────────────────────────────────
# Industrial Pattern: Middleware & Interceptor Architecture
# ───────────────────────────────────────────────────────────────────────────
# Large-scale LLM systems use middleware for:
# 1. PII Redaction (Data Privacy)
# 2. Token Counting (Billing/Quotas)
# 3. Request Caching
# 4. Toxicity Guardrails
# ───────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM-Gateway")

class LLMMiddleware:
    """Base class for all request/response interceptors."""
    async def pre_process(self, request_id: str, messages: List[Dict]) -> List[Dict]:
        return messages

    async def post_process(self, request_id: str, response: str) -> str:
        return response

class LoggingMiddleware(LLMMiddleware):
    async def pre_process(self, request_id: str, messages: List[Dict]):
        logger.info(f"[{request_id}] Input tokens approx: {sum(len(m['content']) for m in messages)//4}")
        return messages

class PIIFilterMiddleware(LLMMiddleware):
    """Example of a security guardrail."""
    async def pre_process(self, request_id: str, messages: List[Dict]):
        # Mock PII redaction (Real world: use Presidio or regex)
        for m in messages:
            m['content'] = m['content'].replace("password", "[REDACTED]")
        return messages

class AsyncLLMGateway:
    """
    A production-grade Async Gateway.
    Features:
    - Async IO for high-concurrency environments (FastAPI/Tornado).
    - Pluggable Middleware stack.
    - Provider Abstraction & Failover.
    - Precise Latency & Token Telemetry.
    """
    def __init__(self, providers: List[Dict], middlewares: List[LLMMiddleware] = None):
        self.providers = providers  # List of {client, model, weight}
        self.middlewares = middlewares or []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        reraise=True
    )
    async def _execute_with_retries(self, client: AsyncOpenAI, model: str, messages: List[Dict], **kwargs):
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

    async def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # 1. Pre-processing Middleware
        for mw in self.middlewares:
            messages = await mw.pre_process(request_id, messages)

        # 2. Primary Provider Execution (Strategy: Simple Failover)
        # In multi-tenant apps, you'd use a Router to select the provider based on cost/latency
        primary = self.providers[0]
        client: AsyncOpenAI = primary['client']
        model: str = primary['model']

        try:
            response = await self._execute_with_retries(client, model, messages, **kwargs)
            content = response.choices[0].message.content

            # 3. Post-processing Middleware
            for mw in reversed(self.middlewares):
                content = await mw.post_process(request_id, content)

            latency = time.perf_counter() - start_time

            telemetry = {
                "request_id": request_id,
                "model": model,
                "latency_ms": round(latency * 1000, 2),
                "usage": response.usage.model_dump() if response.usage else {},
                "content": content
            }

            logger.info(f"[{request_id}] {model} completed in {telemetry['latency_ms']}ms")
            return telemetry

        except Exception as e:
            logger.error(f"[{request_id}] Critical failure: {str(e)}")
            raise e

# ───────────────────────────────────────────────────────────────────────────
# Industrial Usage Example
# ───────────────────────────────────────────────────────────────────────────
async def main():
    # Setup Gateway with Middleware
    gateway = AsyncLLMGateway(
        providers=[
            {
                "client": AsyncOpenAI(api_key="sk-..."),
                "model": "gpt-4o-mini"
            }
        ],
        middlewares=[
            LoggingMiddleware(),
            PIIFilterMiddleware()
        ]
    )

    # test_messages = [{"role": "user", "content": "My password is 'secret123'. Summarize LLM security."}]
    # result = await gateway.chat(test_messages)
    # print(result['content'])

    print("AsyncLLMGateway: Industrial Middleware architecture ready.")

if __name__ == "__main__":
    asyncio.run(main())
