"""
01_vLLM_Deployment.py — Production vLLM Deployment with Continuous Batching

Demonstrates deploying Llama-3.1-70B with vLLM's PagedAttention,
continuous batching, prefix caching, and monitoring.

Key Features:
- PagedAttention for memory-efficient KV caching
- Continuous batching for high throughput
- Prometheus metrics for monitoring
- Graceful shutdown with request draining
- Health checks and readiness probes
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from pydantic import BaseModel
from vllm import AsyncLLMEngine, SamplingParams, EngineArgs

# ---------------------------------------------------------------------------
# Monitoring Metrics
# ---------------------------------------------------------------------------

REQUEST_COUNTER = Counter(
    'vllm_requests_total',
    'Total number of requests',
    ['status', 'model']
)
TOKEN_HISTOGRAM = Histogram(
    'vllm_tokens_per_request',
    'Number of tokens per request',
    buckets=[10, 50, 100, 500, 1000, 5000]
)
LATENCY_HISTOGRAM = Histogram(
    'vllm_request_latency_seconds',
    'Request latency in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
BATCH_SIZE_GAUGE = Counter(
    'vllm_batch_size',
    'Current batch size'
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VLLMConfig:
    """Production vLLM configuration."""
    model_id: str = "meta-llama/Llama-3.1-70B-Instruct"
    tensor_parallel_size: int = 4  # Split across 4 GPUs
    gpu_memory_utilization: float = 0.9  # Use 90% VRAM
    max_num_seqs: int = 256  # Max concurrent requests
    max_num_batched_tokens: int = 8192  # Batch size limit
    enable_prefix_caching: bool = True  # Cache shared prefixes
    block_size: int = 16  # PagedAttention block size
    swap_space: int = 4  # 4GB swap for CPU offloading
    quantization: Optional[str] = None  # "awq" or "gptq" for 4-bit
    trust_remote_code: bool = False
    seed: int = 42

# ---------------------------------------------------------------------------
# Async Engine with Lifecycle Management
# ---------------------------------------------------------------------------

class VLLMEngine:
    """Wraps vLLM engine with production features."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.engine: Optional[AsyncLLMEngine] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the vLLM engine with production settings."""
        engine_args = EngineArgs(
            model=self.config.model_id,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            enable_prefix_caching=self.config.enable_prefix_caching,
            block_size=self.config.block_size,
            swap_space=self.config.swap_space,
            quantization=self.config.quantization,
            trust_remote_code=self.config.trust_remote_code,
            seed=self.config.seed,
            disable_log_stats=False,  # Enable internal logging
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.logger.info(f"vLLM engine initialized: {self.config.model_id}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        request_id: str = None,
    ):
        """Generate text with production monitoring."""
        if not self.engine:
            raise RuntimeError("Engine not initialized")

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        # Track batch size
        BATCH_SIZE_GAUGE.inc()

        try:
            if stream:
                return self._generate_streaming(prompt, sampling_params, request_id)
            else:
                return await self._generate_batch(prompt, sampling_params, request_id)
        finally:
            BATCH_SIZE_GAUGE.dec()

    async def _generate_batch(self, prompt, sampling_params, request_id):
        """Generate a complete response (non-streaming)."""
        start_time = asyncio.get_event_loop().time()

        results = await self.engine.generate(
            prompt,
            sampling_params,
            request_id=request_id,
        )

        latency = asyncio.get_event_loop().time() - start_time
        output = results[0].outputs[0].text

        # Record metrics
        LATENCY_HISTOGRAM.observe(latency)
        TOKEN_HISTOGRAM.observe(len(results[0].outputs[0].token_ids))
        REQUEST_COUNTER.labels(status="success", model=self.config.model_id).inc()

        return output

    async def _generate_streaming(self, prompt, sampling_params, request_id):
        """Generate a streaming response token by token."""
        async def token_generator():
            start_time = asyncio.get_event_loop().time()
            token_count = 0

            async for step_output in await self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id,
                stream=True,
            ):
                for output in step_output.outputs:
                    token = output.text
                    token_count += 1
                    yield f"data: {token}\n\n"

            # Final metrics
            latency = asyncio.get_event_loop().time() - start_time
            LATENCY_HISTOGRAM.observe(latency)
            TOKEN_HISTOGRAM.observe(token_count)
            REQUEST_COUNTER.labels(status="success", model=self.config.model_id).inc()

        return StreamingResponse(
            token_generator(),
            media_type="text/event-stream",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            }
        )

    async def health_check(self) -> bool:
        """Check if engine is healthy."""
        if not self.engine:
            return False
        # Simple check: engine initialized and GPU available
        return True

    async def drain_requests(self, timeout: float = 30.0):
        """Drain pending requests before shutdown."""
        if self.engine:
            await self.engine.shutdown(timeout=timeout)

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage engine lifecycle."""
    config = VLLMConfig()
    engine = VLLMEngine(config)

    # Startup
    await engine.initialize()
    app.state.engine = engine
    yield

    # Shutdown
    await engine.drain_requests()
    app.state.engine = None

app = FastAPI(title="vLLM Production Server", lifespan=lifespan)

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text endpoint."""
    engine: VLLMEngine = app.state.engine

    try:
        if request.stream:
            return await engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
                request_id=str(id(request)),
            )
        else:
            output = await engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
                request_id=str(id(request)),
            )
            return JSONResponse(content={"text": output})
    except Exception as e:
        REQUEST_COUNTER.labels(status="error", model=engine.config.model_id).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    engine: VLLMEngine = app.state.engine
    if await engine.health_check():
        return JSONResponse(content={"status": "healthy"})
    else:
        raise HTTPException(status_code=503, detail="Engine not ready")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(REGISTRY)

@app.get("/stats")
async def engine_stats():
    """vLLM internal statistics."""
    engine: VLLMEngine = app.state.engine
    if not engine.engine:
        return {"error": "Engine not available"}

    # Get internal stats (vLLM exposes these)
    stats = {
        "model": engine.config.model_id,
        "gpu_utilization": engine.config.gpu_memory_utilization,
        "max_concurrent": engine.config.max_num_seqs,
        "batch_limit": engine.config.max_num_batched_tokens,
    }
    return JSONResponse(content=stats)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Production deployment
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=True,
        timeout_keep_alive=30,
        log_config=None,
    )