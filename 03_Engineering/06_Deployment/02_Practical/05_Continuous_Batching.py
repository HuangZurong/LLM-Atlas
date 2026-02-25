"""
05_Continuous_Batching.py — Simulation of Continuous Batching & Request Scheduling

Continuous batching is a core optimization in LLM serving (vLLM, TGI) that
allows new requests to join a running batch without waiting for earlier
requests to finish.

This script provides a high-level simulation of:
1. Request queuing with priorities
2. Continuous batching logic (Iteration-level scheduling)
3. Pre-fill vs. Decoding phase management
4. Performance metrics (Throughput, Latency, Waiting Time)
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    COMPLETED = "completed"

@dataclass
class LLMRequest:
    id: int
    prompt_len: int
    output_len: int
    priority: int = 1  # 1: Low, 2: Medium, 3: High
    arrival_time: float = field(default_factory=time.time)

    # Internal state
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: int = 0
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.generated_tokens

# ---------------------------------------------------------------------------
# Continuous Batching Scheduler
# ---------------------------------------------------------------------------

class ContinuousBatcher:
    """
    Simulates an LLM scheduler with continuous batching support.
    """
    def __init__(self, max_batch_size: int = 4, tokens_per_iter: int = 1):
        self.max_batch_size = max_batch_size
        self.tokens_per_iter = tokens_per_iter  # Tokens generated per 'step'
        self.waiting_queue: List[LLMRequest] = []
        self.active_batch: List[LLMRequest] = []
        self.completed_requests: List[LLMRequest] = []

        # Metrics
        self.total_iterations = 0
        self.total_tokens_generated = 0

    def add_request(self, request: LLMRequest):
        """Add a new request to the waiting queue."""
        # Sort by priority (higher first), then by arrival time
        self.waiting_queue.append(request)
        self.waiting_queue.sort(key=lambda x: (-x.priority, x.arrival_time))
        print(f"[Queue] Request {request.id} added (P{request.priority}, Prompt: {request.prompt_len})")

    def step(self):
        """Perform one iteration of continuous batching."""
        self.total_iterations += 1

        # 1. Admit new requests into the batch if there is space
        while len(self.active_batch) < self.max_batch_size and self.waiting_queue:
            req = self.waiting_queue.pop(0)
            req.status = RequestStatus.PREFILL
            req.start_time = time.time()
            self.active_batch.append(req)
            print(f"  [Admit] Request {req.id} joined the batch (Status: {req.status})")

        # 2. Process active batch
        if not self.active_batch:
            return

        print(f"  [Iter {self.total_iterations}] Batch Size: {len(self.active_batch)}")

        finished_this_step = []
        for req in self.active_batch:
            # Pre-fill phase (happens in the first iteration for the request)
            if req.status == RequestStatus.PREFILL:
                # In real life, pre-fill processes all prompt tokens at once
                # Here we simulate it moving to decoding immediately
                req.status = RequestStatus.DECODING
                print(f"    - Req {req.id}: Prefill done")

            # Decoding phase
            if req.status == RequestStatus.DECODING:
                req.generated_tokens += self.tokens_per_iter
                self.total_tokens_generated += self.tokens_per_iter

                # Check if request is finished
                if req.generated_tokens >= req.output_len:
                    req.status = RequestStatus.COMPLETED
                    req.finish_time = time.time()
                    finished_this_step.append(req)

        # 3. Clean up completed requests
        for req in finished_this_step:
            self.active_batch.remove(req)
            self.completed_requests.append(req)
            latency = req.finish_time - req.arrival_time
            print(f"  [Finish] Request {req.id} completed (Latency: {latency:.2f}s)")

    def is_empty(self) -> bool:
        return not self.waiting_queue and not self.active_batch

# ---------------------------------------------------------------------------
# Simulation Runner
# ---------------------------------------------------------------------------

async def run_simulation():
    batcher = ContinuousBatcher(max_batch_size=4)

    # Initial requests
    requests = [
        LLMRequest(id=1, prompt_len=10, output_len=5, priority=1),
        LLMRequest(id=2, prompt_len=5,  output_len=10, priority=2),
    ]

    for r in requests:
        batcher.add_request(r)

    # Simulation loop
    req_counter = 3
    while not batcher.is_empty() or req_counter < 8:
        print("-" * 40)
        batcher.step()

        # Simulate new requests arriving randomly
        if req_counter < 8 and random.random() > 0.5:
            new_req = LLMRequest(
                id=req_counter,
                prompt_len=random.randint(5, 15),
                output_len=random.randint(5, 15),
                priority=random.randint(1, 3)
            )
            batcher.add_request(new_req)
            req_counter += 1

        await asyncio.sleep(0.1) # Simulate time passing

    # Show Final Metrics
    print("\n" + "=" * 40)
    print("Simulation Complete - Final Metrics")
    print("=" * 40)

    total_latency = 0
    for r in batcher.completed_requests:
        latency = r.finish_time - r.arrival_time
        total_latency += latency
        print(f"Req {r.id}: Wait: {r.start_time-r.arrival_time:.2f}s, Latency: {latency:.2f}s, Tokens: {r.total_tokens}")

    avg_latency = total_latency / len(batcher.completed_requests)
    throughput = batcher.total_tokens_generated / batcher.total_iterations

    print("-" * 40)
    print(f"Average Latency: {avg_latency:.2f}s")
    print(f"Avg Tokens/Iter: {throughput:.2f}")
    print(f"Total Iterations: {batcher.total_iterations}")

if __name__ == "__main__":
    asyncio.run(run_simulation())
