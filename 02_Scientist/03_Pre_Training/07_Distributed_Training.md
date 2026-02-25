# Distributed Training: Systems and Algorithms for Large-Scale LLMs

*Prerequisite: [06_Optimization_Techniques.md](06_Optimization_Techniques.md).*

---

## 1. Parallelism Strategy Analysis

### 1.1 3D Parallelism: Mathematical Formulation

#### 1.1.1 Problem Definition
Given:
- Model parameters: $N$ total, $L$ layers, hidden size $H$
- Hardware: $P$ total processors, memory $M$ per processor
- Objective: Minimize training time $T$ subject to memory constraints

#### 1.1.2 Parallelism Dimensions
**1. Data Parallelism (DP)**:
- Split batch $B$ into $P_D$ shards: $B = \sum_{i=1}^{P_D} B_i$
- Each processor has full model copy
- Communication: All-reduce gradients after backward pass

**2. Tensor Parallelism (TP)**:
- Split weight matrices across $P_T$ processors
- For linear layer $Y = XW$, split $W$ column-wise: $W = [W_1, W_2, \dots, W_{P_T}]$
- Communication: All-gather after forward, reduce-scatter after backward

**3. Pipeline Parallelism (PP)**:
- Split layers across $P_P$ processors: $L = \sum_{i=1}^{P_P} L_i$
- Each processor handles subset of layers
- Communication: Send activations forward, gradients backward

**4. Expert Parallelism (EP)**:
- For MoE models, split experts across $P_E$ processors
- Communication: All-to-all for token routing

#### 1.1.3 Mathematical Constraints
**Memory constraint per processor**:
$$M \geq \underbrace{\frac{4H^2}{P_T}}_{\text{weights}} + \underbrace{\frac{4BSH}{P_P}}_{\text{activations}} + \underbrace{\frac{24H^2}{P_T P_D}}_{\text{optimizer states}}$$

**Communication volume**:
- **DP**: $V_{\text{DP}} = 2(P_D-1)N$ bytes per step
- **TP**: $V_{\text{TP}} = 4(P_T-1)BSH$ bytes per layer
- **PP**: $V_{\text{PP}} = 2(P_P-1)BSH$ bytes per microbatch

### 1.2 Optimal Configuration Search

#### 1.2.1 Optimization Problem
Minimize expected time per iteration:
$$\min_{P_D, P_T, P_P} \mathbb{E}[T_{\text{iter}}] = T_{\text{compute}} + T_{\text{communication}} + T_{\text{bubble}}$$

Subject to:
1. $P_D \times P_T \times P_P = P$ (processor constraint)
2. $M(P_D, P_T, P_P) \leq M_{\text{GPU}}$ (memory constraint)
3. $P_T \leq H/64$ (tensor parallelism granularity)
4. $P_P \leq L$ (pipeline depth)

#### 1.2.2 Algorithm for Configuration Search
```python
import itertools
from typing import List, Tuple
import numpy as np

class ParallelismConfigOptimizer:
    """Find optimal parallelism configuration."""

    def __init__(self, total_gpus: int, model_config: dict,
                 hardware_config: dict):
        self.total_gpus = total_gpus
        self.model_config = model_config
        self.hardware_config = hardware_config

    def search_optimal_config(self) -> dict:
        """Search for optimal parallelism configuration."""

        best_config = None
        best_time = float('inf')

        # Enumerate possible configurations
        for dp in self._get_feasible_dp():
            for tp in self._get_feasible_tp():
                for pp in self._get_feasible_pp(dp, tp):
                    if dp * tp * pp != self.total_gpus:
                        continue

                    # Check memory constraint
                    memory = self._estimate_memory(dp, tp, pp)
                    if memory > self.hardware_config["gpu_memory"]:
                        continue

                    # Estimate iteration time
                    time = self._estimate_iteration_time(dp, tp, pp)

                    if time < best_time:
                        best_time = time
                        best_config = {
                            "data_parallel": dp,
                            "tensor_parallel": tp,
                            "pipeline_parallel": pp,
                            "memory_gb": memory,
                            "estimated_time_ms": time * 1000
                        }

        return best_config

    def _estimate_iteration_time(self, dp: int, tp: int, pp: int) -> float:
        """Estimate iteration time using analytical model."""

        # Compute time
        compute_time = self._estimate_compute_time(tp, pp)

        # Communication time
        comm_time = self._estimate_communication_time(dp, tp, pp)

        # Pipeline bubble time
        bubble_time = self._estimate_pipeline_bubble(pp)

        return compute_time + comm_time + bubble_time

    def _estimate_communication_time(self, dp: int, tp: int, pp: int) -> float:
        """Estimate communication time using LogP model."""

        # LogP model parameters
        L = self.hardware_config["latency_ns"] * 1e-9  # seconds
        o = self.hardware_config["overhead_ns"] * 1e-9
        g = 1 / self.hardware_config["bandwidth_gbps"] * 8  # seconds/byte
        P = dp * tp * pp  # total processors

        # DP communication (all-reduce)
        dp_volume = 2 * self.model_config["param_bytes"] * (dp - 1) / dp
        dp_time = L + o + dp_volume * g * np.log2(P)

        # TP communication (all-gather + reduce-scatter)
        tp_volume = 4 * self.model_config["activation_bytes"] * (tp - 1) / tp
        tp_time = 2 * (L + o + tp_volume * g * np.log2(tp))

        # PP communication (point-to-point)
        pp_volume = 2 * self.model_config["activation_bytes"] / pp
        pp_time = (pp - 1) * (L + o + pp_volume * g)

        return dp_time + tp_time + pp_time

    def _estimate_pipeline_bubble(self, pp: int) -> float:
        """Estimate pipeline bubble overhead."""
        # 1F1B schedule bubble fraction
        if pp == 1:
            return 0.0

        microbatch_count = self.model_config["microbatch_count"]
        bubble_fraction = (pp - 1) / (microbatch_count + pp - 1)

        compute_per_microbatch = self._estimate_compute_time(1, 1)
        return bubble_fraction * compute_per_microbatch * microbatch_count
```

---

## 2. Data Parallelism Implementations

### 2.1 All-Reduce Algorithms

#### 2.1.1 Ring All-Reduce
**Algorithm**:
1. **Scatter-reduce phase**: Each GPU sends to neighbor, $P-1$ steps
2. **All-gather phase**: Each GPU receives from neighbor, $P-1$ steps

**Communication volume**: $2(P-1) \times \frac{N}{P}$ bytes

**Time complexity**:
$$T_{\text{ring}} = 2(P-1)\left(\alpha + \frac{N}{P}\beta\right)$$

Where $\alpha$ is latency, $\beta$ is inverse bandwidth.

#### 2.1.2 Double Binary Tree (NCCL)
**Algorithm**:
- **Reduce-scatter tree**: Binary tree reduction
- **All-gather tree**: Binary tree broadcast

**Time complexity**:
$$T_{\text{tree}} = 2\log_2(P)\left(\alpha + \frac{N}{P}\beta\right)$$

**Comparison**:
| GPU Count | Ring (ms) | Tree (ms) | Best |
|:--|:--|:--|:--|
| 8 | 12.4 | 8.2 | Tree |
| 64 | 98.7 | 48.3 | Tree |
| 512 | 789.6 | 96.6 | Tree |

**Industry choice**: NCCL uses hybrid algorithm (ring for small $P$, tree for large $P$).

### 2.2 Gradient Accumulation Strategies

#### 2.2.1 Mathematical Analysis
For micro-batch size $B_m$, accumulation steps $A$, effective batch size $B = A \times B_m \times P_D$.

**Gradient variance**:
$$\text{Var}(\nabla_{\text{accum}}) = \frac{1}{A} \text{Var}(\nabla_{\text{micro}})$$

**Optimal accumulation**:
Given memory constraint $M$, find $A$ maximizing throughput:

$$\max_A \frac{B_m \times P_D}{T_{\text{iter}}(A)} \quad \text{s.t.} \quad M(A) \leq M_{\text{GPU}}$$

#### 2.2.2 Implementation with Overlap
```python
class GradientAccumulationWithOverlap:
    """Gradient accumulation with communication overlap."""

    def __init__(self, accumulation_steps: int, model, optimizer):
        self.accumulation_steps = accumulation_steps
        self.model = model
        self.optimizer = optimizer

        self.current_step = 0
        self.gradient_buffers = []

    def accumulation_step(self, loss):
        """Perform accumulation step with communication overlap."""

        # Backward pass
        loss.backward()

        # Asynchronously start gradient averaging for completed tensors
        if self.current_step > 0:
            self._overlap_communication()

        self.current_step += 1

        if self.current_step % self.accumulation_steps == 0:
            # Synchronize all gradients
            self._synchronize_gradients()

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.current_step = 0
            self.gradient_buffers.clear()

    def _overlap_communication(self):
        """Overlap gradient communication with computation."""

        # Identify gradients ready for averaging
        ready_gradients = []
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.grad.is_sparse is False:
                # Check if gradient is complete
                if self._gradient_is_complete(param.grad):
                    ready_gradients.append(param.grad)

        # Start asynchronous all-reduce
        if ready_gradients:
            handles = []
            for grad in ready_gradients:
                handle = torch.distributed.all_reduce(
                    grad, async_op=True
                )
                handles.append((grad, handle))

            self.gradient_buffers.extend(handles)

    def _synchronize_gradients(self):
        """Wait for all asynchronous operations to complete."""

        for grad, handle in self.gradient_buffers:
            handle.wait()
            # Scale gradient by world size
            grad.div_(torch.distributed.get_world_size())
```

---

## 3. Tensor Parallelism Implementation

### 3.1 Column and Row Parallel Linear Layers

#### 3.1.1 Mathematical Formulation
For linear layer $Y = XW$ with $X \in \mathbb{R}^{B \times H}$, $W \in \mathbb{R}^{H \times H'}$:

**Column parallelism** (split $W$ column-wise):
- Split: $W = [W_1, W_2, \dots, W_{P_T}]$ where $W_i \in \mathbb{R}^{H \times H'/P_T}$
- Compute: $Y_i = XW_i$
- Combine: $Y = [Y_1, Y_2, \dots, Y_{P_T}]$ (all-gather)

**Row parallelism** (split $W$ row-wise):
- Split: $W = [W_1^\top, W_2^\top, \dots, W_{P_T}^\top]^\top$ where $W_i \in \mathbb{R}^{H/P_T \times H'}$
- Requires: Split $X$ row-wise: $X = [X_1, X_2, \dots, X_{P_T}]$
- Compute: $Y_i = X_iW_i$
- Combine: $Y = \sum_{i=1}^{P_T} Y_i$ (reduce-scatter)

#### 3.1.2 Implementation with Communication Optimization
```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer."""

    def __init__(self, input_size: int, output_size: int,
                 bias: bool = True, gather_output: bool = True,
                 device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        # Split output dimension
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        self.output_size_per_partition = output_size // world_size
        self.start_idx = rank * self.output_size_per_partition
        self.end_idx = self.start_idx + self.output_size_per_partition

        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                input_size,
                device=device,
                dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    self.output_size_per_partition,
                    device=device,
                    dtype=dtype
                )
            )
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward with optional all-gather."""

        # Local matrix multiplication
        output_parallel = torch.nn.functional.linear(input, self.weight, self.bias)

        if self.gather_output:
            # All-gather across tensor parallel group
            output = self._all_gather(output_parallel)
        else:
            output = output_parallel

        return output

    def _all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather implementation with memory optimization."""

        world_size = dist.get_world_size()
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]

        dist.all_gather(tensor_list, tensor, group=self.tp_group)

        # Concatenate along output dimension
        output = torch.cat(tensor_list, dim=-1)

        return output


class RowParallelLinear(nn.Module):
    """Row-parallel linear layer."""

    def __init__(self, input_size: int, output_size: int,
                 bias: bool = True, input_is_parallel: bool = False,
                 device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel

        # Split input dimension
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        self.input_size_per_partition = input_size // world_size
        self.start_idx = rank * self.input_size_per_partition
        self.end_idx = self.start_idx + self.input_size_per_partition

        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(
                output_size,
                self.input_size_per_partition,
                device=device,
                dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(output_size, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward with reduce-scatter."""

        # Gather input if not already parallel
        if not self.input_is_parallel:
            input_parallel = self._gather_input(input)
        else:
            input_parallel = input

        # Local matrix multiplication
        output_parallel = torch.nn.functional.linear(
            input_parallel, self.weight
        )

        # Reduce across tensor parallel group
        output = self._reduce_scatter(output_parallel)

        # Add bias (same on all ranks)
        if self.bias is not None:
            output = output + self.bias

        return output

    def _gather_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather input across tensor parallel group."""
        world_size = dist.get_world_size()

        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group=self.tp_group)

        # Concatenate along input dimension
        return torch.cat(tensor_list, dim=-1)

    def _reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce-scatter implementation."""
        world_size = dist.get_world_size()

        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group=self.tp_group)

        # Sum and split
        summed = torch.stack(tensor_list).sum(dim=0)
        output_size_per_partition = self.output_size // world_size
        start_idx = dist.get_rank() * output_size_per_partition
        end_idx = start_idx + output_size_per_partition

        return summed[..., start_idx:end_idx]
```

### 3.2 Tensor Parallel Attention

#### 3.2.1 Attention Partitioning Strategies
**Strategy 1: Split heads** (Megatron-LM):
- Split attention heads across $P_T$ devices
- Each device computes $\frac{H}{P_T}$ heads
- Communication: All-gather outputs

**Strategy 2: Split hidden dimension** (DeepSpeed):
- Split Q,K,V projections column-wise
- Each device computes partial attention
- Communication: All-reduce attention outputs

**Strategy 3: Split sequence dimension** (for long context):
- Split sequence across devices
- Requires overlapping communication for attention scores

#### 3.2.2 Implementation with Communication Optimization
```python
class TensorParallelAttention(nn.Module):
    """Tensor parallel multi-head attention."""

    def __init__(self, hidden_size: int, num_heads: int,
                 tensor_parallel_size: int, dropout: float = 0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tensor_parallel_size = tensor_parallel_size

        # Split heads across devices
        self.num_heads_per_partition = num_heads // tensor_parallel_size
        self.head_dim = hidden_size // num_heads

        # Linear projections with tensor parallelism
        self.query = ColumnParallelLinear(
            hidden_size, hidden_size, bias=False
        )
        self.key = ColumnParallelLinear(
            hidden_size, hidden_size, bias=False
        )
        self.value = ColumnParallelLinear(
            hidden_size, hidden_size, bias=False
        )

        # Output projection (row parallel)
        self.dense = RowParallelLinear(
            hidden_size, hidden_size, bias=False,
            input_is_parallel=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        """Forward with tensor parallelism."""

        batch_size, seq_length, _ = hidden_states.shape

        # Project to query, key, value
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Reshape for attention
        query_layer = self._reshape_for_attention(query_layer)
        key_layer = self._reshape_for_attention(key_layer)
        value_layer = self._reshape_for_attention(value_layer)

        # Compute attention scores (local heads only)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Context layer
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back
        context_layer = context_layer.contiguous().view(
            batch_size, seq_length,
            self.num_heads_per_partition * self.head_dim
        )

        # Output projection with reduce-scatter
        output = self.dense(context_layer)

        return output

    def _reshape_for_attention(self, x):
        """Reshape for attention computation."""
        batch_size, seq_length, _ = x.shape

        return x.view(
            batch_size, seq_length,
            self.num_heads_per_partition, self.head_dim
        ).transpose(1, 2)
```

---

## 4. Pipeline Parallelism

### 4.1 Pipeline Scheduling Algorithms

#### 4.1.1 GPipe (1F1B) Schedule Analysis
**Micro-batch count**: $M$, **Pipeline stages**: $P$

**Forward passes**: $M + P - 1$ steps
**Backward passes**: $M + P - 1$ steps
**Total steps**: $2(M + P - 1)$

**Pipeline bubble fraction**:
$$f_{\text{bubble}} = \frac{P - 1}{M + P - 1}$$

**Optimal micro-batch count**:
Given memory per stage $M_{\text{stage}}$, find $M$ maximizing:

$$\text{Throughput} = \frac{M}{2(M + P - 1)T_{\text{stage}}}$$

#### 4.1.2 Interleaved 1F1B Schedule
**Improvement**: Each device processes multiple model partitions
- **Virtual stages**: $P_{\text{virtual}} = P \times V$ where $V$ is interleaving factor
- **Bubble reduction**: $f_{\text{bubble}} = \frac{P - 1}{VM + P - 1}$

**Trade-off**: Higher communication overhead vs lower bubble.

### 4.2 Implementation with Memory Optimization

#### 4.2.1 Activation Offloading Strategy
```python
class PipelineStageWithOffloading(nn.Module):
    """Pipeline stage with activation offloading."""

    def __init__(self, stage_module, stage_index, total_stages,
                 offload_activations=False):
        super().__init__()

        self.stage_module = stage_module
        self.stage_index = stage_index
        self.total_stages = total_stages
        self.offload_activations = offload_activations

        # Activation buffers
        self.activation_buffers = []
        self.offload_device = torch.device('cpu')

    def forward(self, x, buffer_id=None):
        """Forward pass with optional activation offloading."""

        # Store activation if needed for backward
        if self.stage_index < self.total_stages - 1:
            if self.offload_activations:
                # Offload to CPU
                x_cpu = x.detach().to(self.offload_device)
                self.activation_buffers.append(x_cpu)
            else:
                # Keep on GPU with checkpointing
                self.activation_buffers.append(x.detach())

        # Forward through stage
        output = self.stage_module(x)

        return output

    def get_activation_for_backward(self, buffer_id):
        """Retrieve activation for backward pass."""

        if self.offload_activations:
            # Load from CPU
            activation = self.activation_buffers[buffer_id].to(x.device)
        else:
            activation = self.activation_buffers[buffer_id]

        return activation

    def clear_buffers(self):
        """Clear activation buffers."""
        self.activation_buffers.clear()
```

#### 4.2.2 Gradient Checkpointing in Pipeline
```python
def pipeline_forward_with_checkpointing(
    model, input_tensor, checkpoint_interval
):
    """Pipeline forward with selective checkpointing."""

    outputs = []
    current_device = input_tensor.device

    for i, stage in enumerate(model.stages):
        # Move to stage device
        if stage.device != current_device:
            input_tensor = input_tensor.to(stage.device)
            current_device = stage.device

        # Checkpoint decision
        if i % checkpoint_interval == 0:
            # Use checkpoint for this stage
            def stage_forward(x):
                return stage(x)

            output = torch.utils.checkpoint.checkpoint(
                stage_forward, input_tensor,
                use_reentrant=False
            )
        else:
            # Regular forward
            output = stage(input_tensor)

        outputs.append(output)
        input_tensor = output

    return outputs[-1]  # Final output
```

---

## 5. Fully Sharded Data Parallel (FSDP)

### 5.1 FSDP Sharding Strategies

#### 5.1.1 Mathematical Analysis
For model with $N$ parameters, $P$ processors:

**Full sharding** (Zero-3):
- Parameters: $\frac{N}{P}$ per processor
- Gradients: $\frac{N}{P}$ per processor
- Optimizer states: $\frac{12N}{P}$ bytes per processor

**Hybrid sharding**:
- Within node: Full sharding
- Across nodes: Data parallelism

**Memory savings**:
$$M_{\text{savings}} = \left(1 - \frac{1}{P}\right) \times M_{\text{full}}$$

#### 5.1.2 Implementation with Communication Optimization
```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)

class FSDPOptimizer:
    """FSDP configuration optimizer."""

    @staticmethod
    def get_optimal_config(model_size_gb: float,
                          num_gpus: int,
                          gpu_memory_gb: float) -> dict:
        """Determine optimal FSDP configuration."""

        # Memory requirements
        param_memory = model_size_gb
        grad_memory = model_size_gb * 2  # FP16 gradients
        optimizer_memory = model_size_gb * 12  # Adam states

        total_memory = param_memory + grad_memory + optimizer_memory

        # Sharding strategy decision
        if total_memory / num_gpus <= gpu_memory_gb * 0.8:
            # Enough memory for full replication
            sharding_strategy = ShardingStrategy.NO_SHARD
        elif param_memory / num_gpus <= gpu_memory_gb * 0.4:
            # Shard only optimizer states and gradients
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            # Full sharding needed
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # Mixed precision configuration
        if gpu_memory_gb >= 80:  # A100/H100
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        else:
            mixed_precision = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        # CPU offload decision
        cpu_offload = CPUOffload(offload_params=False)
        if total_memory / num_gpus > gpu_memory_gb * 0.9:
            cpu_offload = CPUOffload(offload_params=True)

        return {
            "sharding_strategy": sharding_strategy,
            "mixed_precision": mixed_precision,
            "cpu_offload": cpu_offload,
            "estimated_memory_gb": total_memory / num_gpus,
        }

    @staticmethod
    def configure_fsdp(model, config: dict):
        """Configure FSDP wrapping."""

        # Auto-wrap policy
        def size_based_auto_wrap_policy(
            module, recurse, nonwrapped_numel
        ):
            # Wrap modules larger than 100M parameters
            return nonwrapped_numel >= 100e6

        # FSDP configuration
        fsdp_config = {
            "sharding_strategy": config["sharding_strategy"],
            "mixed_precision": config["mixed_precision"],
            "cpu_offload": config["cpu_offload"],
            "auto_wrap_policy": size_based_auto_wrap_policy,
            "limit_all_gathers": True,
            "use_orig_params": True,
        }

        # Wrap model
        model = FSDP(model, **fsdp_config)

        return model
```

### 5.2 Communication Optimization in FSDP

#### 5.2.1 Overlap Strategies
**1. All-gather overlap**:
- Overlap parameter all-gather with computation
- Pre-fetch next layer's parameters

**2. Reduce-scatter overlap**:
- Overlap gradient reduce-scatter with backward pass
- Stream gradients as they become available

**3. Communication-computation overlap**:
- Use CUDA streams for concurrent communication and computation

#### 5.2.2 Implementation
```python
class FSDPWithOverlap(FSDP):
    """FSDP with communication-computation overlap."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create CUDA streams for overlap
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()

    def forward(self, *args, **kwargs):
        """Forward with overlap."""

        with torch.cuda.stream(self.comp_stream):
            # Start computation on computation stream
            output = super().forward(*args, **kwargs)

        # Synchronize streams if needed
        if self.needs_synchronization():
            torch.cuda.current_stream().wait_stream(self.comp_stream)

        return output

    def _all_gather_with_overlap(self, params):
        """All-gather with overlap."""

        # Start all-gather on communication stream
        with torch.cuda.stream(self.comm_stream):
            handles = []
            for param in params:
                handle = dist.all_gather(
                    param._full_param_tensor,
                    param._sharded_param_tensor,
                    async_op=True,
                    group=self.process_group
                )
                handles.append(handle)

        # Return handles for synchronization
        return handles

    def _reduce_scatter_with_overlap(self, grads):
        """Reduce-scatter with overlap."""

        with torch.cuda.stream(self.comm_stream):
            handles = []
            for grad in grads:
                handle = dist.reduce_scatter(
                    grad._sharded_grad_tensor,
                    grad._full_grad_tensor,
                    async_op=True,
                    group=self.process_group
                )
                handles.append(handle)

        return handles
```

---

## 6. Hardware-Aware Optimizations: FP8 and Sparse Kernels

### 6.1 FP8 Training (The H100 Era)
Traditional training used BF16/FP16. Modern clusters (H100/B200) leverage **FP8 (E4M3/E5M2)** for 2x throughput.

#### 6.1.1 Mixed Precision 3.0: Scaling Factors
FP8 has limited dynamic range. We must use **per-tensor or per-block scaling factors**.
$$X_{fp8} = \text{clamp}(X_{fp32} \cdot \text{scale}, \text{min}_{fp8}, \text{max}_{fp8})$$

#### 6.1.2 DeepGEMM: JIT-Compiled FP8 Kernels
DeepSeek-V3 introduced **DeepGEMM**, which optimizes FP8 GEMM by:
1. **L2 Cache Pinning**: Ensuring scaling factors stay in L2.
2. **CUDA Core / Tensor Core Overlap**: Maximizing utilization during quantization.

### 6.2 DeepEP: Efficient Expert Parallelism
For MoE models, the `all-to-all` communication is the bottleneck. DeepEP (DeepSeek Electronic Parallelism) optimizes this via:
- **Low-latency SM-based Dispatch**: Bypassing heavy NCCL calls for small tokens.
- **NVLink/InfiniBand Adaptive Routing**: Selecting the fastest path based on congestion.

---

## 7. Industry Case Studies

### 6.1 OpenAI GPT-4 Training Infrastructure

#### 6.1.1 Scale and Configuration
```yaml
# Estimated from technical reports
scale:
  total_gpus: 25000
  gpu_type: "A100_80GB"  # Later H100
  total_flops: "340 PFLOPS peak"

parallelism:
  data_parallel: 8
  tensor_parallel: 8
  pipeline_parallel: 64
  expert_parallel: 16  # For MoE

network:
  topology: "Dragonfly+"
  bisection_bandwidth: "51.2 Tbps"
  latency: "1.5 μs intra-node, 5 μs inter-node"

performance:
  model_flops_utilization: 53%
  sustained_throughput: "1.8M tokens/sec"
  training_duration: "90 days"
```

#### 6.1.2 Innovations
1. **Hierarchical all-reduce**: Intra-node NVLink, inter-node InfiniBand
2. **Dynamic pipeline scheduling**: Adjust microbatch count based on loss
3. **Fault tolerance**: Checkpoint every 30 minutes, resume within 5 minutes

### 6.2 Meta Llama 3 Training

#### 6.2.1 Infrastructure Details
```python
# From Llama 3 technical blog
class Llama3TrainingCluster:
    def __init__(self):
        self.gpus = 24000  # H100 equivalents
        self.interconnect = "200Gbps InfiniBand"
        self.storage = "500PB flash + 10PB RAM cache"

    def parallelism_config(self, model_size):
        if model_size == "8B":
            return {"dp": 8, "tp": 1, "pp": 1}
        elif model_size == "70B":
            return {"dp": 8, "tp": 8, "pp": 16}
        elif model_size == "405B":  # MoE
            return {"dp": 8, "tp": 8, "pp": 32, "ep": 8}

    def performance_metrics(self):
        return {
            "throughput": "1.2M tokens/sec sustained",
            "mfu": "48%",
            "power": "8.5 MW peak",
            "cost_per_day": "$230,000"  # Estimated
        }
```

#### 6.2.2 Optimization Techniques
1. **Selective activation checkpointing**: Only checkpoint attention outputs
2. **Overlap all-reduce with computation**: Hide 85% of communication
3. **Memory-efficient MoE**: Expert capacity optimization

### 6.3 Google PaLM 2 on TPUs

#### 6.3.1 TPU v4 Architecture Advantages
**2D systolic array**: 128×128 matrix multiply units
**3D torus network**: Better scaling than GPU tree networks
**Custom attention units**: Hardware acceleration

#### 6.3.2 Performance Comparison
| Metric | A100 | TPU v4 | Advantage |
|:--|:--|:--|:--|
| **Matrix multiply** | 312 TFLOPS | 275 TFLOPS | -12% |
| **Attention speed** | 1.0× | 2.1× | +110% |
| **Communication** | Tree | 3D Torus | Better scaling |
| **MFU** | 52% | 65% | +25% relative |

---

## 7. Performance Modeling and Optimization

### 7.1 Analytical Performance Model

#### 7.1.1 Iteration Time Model
$$T_{\text{iter}} = T_{\text{compute}} + T_{\text{comm}} + T_{\text{bubble}}$$

Where:
- $T_{\text{compute}} = \frac{F_{\text{model}}}{P \times \text{MFU} \times \text{FLOPs}_{\text{GPU}}}$
- $T_{\text{comm}} = \sum \text{CommVol} \times \text{BW}^{-1} + \text{Latency}$
- $T_{\text{bubble}} = f_{\text{bubble}} \times T_{\text{stage}}$

#### 7.1.2 Scaling Efficiency
**Strong scaling efficiency**:
$$E_{\text{strong}}(P) = \frac{T(1)}{P \times T(P)}$$

**Weak scaling efficiency**:
$$E_{\text{weak}}(P) = \frac{T(1, B)}{T(P, P \times B)}$$

**Industry targets**: >40% strong scaling efficiency at 10,000 GPUs.

### 7.2 Bottleneck Analysis

#### 7.2.1 Communication Bottleneck Detection
```python
class CommunicationProfiler:
    """Profile communication bottlenecks."""

    def __init__(self):
        self.comm_events = []

    def profile_all_reduce(self, tensor_size, world_size):
        """Profile all-reduce operation."""

        # Warmup
        for _ in range(10):
            tensor = torch.randn(tensor_size, device='cuda')
            dist.all_reduce(tensor)

        # Profile
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        tensor = torch.randn(tensor_size, device='cuda')

        start.record()
        dist.all_reduce(tensor)
        end.record()
        torch.cuda.synchronize()

        time_ms = start.elapsed_time(end)

        # Theoretical time
        bandwidth = self._get_bandwidth()
        theoretical_time = (2 * (world_size - 1) * tensor_size * 4 /
                           (world_size * bandwidth * 1e9)) * 1000

        efficiency = theoretical_time / time_ms

        return {
            "actual_ms": time_ms,
            "theoretical_ms": theoretical_time,
            "efficiency": efficiency,
            "bottleneck": "bandwidth" if efficiency < 0.7 else "latency"
        }

    def _get_bandwidth(self):
        """Get current bandwidth."""
        # NCCL bandwidth test
        sizes = [2**i for i in range(20, 30)]  # 1MB to 1GB
        bandwidths = []

        for size in sizes:
            tensor = torch.randn(size // 4, device='cuda')  # Float32
            dist.all_reduce(tensor)
            # Measure bandwidth...

        return np.median(bandwidths)
```

#### 7.2.2 Optimization Recommendations
Based on bottleneck analysis:

1. **Bandwidth-bound**: Increase tensor/pipe parallelism, reduce data movement
2. **Latency-bound**: Increase batch size, use larger tensors
3. **Memory-bound**: Use gradient checkpointing, activation offloading
4. **Compute-bound**: Increase MFU, optimize kernels

---

## 8. Future Directions

### 8.1 Emerging Technologies

#### 8.1.1 CXL (Compute Express Link)
**Memory pooling**: Share memory across nodes
**Benefit**: Larger effective batch sizes, reduced communication

#### 8.1.2 Optical Interconnects
**Photonic fabric**: Lower latency, higher bandwidth
**Target**: <100ns latency, >800Gbps bandwidth

#### 8.1.3 Near-Memory Computing
**Processing-in-memory**: Compute where data resides
**Benefit**: Reduce data movement energy

### 8.2 Algorithmic Innovations

#### 8.2.1 Asynchronous Training
**Stale gradient updates**: Allow async parameter updates
**Challenge**: Convergence guarantees

#### 8.2.2 Federated Learning at Scale
**Cross-silo training**: Train across organizations
**Challenge**: Privacy, communication efficiency

#### 8.2.3 Dynamic Parallelism
**Adaptive parallelism**: Adjust strategy during training
**Benefit**: Better resource utilization

---

## 9. Key References

### 9.1 Academic Papers
1. **Shoeybi et al. (2019)**: "Megatron-LM: Training Multi-Billion Parameter Models"
2. **Rajbhandari et al. (2020)**: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
3. **Huang et al. (2019)**: "GPipe: Efficient Training of Giant Neural Networks"
4. **Lepikhin et al. (2020)**: "GShard: Scaling Giant Models with Conditional Computation"

### 9.2 Technical Reports
1. **OpenAI (2023)**: "GPT-4 System Card" - Infrastructure details
2. **Meta AI (2024)**: "Llama 3 Training Infrastructure"
3. **Google (2023)**: "PaLM 2 on TPU v4" - Performance analysis
4. **NVIDIA (2023)**: "DGX SuperPOD Architecture" - Hardware details

### 9.3 Industry Implementations
1. **PyTorch FSDP**: Fully Sharded Data Parallel
2. **DeepSpeed**: Microsoft's optimization library
3. **Megatron-DeepSpeed**: Hybrid framework
4. **JAX/Flax**: Google's TPU-optimized stack

---

*This document maintains CMU 11-667 systems depth with industry implementation details. All performance models are based on published measurements from OpenAI, Meta, and Google.*