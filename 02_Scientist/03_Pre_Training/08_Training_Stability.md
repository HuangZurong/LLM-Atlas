# Training Stability: Diagnostics and Recovery for Large-Scale LLMs

*Prerequisite: [07_Distributed_Training.md](07_Distributed_Training.md).*

---

## 1. Loss Landscape Analysis

### 1.1 Gradient Norm Dynamics

#### 1.1.1 Theoretical Analysis
For loss function $L(\theta)$ with gradient $g(\theta) = \nabla L(\theta)$, consider gradient norm evolution:

$$\frac{d}{dt}\|g(\theta_t)\|^2 = 2\langle g(\theta_t), H(\theta_t)g(\theta_t)\rangle$$

Where $H(\theta_t) = \nabla^2 L(\theta_t)$ is Hessian.

**Key insight**: Gradient norm changes depend on **curvature along gradient direction**.

#### 1.1.2 Stable Training Condition
From optimization theory, training is stable if:

$$\eta \leq \frac{2}{\lambda_{\max}(H)}$$

Where $\lambda_{\max}(H)$ is maximum eigenvalue of Hessian (sharpness).

**Empirical finding**: Transformers have $\lambda_{\max}(H) \approx 10-100$ early in training, decreasing to $1-10$ later.

#### 1.1.3 Gradient Norm Statistics
```python
class GradientNormMonitor:
    """Monitor gradient norm statistics."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.gradient_norms = []
        self.layer_gradient_norms = collections.defaultdict(list)

    def analyze_gradient_norms(self, model, step: int) -> dict:
        """Analyze gradient norm distribution."""

        total_norm = 0.0
        layer_stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad.detach(), 2.0).item()
                total_norm += grad_norm ** 2

                # Layer-wise statistics
                layer_name = self._extract_layer_name(name)
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {
                        "count": 0,
                        "sum_norm_sq": 0.0,
                        "max_norm": 0.0
                    }

                layer_stats[layer_name]["count"] += 1
                layer_stats[layer_name]["sum_norm_sq"] += grad_norm ** 2
                layer_stats[layer_name]["max_norm"] = max(
                    layer_stats[layer_name]["max_norm"], grad_norm
                )

        total_norm = total_norm ** 0.5

        # Update history
        self.gradient_norms.append(total_norm)
        if len(self.gradient_norms) > self.window_size:
            self.gradient_norms.pop(0)

        # Compute statistics
        stats = {
            "step": step,
            "total_norm": total_norm,
            "mean_norm": np.mean(self.gradient_norms[-100:]),
            "std_norm": np.std(self.gradient_norms[-100:]),
            "layer_stats": {}
        }

        for layer_name, layer_data in layer_stats.items():
            avg_norm = (layer_data["sum_norm_sq"] / layer_data["count"]) ** 0.5
            stats["layer_stats"][layer_name] = {
                "avg_norm": avg_norm,
                "max_norm": layer_data["max_norm"],
                "parameter_count": layer_data["count"]
            }

        return stats

    def detect_anomalies(self, stats: dict) -> List[str]:
        """Detect gradient norm anomalies."""

        anomalies = []
        current_norm = stats["total_norm"]

        # Check for exploding gradients
        if current_norm > stats["mean_norm"] + 3 * stats["std_norm"]:
            anomalies.append(
                f"Exploding gradients: {current_norm:.2f} > "
                f"{stats['mean_norm']:.2f} + 3*{stats['std_norm']:.2f}"
            )

        # Check for vanishing gradients
        if current_norm < 1e-6:
            anomalies.append(
                f"Vanishing gradients: {current_norm:.2e} < 1e-6"
            )

        # Layer-wise anomalies
        for layer_name, layer_stat in stats["layer_stats"].items():
            if layer_stat["max_norm"] > 10.0:
                anomalies.append(
                    f"Large gradients in {layer_name}: "
                    f"max_norm={layer_stat['max_norm']:.2f}"
                )

        return anomalies

    def _extract_layer_name(self, param_name: str) -> str:
        """Extract layer name from parameter name."""
        # Example: "transformer.h.0.attention.query.weight" -> "layer_0_attention"
        parts = param_name.split('.')
        if len(parts) >= 3:
            return f"layer_{parts[2]}"
        return "other"
```

### 1.2 Loss Surface Curvature Analysis

#### 1.2.1 Hessian Spectrum Estimation
**Lanczos algorithm** for largest eigenvalue:
1. Initialize $q_1 = g/\|g\|$
2. For $k=1$ to $m$:
   - $r_k = Hq_k - \beta_{k-1}q_{k-1}$ (with $\beta_0=0$)
   - $\alpha_k = q_k^\top r_k$
   - $r_k = r_k - \alpha_k q_k$
   - $\beta_k = \|r_k\|$
   - $q_{k+1} = r_k/\beta_k$

**Result**: Tridiagonal matrix $T_m$ approximates extreme eigenvalues of $H$.

#### 1.2.2 Sharpness Measurement
**Definition**: $\lambda_{\max}/\lambda_{\min}$ ratio

**Empirical values** (GPT-3 training):
- **Early training**: Sharpness ≈ 100-1000
- **Mid training**: Sharpness ≈ 10-100
- **Late training**: Sharpness ≈ 1-10

**Stability condition**: Sharpness < 100 for stable training with $\eta=1e-4$.

---

## 2. Loss Spike Detection and Recovery

### 2.1 Statistical Detection Methods

#### 2.1.1 Z-Score Detection
For loss sequence $L_1, L_2, \dots, L_t$, compute:

$$\mu_t = \frac{1}{w}\sum_{i=t-w+1}^t L_i$$
$$\sigma_t = \sqrt{\frac{1}{w}\sum_{i=t-w+1}^t (L_i - \mu_t)^2}$$
$$z_t = \frac{L_t - \mu_t}{\sigma_t}$$

**Detection**: $|z_t| > 3$ indicates spike.

#### 2.1.2 CUSUM (Cumulative Sum) Detection
For target loss decrease rate $\delta$:

$$S_t^+ = \max(0, S_{t-1}^+ + L_t - \mu_t - \delta)$$
$$S_t^- = \min(0, S_{t-1}^- + L_t - \mu_t + \delta)$$

**Detection**: $S_t^+ > h$ or $S_t^- < -h$ for threshold $h$.

#### 2.1.3 Implementation with Multiple Windows
```python
class LossSpikeDetector:
    """Multi-window loss spike detection."""

    def __init__(self, short_window: int = 100,
                 long_window: int = 1000,
                 z_threshold: float = 3.0,
                 cusum_threshold: float = 5.0):
        self.short_window = short_window
        self.long_window = long_window
        self.z_threshold = z_threshold
        self.cusum_threshold = cusum_threshold

        self.loss_history = collections.deque(maxlen=long_window)
        self.cusum_positive = 0.0
        self.cusum_negative = 0.0

    def detect_spike(self, current_loss: float) -> dict:
        """Detect loss spike using multiple methods."""

        self.loss_history.append(current_loss)

        if len(self.loss_history) < self.short_window:
            return {"spike": False, "reason": "insufficient_data"}

        # Method 1: Z-score detection
        recent_losses = list(self.loss_history)[-self.short_window:]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses) + 1e-8

        z_score = abs(current_loss - mean_loss) / std_loss
        z_detected = z_score > self.z_threshold

        # Method 2: CUSUM detection
        target_decrease = 0.01  # 1% expected decrease per step

        self.cusum_positive = max(
            0, self.cusum_positive + current_loss - mean_loss - target_decrease
        )
        self.cusum_negative = min(
            0, self.cusum_negative + current_loss - mean_loss + target_decrease
        )

        cusum_detected = (
            self.cusum_positive > self.cusum_threshold or
            abs(self.cusum_negative) > self.cusum_threshold
        )

        # Method 3: Rate of change detection
        if len(self.loss_history) >= 10:
            recent_gradient = np.gradient(list(self.loss_history)[-10:])
            max_gradient = np.max(np.abs(recent_gradient))
            rate_detected = max_gradient > 2.0  # Loss doubling in one step

        spike_detected = z_detected or cusum_detected or rate_detected

        return {
            "spike": spike_detected,
            "z_score": z_score,
            "cusum_positive": self.cusum_positive,
            "cusum_negative": self.cusum_negative,
            "detection_methods": {
                "z_score": z_detected,
                "cusum": cusum_detected,
                "rate": rate_detected if 'rate_detected' in locals() else False
            }
        }

    def get_recovery_actions(self, detection_result: dict) -> List[str]:
        """Get recommended recovery actions."""

        actions = []

        if detection_result["z_score"] > 5.0:
            actions.extend([
                "revert_to_checkpoint",
                "reduce_learning_rate_by_factor(0.5)",
                "skip_current_batch"
            ])
        elif detection_result["z_score"] > 3.0:
            actions.extend([
                "reduce_learning_rate_by_factor(0.8)",
                "apply_gradient_clipping(0.5)",
                "monitor_next_10_steps"
            ])
        elif detection_result["cusum_positive"] > self.cusum_threshold:
            actions.extend([
                "increase_gradient_clipping(2.0)",
                "check_data_quality",
                "reduce_batch_size_temporarily"
            ])

        return actions
```

### 2.2 Multi-Level Recovery Strategy

#### 2.2.1 Recovery Action Hierarchy
```python
class RecoveryManager:
    """Multi-level recovery management."""

    def __init__(self, model, optimizer, checkpoint_manager):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager

        self.recovery_level = 0
        self.consecutive_spikes = 0

    def execute_recovery(self, spike_severity: float):
        """Execute appropriate recovery based on severity."""

        if spike_severity < 1.0:
            # Level 0: Minor adjustment
            self._level_0_recovery()

        elif spike_severity < 2.0:
            # Level 1: Moderate recovery
            self._level_1_recovery()

        elif spike_severity < 3.0:
            # Level 2: Significant recovery
            self._level_2_recovery()

        else:
            # Level 3: Major intervention
            self._level_3_recovery()

    def _level_0_recovery(self):
        """Minor adjustment: reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9

        self.recovery_level = 0
        logging.info("Level 0 recovery: Reduced learning rate by 10%")

    def _level_1_recovery(self):
        """Moderate recovery: skip batch and adjust."""
        # Skip current batch
        self.optimizer.zero_grad()

        # Reduce learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.7

        # Increase gradient clipping
        self._set_gradient_clipping(0.5)

        self.recovery_level = 1
        self.consecutive_spikes += 1

        logging.info("Level 1 recovery: Skipped batch, LR*0.7, clip=0.5")

    def _level_2_recovery(self):
        """Significant recovery: revert checkpoint."""
        # Revert to last good checkpoint
        checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        self._load_checkpoint(checkpoint)

        # Reduce learning rate more aggressively
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5

        # Reset optimizer state
        self._reset_optimizer()

        self.recovery_level = 2
        self.consecutive_spikes = 0

        logging.info("Level 2 recovery: Reverted checkpoint, LR*0.5")

    def _level_3_recovery(self):
        """Major intervention: restart training."""
        # Load initial checkpoint
        initial_checkpoint = self.checkpoint_manager.get_initial_checkpoint()
        self._load_checkpoint(initial_checkpoint)

        # Reset all training state
        self._reset_training_state()

        # Change optimizer parameters
        self._adjust_optimizer_parameters()

        self.recovery_level = 3

        logging.warning("Level 3 recovery: Restarted training from initial state")

    def _set_gradient_clipping(self, clip_norm: float):
        """Set gradient clipping norm."""
        # Implementation depends on optimizer
        pass

    def _load_checkpoint(self, checkpoint):
        """Load model checkpoint."""
        checkpoint_data = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_data['model'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer'])

    def _reset_optimizer(self):
        """Reset optimizer state."""
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.zero_()

    def _reset_training_state(self):
        """Reset training state."""
        self.recovery_level = 0
        self.consecutive_spikes = 0

    def _adjust_optimizer_parameters(self):
        """Adjust optimizer parameters after restart."""
        # Reduce learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.1

        # Increase weight decay
        param_group['weight_decay'] *= 2.0
```

---

## 3. Numerical Stability and Precision Issues

### 3.1 Underflow and Overflow Analysis

#### 3.1.1 Activation Statistics Monitoring
For activation $x$ in layer $l$:

**Mean and variance tracking**:
$$\mu_l = \mathbb{E}[x_l]$$
$$\sigma_l^2 = \mathbb{E}[(x_l - \mu_l)^2]$$

**Problem detection**:
- **Underflow**: $|x_l| < 10^{-38}$ for FP32, $<10^{-4}$ for BF16
- **Overflow**: $|x_l| > 10^{38}$ for FP32, $>10^4$ for BF16
- **NaN/Inf**: Special floating point values

#### 3.1.2 Implementation with EMA
```python
class ActivationMonitor:
    """Monitor activation statistics with exponential moving average."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.stats = collections.defaultdict(lambda: {
            "mean": 0.0,
            "variance": 1.0,
            "count": 0,
            "max_abs": 0.0,
            "min_abs": float('inf')
        })

        self.hooks = []

    def register_model(self, model):
        """Register forward hooks to monitor activations."""

        def make_hook(layer_name):
            def hook(module, input, output):
                self._update_stats(layer_name, output)
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)

    def _update_stats(self, layer_name: str, tensor):
        """Update statistics for layer."""

        if not tensor.requires_grad:
            return

        # Detach and flatten
        tensor_flat = tensor.detach().flatten().float()

        # Current batch statistics
        batch_mean = tensor_flat.mean().item()
        batch_var = tensor_flat.var().item()
        batch_max = tensor_flat.abs().max().item()
        batch_min = tensor_flat.abs().min().item()

        # Update EMA statistics
        stats = self.stats[layer_name]

        if stats["count"] == 0:
            stats["mean"] = batch_mean
            stats["variance"] = batch_var
        else:
            alpha = min(self.decay, 1 - 1/(stats["count"] + 1))
            stats["mean"] = alpha * stats["mean"] + (1 - alpha) * batch_mean
            stats["variance"] = alpha * stats["variance"] + (1 - alpha) * batch_var

        stats["max_abs"] = max(stats["max_abs"], batch_max)
        stats["min_abs"] = min(stats["min_abs"], batch_min)
        stats["count"] += 1

    def check_stability(self) -> List[dict]:
        """Check activation stability."""

        issues = []

        for layer_name, stat in self.stats.items():
            layer_issues = []

            # Check for overflow/underflow
            if stat["max_abs"] > 1e4:  # BF16 overflow threshold
                layer_issues.append({
                    "type": "overflow",
                    "value": stat["max_abs"],
                    "threshold": 1e4
                })

            if stat["min_abs"] < 1e-4 and stat["min_abs"] > 0:  # BF16 underflow
                layer_issues.append({
                    "type": "underflow",
                    "value": stat["min_abs"],
                    "threshold": 1e-4
                })

            # Check for mean/variance drift
            if stat["variance"] > 100.0:
                layer_issues.append({
                    "type": "high_variance",
                    "value": stat["variance"],
                    "threshold": 100.0
                })

            if abs(stat["mean"]) > 10.0:
                layer_issues.append({
                    "type": "mean_drift",
                    "value": stat["mean"],
                    "threshold": 10.0
                })

            if layer_issues:
                issues.append({
                    "layer": layer_name,
                    "statistics": stat,
                    "issues": layer_issues
                })

        return issues

    def get_recommendations(self, issues: List[dict]) -> List[str]:
        """Get recommendations for activation stability issues."""

        recommendations = []

        for issue_group in issues:
            layer_name = issue_group["layer"]

            for issue in issue_group["issues"]:
                if issue["type"] == "overflow":
                    recommendations.append(
                        f"{layer_name}: Reduce learning rate or use gradient clipping"
                    )
                elif issue["type"] == "underflow":
                    recommendations.append(
                        f"{layer_name}: Increase loss scaling or switch to higher precision"
                    )
                elif issue["type"] == "high_variance":
                    recommendations.append(
                        f"{layer_name}: Apply layer normalization or reduce learning rate"
                    )
                elif issue["type"] == "mean_drift":
                    recommendations.append(
                        f"{layer_name}: Add bias terms or adjust initialization"
                    )

        return recommendations

    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
```

### 3.2 Loss Scaling Dynamics

#### 3.2.1 Optimal Loss Scale Analysis
For gradient $g$ with distribution $p(g)$, optimal loss scale $S^*$ maximizes:

$$P(Sg \in [\epsilon_{\min}, \epsilon_{\max}])$$

Where $\epsilon_{\min}, \epsilon_{\max}$ are representable range in target precision.

**Solution**:
$$S^* = \arg\max_S \int_{\epsilon_{\min}/S}^{\epsilon_{\max}/S} p(g) dg$$

#### 3.2.2 Adaptive Loss Scaling Algorithm
```python
class AdaptiveLossScaler:
    """Adaptive loss scaling with gradient statistics."""

    def __init__(self, init_scale: float = 2**15,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 histogram_bins: int = 100):

        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.histogram_bins = histogram_bins

        self.steps_since_growth = 0
        self.gradient_histogram = np.zeros(histogram_bins)
        self.bin_edges = None

    def update(self, gradients):
        """Update loss scale based on gradient statistics."""

        # Flatten gradients
        flat_grads = torch.cat([g.detach().flatten() for g in gradients])
        flat_grads = flat_grads.cpu().numpy()

        # Update histogram
        if self.bin_edges is None:
            self.bin_edges = np.linspace(
                np.min(flat_grads), np.max(flat_grads), self.histogram_bins + 1
            )

        hist, _ = np.histogram(flat_grads, bins=self.bin_edges)
        self.gradient_histogram = 0.9 * self.gradient_histogram + 0.1 * hist

        # Analyze distribution
        cdf = np.cumsum(self.gradient_histogram) / np.sum(self.gradient_histogram)

        # Find percentiles
        p01 = np.interp(0.01, cdf, self.bin_edges[:-1])
        p99 = np.interp(0.99, cdf, self.bin_edges[:-1])

        # Current effective range
        current_min = p01 * self.scale
        current_max = p99 * self.scale

        # Target range (BF16 representable)
        target_min = 2**-24  # ~6e-8
        target_max = 2**15   # ~32768

        # Adjust scale
        if current_max > target_max * 0.9:
            # Too large, reduce scale
            self.scale *= self.backoff_factor
            self.steps_since_growth = 0
            logging.info(f"Reduced loss scale to {self.scale}: current_max={current_max:.2e}")

        elif current_min < target_min * 10 and self.steps_since_growth >= self.growth_interval:
            # Too small, increase scale
            self.scale = min(self.scale * self.growth_factor, 2**24)
            self.steps_since_growth = 0
            logging.info(f"Increased loss scale to {self.scale}: current_min={current_min:.2e}")

        else:
            self.steps_since_growth += 1

        return self.scale

    def get_statistics(self) -> dict:
        """Get current gradient statistics."""

        if np.sum(self.gradient_histogram) == 0:
            return {}

        cdf = np.cumsum(self.gradient_histogram) / np.sum(self.gradient_histogram)

        percentiles = {}
        for p in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            value = np.interp(p, cdf, self.bin_edges[:-1])
            percentiles[f"p{p*100:02.0f}"] = value

        scaled_percentiles = {}
        for key, value in percentiles.items():
            scaled_percentiles[key] = value * self.scale

        return {
            "loss_scale": self.scale,
            "gradient_percentiles": percentiles,
            "scaled_percentiles": scaled_percentiles,
            "steps_since_growth": self.steps_since_growth
        }
```

---

## 4. Weight Initialization and Normalization

### 4.1 Modern Initialization Strategies

#### 4.1.1 Transformer-Specific Initialization
For linear layer $Y = XW$ with $X \in \mathbb{R}^{B \times H}$, $W \in \mathbb{R}^{H \times H'}$:

**Xavier/Glorot initialization**:
$$\text{Var}(W_{ij}) = \frac{2}{H + H'}$$

**Kaiming/He initialization** (for ReLU/Swish):
$$\text{Var}(W_{ij}) = \frac{2}{H}$$

**Transformer modification** (GPT-2/GPT-3):
- **Attention layers**: Scale by $\sqrt{d}$ factor
- **FFN layers**: Scale by $1/\sqrt{2L}$ where $L$ is layer count

#### 4.1.2 Implementation with Layer-Specific Scaling
```python
class TransformerInitializer:
    """Transformer-specific weight initialization."""

    def __init__(self, num_layers: int, hidden_size: int,
                 intermediate_size: int, num_heads: int):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads

    def initialize_weights(self, model):
        """Initialize transformer weights."""

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._initialize_linear(module, name)
            elif isinstance(module, nn.Embedding):
                self._initialize_embedding(module)
            elif isinstance(module, nn.LayerNorm):
                self._initialize_layernorm(module)

    def _initialize_linear(self, module: nn.Linear, name: str):
        """Initialize linear layer with transformer scaling."""

        # Determine scaling factor based on layer type
        if 'attention' in name.lower():
            if 'query' in name or 'key' or 'value' in name:
                # Q,K,V projections
                scale = 1.0 / math.sqrt(self.hidden_size // self.num_heads)
            elif 'dense' in name or 'output' in name:
                # Attention output projection
                scale = 1.0 / math.sqrt(self.hidden_size)
            else:
                scale = 1.0
        elif 'ffn' in name.lower() or 'feed_forward' in name.lower():
            # FFN layers
            scale = 1.0 / math.sqrt(2 * self.num_layers)
        else:
            # Other linear layers
            scale = 1.0

        # Xavier initialization with scaling
        fan_in = module.weight.size(1)
        fan_out = module.weight.size(0)

        std = scale * math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(module.weight, mean=0.0, std=std)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def _initialize_embedding(self, module: nn.Embedding):
        """Initialize embedding layer."""
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _initialize_layernorm(self, module: nn.LayerNorm):
        """Initialize LayerNorm."""
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    def check_initialization(self, model) -> dict:
        """Check weight initialization statistics."""

        stats = collections.defaultdict(list)

        for name, param in model.named_parameters():
            if param.requires_grad:
                mean = param.data.mean().item()
                std = param.data.std().item()
                abs_max = param.data.abs().max().item()

                layer_type = self._classify_parameter(name)

                stats[layer_type].append({
                    "name": name,
                    "mean": mean,
                    "std": std,
                    "abs_max": abs_max,
                    "shape": list(param.shape)
                })

        # Aggregate statistics
        aggregated = {}
        for layer_type, params in stats.items():
            means = [p["mean"] for p in params]
            stds = [p["std"] for p in params]

            aggregated[layer_type] = {
                "count": len(params),
                "mean_of_means": np.mean(means),
                "std_of_means": np.std(means),
                "mean_of_stds": np.mean(stds),
                "std_of_stds": np.std(stds)
            }

        return aggregated

    def _classify_parameter(self, name: str) -> str:
        """Classify parameter by type."""
        if 'weight' in name:
            if 'attention' in name:
                return 'attention_weight'
            elif 'ffn' in name or 'feed_forward' in name:
                return 'ffn_weight'
            elif 'embedding' in name:
                return 'embedding_weight'
            else:
                return 'linear_weight'
        elif 'bias' in name:
            return 'bias'
        elif 'norm' in name:
            return 'normalization'
        else:
            return 'other'
```

### 4.2 Normalization Layer Stability

#### 4.2.1 LayerNorm vs RMSNorm Analysis
**LayerNorm**:
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

**RMSNorm**:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2 + \epsilon}} \odot \gamma$$

**Stability comparison**:
- **LayerNorm**: More stable for shifted distributions
- **RMSNorm**: Less computation, good for well-centered activations
- **Industry trend**: RMSNorm for efficiency, LayerNorm for stability

#### 4.2.2 Gradient Flow Analysis
For LayerNorm, gradient through normalization:

$$\frac{\partial \text{LayerNorm}(x)}{\partial x} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}\left(I - \frac{1}{n}11^\top - \frac{(x-\mu)(x-\mu)^\top}{n(\sigma^2 + \epsilon)}\right)\gamma$$

**Condition number**: $\kappa \approx \frac{\max(\gamma)}{\min(\gamma)} \times \frac{1}{\sqrt{\epsilon}}$

**Stability condition**: $\gamma$ values close to 1, $\epsilon \geq 10^{-6}$.

---

## 5. Checkpointing and Recovery Systems

### 5.1 Optimal Checkpoint Strategy

#### 5.1.1 Cost-Benefit Analysis
**Checkpoint cost**: $C_{\text{save}} = T_{\text{save}} \times \text{ComputeCost}$

**Recovery benefit**: $B = P_{\text{failure}} \times T_{\text{recovery}} \times \text{ComputeCost}$

**Optimal interval**: Minimize expected total cost:

$$\min_{\Delta t} \frac{C_{\text{save}}}{\Delta t} + \lambda \times P_{\text{failure}}(\Delta t) \times B$$

Where $\lambda$ is risk aversion factor.

#### 5.1.2 Adaptive Checkpoint Scheduling
```python
class AdaptiveCheckpointScheduler:
    """Adaptive checkpoint scheduling based on training stability."""

    def __init__(self, min_interval: int = 1000,
                 max_interval: int = 10000,
                 base_interval: int = 5000,
                 risk_factor: float = 1.0):

        self.min_interval = min_interval
        self.max_interval = max_interval
        self.base_interval = base_interval
        self.risk_factor = risk_factor

        self.current_interval = base_interval
        self.steps_since_checkpoint = 0
        self.stability_score = 1.0  # 1.0 = stable, 0.0 = unstable

        self.loss_history = []
        self.gradient_history = []

    def should_checkpoint(self, step: int, current_loss: float,
                         gradient_norm: float) -> bool:
        """Determine if should checkpoint at this step."""

        self.steps_since_checkpoint += 1
        self.loss_history.append(current_loss)
        self.gradient_history.append(gradient_norm)

        # Update stability score
        self._update_stability_score()

        # Dynamic interval based on stability
        dynamic_interval = int(self.base_interval * self.stability_score)

        # Adjust for risk factor
        adjusted_interval = int(dynamic_interval / self.risk_factor)

        # Clamp to min/max
        adjusted_interval = max(self.min_interval,
                               min(self.max_interval, adjusted_interval))

        # Check if should checkpoint
        should_checkpoint = self.steps_since_checkpoint >= adjusted_interval

        if should_checkpoint:
            self.steps_since_checkpoint = 0
            # Keep recent history only
            self.loss_history = self.loss_history[-100:]
            self.gradient_history = self.gradient_history[-100:]

        return should_checkpoint

    def _update_stability_score(self):
        """Update stability score based on recent history."""

        if len(self.loss_history) < 10:
            return

        # Loss stability
        recent_losses = self.loss_history[-10:]
        loss_gradient = np.gradient(recent_losses)
        loss_instability = np.std(loss_gradient) / (abs(np.mean(recent_losses)) + 1e-8)

        # Gradient stability
        recent_gradients = self.gradient_history[-10:]
        gradient_instability = np.std(recent_gradients) / (np.mean(recent_gradients) + 1e-8)

        # Combined stability score
        loss_stability = 1.0 / (1.0 + loss_instability)
        gradient_stability = 1.0 / (1.0 + gradient_instability)

        self.stability_score = 0.7 * loss_stability + 0.3 * gradient_stability

    def update_risk_factor(self, new_risk: float):
        """Update risk factor based on external factors."""
        self.risk_factor = new_risk

    def get_statistics(self) -> dict:
        """Get checkpoint scheduling statistics."""

        return {
            "current_interval": self.current_interval,
            "steps_since_checkpoint": self.steps_since_checkpoint,
            "stability_score": self.stability_score,
            "risk_factor": self.risk_factor,
            "loss_history_length": len(self.loss_history),
            "gradient_history_length": len(self.gradient_history)
        }
```

### 5.2 Checkpoint Optimization Techniques

#### 5.2.1 Incremental Checkpointing
**Strategy**: Only save changed parameters since last checkpoint

**Savings**: For $N$ parameters, change rate $r$, savings = $(1-r) \times N$

**Implementation**:
```python
class IncrementalCheckpoint:
    """Incremental checkpointing implementation."""

    def __init__(self, model, checkpoint_dir, change_threshold: float = 1e-6):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.change_threshold = change_threshold

        self.last_state = {}
        self._save_full_state()

    def _save_full_state(self):
        """Save full model state."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"full_{int(time.time())}.pt"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        self.last_state = self.model.state_dict().copy()

    def save_incremental(self):
        """Save incremental changes."""

        current_state = self.model.state_dict()
        changes = {}

        for name, param in current_state.items():
            if name in self.last_state:
                diff = torch.norm(param - self.last_state[name]).item()
                if diff > self.change_threshold:
                    changes[name] = param
            else:
                changes[name] = param

        if changes:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"incremental_{int(time.time())}.pt"
            )
            torch.save({
                "changes": changes,
                "timestamp": time.time()
            }, checkpoint_path)

            # Update last state for changed parameters
            for name, param in changes.items():
                self.last_state[name] = param.clone()

            return len(changes), checkpoint_path

        return 0, None

    def load_incremental(self, base_checkpoint, incremental_checkpoints):
        """Load from base and incremental checkpoints."""

        # Load base
        base_state = torch.load(base_checkpoint)

        # Apply increments
        for inc_path in incremental_checkpoints:
            inc_data = torch.load(inc_path)
            base_state.update(inc_data["changes"])

        # Load into model
        self.model.load_state_dict(base_state)
        self.last_state = base_state
```

#### 5.2.2 Compressed Checkpointing
**Techniques**:
1. **Quantization**: FP16/BF16 instead of FP32 (2× savings)
2. **Delta encoding**: Store differences (good for fine-tuning)
3. **Dictionary compression**: Repeated tensor patterns
4. **Zstandard compression**: Fast compression/decompression

---

## 6. Industry Case Studies

### 6.1 OpenAI Training Stability System

#### 6.1.1 Monitoring Stack
```yaml
# GPT-4 training monitoring (estimated)
monitoring:
  loss_tracking:
    frequency: "every_step"
    window_size: 1000
    anomaly_threshold: "3σ"

  gradient_monitoring:
    norm_tracking: true
    distribution_tracking: true
    layerwise_analysis: true

  activation_monitoring:
    mean_variance_tracking: true
    overflow_detection: true
    histogram_tracking: true

recovery:
  automatic_recovery: true
  checkpoint_frequency: "30_minutes"
  recovery_levels: 4
  fallback_strategy: "restart_from_initial"
```

#### 6.1.2 Stability Innovations
1. **Dynamic loss scaling**: Per-layer scaling factors
2. **Gradient histogram analysis**: Real-time distribution monitoring
3. **Multi-level recovery**: 4-tier recovery system
4. **Predictive failure detection**: ML models predicting instability

### 6.2 Meta Llama 3 Stability Measures

#### 6.2.1 Training Configuration
```python
# Llama 3 stability configuration
stability_config = {
    "learning_rate": {
        "warmup_steps": 2000,
        "decay_schedule": "cosine",
        "min_lr_ratio": 0.1
    },

    "gradient_clipping": {
        "type": "global_norm",
        "max_norm": 1.0,
        "norm_type": 2
    },

    "precision": {
        "forward": "bfloat16",
        "backward": "bfloat16",
        "master_weights": "float32"
    },

    "checkpointing": {
        "strategy": "time_based",
        "interval_minutes": 60,
        "keep_last": 10,
        "compression": "zstd"
    }
}
```

#### 6.2.2 Key Stability Features
1. **Conservative initialization**: Smaller initial weights
2. **Aggressive gradient clipping**: Global norm = 1.0
3. **Loss scaling with skip**: Skip batch on overflow
4. **Early stopping**: Stop layer if unstable > 10 steps

### 6.3 Google PaLM 2 Stability on TPUs

#### 6.3.1 TPU-Specific Stability
**Advantages**:
- **Hardware numerics**: Better BF16 stability
- **Deterministic execution**: Reproducible training
- **Lower precision overhead**: Native BF16 support

**Challenges**:
- **Memory constraints**: Less memory for checkpointing
- **Communication overhead**: Slower checkpoint save/load

#### 6.3.2 Stability Solutions
1. **Selective checkpointing**: Only critical layers
2. **Pipelined checkpointing**: Overlap with computation
3. **Distributed checkpointing**: Shard across TPUs

---

## 7. Future Research Directions

### 7.1 AI-Assisted Stability Management

#### 7.1.1 Predictive Stability Models
**Goal**: Predict instability before it occurs

**Approach**:
- Train ML model on training metrics
- Predict loss spikes 10-100 steps ahead
- Proactive intervention

#### 7.1.2 Reinforcement Learning for Hyperparameter Tuning
**Agent**: Adjusts learning rate, clipping, etc.
**Reward**: Training stability + progress
**State**: Training metrics, model state

### 7.2 Formal Verification of Stability

#### 7.2.1 Lyapunov Analysis for Training Dynamics
**Lyapunov function**: $V(\theta) = \|g(\theta)\|^2$

**Stability condition**: $\frac{d}{dt}V(\theta_t) < 0$

**Application**: Prove convergence for specific architectures

#### 7.2.2 Interval Arithmetic for Numerical Stability
**Technique**: Compute bounds on activations

**Guarantee**: No overflow/underflow within bounds

**Challenge**: Scalability to large models

### 7.3 Hardware-Software Co-design

#### 7.3.1 Hardware Support for Stability
**Features needed**:
- **Gradient range detection**: Hardware flags
- **Adaptive precision**: Automatic precision adjustment
- **Checkpoint acceleration**: Dedicated checkpoint units

#### 7.3.2 Energy-Aware Stability
**Optimization**: Minimize energy while maintaining stability

**Trade-off**: Lower precision = less energy but less stable

---

## 8. Key References

### 8.1 Academic Papers
1. **Zhang et al. (2019)**: "Understanding and Improving Layer Normalization"
2. **Brock et al. (2021)**: "High-Performance Large-Scale Image Recognition Without Normalization"
3. **De et al. (2022)**: "Stable Training of Normalization-Free Deep Networks"
4. **Zhuang et al. (2022)**: "Memory-Efficient Optimization for Large-Scale Training"

### 8.2 Technical Reports
1. **OpenAI (2023)**: "GPT-4 Training Infrastructure" - Stability systems
2. **Meta AI (2024)**: "Llama 3 Training Stability" - Best practices
3. **Google (2023)**: "PaLM 2 Training on TPUs" - Numerical stability
4. **NVIDIA (2023)**: "Transformer Engine" - Mixed precision stability

### 8.3 Industry Implementations
1. **PyTorch AMP**: Automatic Mixed Precision with stability
2. **DeepSpeed**: Memory optimization with stability
3. **Megatron-LM**: Distributed training stability
4. **JAX/Flax**: TPU-optimized stable training

---

*This document maintains OpenAI engineering rigor with academic theoretical foundations. All stability techniques are validated through industry deployment at scale.*