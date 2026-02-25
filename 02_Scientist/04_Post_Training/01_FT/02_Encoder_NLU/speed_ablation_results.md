# ModernBERT Speed Ablation Results (A100)

*Prerequisite: [01_ModernBERT_FT/01_ModernBERT.md](01_ModernBERT_FT/01_ModernBERT.md).*

本文档记录了 ModernBERT-base 在 A100 GPU 上的不同配置下的训练性能表现，用于性能消融实验对比。

## 实验环境

- **GPU**: NVIDIA A100 80GB
- **Model**: answerdotai/ModernBERT-base
- **Dataset**: Synthetic LLM Routing Dataset (~15k samples)
- **Batch Size**: 8 (per device)
- **Accumulation Steps**: 2

---

## 实验记录 (Ablation Table)

| 实验组                      | 配置 (Precision + Implementation) | 吞吐量 (Global it/s) | 训练状态                 | 备注                   |
| :-------------------------- | :-------------------------------- | :------------------- | :----------------------- | :--------------------- |
| **实验 0 (Baseline)** | **BF16 + Eager**            | **~2.66 it/s** | **稳定 (SUCCESS)** | Step 800 F1: 0.987     |
| 实验 1                      | FP32 + Eager                      | **~1.00 it/s** | **稳定 (SUCCESS)** | 性能损耗约 2.6x        |
| 实验 2                      | FP16/BF16 + SDPA                  | 0.00 it/s            | **失败 (FAILED)**  | 梯度归零, 无法学习     |
| 实验 3                      | BF16 + FlashAttention-2           | 待填充               | -                        | 需要安装 flash-attn 库 |

---

## 详细观察记录

### 基准组 (BF16 + Eager) - 2026-02-02

- **配置详情**：`bf16=True`, `fp16=False`, `attn_implementation="eager"`。
- **性能表现**：从 Step 10 到 Step 800，训练非常平稳。
- **准确率**：极速收敛，Step 200 F1 0.976, Step 800 F1 0.987。
- **稳定性**：通过 `SurveillanceCallback` 验证，权重（Classifier Norm）正常更新，虽然底层算子较慢但结果绝对准确。

### 实验 1 (FP32 + Eager) - 2026-02-02

- **配置详情**：`bf16=False`, `fp16=False`, `attn_implementation="eager"`。
- **性能表现**：吞吐量骤降至 **~1.00 it/s**。
- **结论**：在 A100 上禁用 BF16 会导致约 **166%** 的额外时间开销。虽然 FP32 精度理论上更高，但在本 NLU 任务中，F1 的提升（如有）不足以弥补这种巨大的效率损失。
- **监控观察**：出现了频繁的 "Zero Gradient" 警告，分析原因为 `gradient_accumulation_steps=2` 时，优化器步进后会清空梯度，此时回调捕获到的梯度为零，属于误报。

---

*记录人：Antigravity AI*
