"""
01_Attention_Visualization.py — Scaled Dot-Product Attention from Scratch

Demonstrates:
1. Manual computation of Q, K, V and attention weights
2. Masking (causal / padding)
3. Heatmap visualization of attention patterns

Prerequisites: pip install matplotlib numpy
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Scaled Dot-Product Attention (Pure NumPy)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (seq_q, d_k)
        K: (seq_k, d_k)
        V: (seq_k, d_v)
        mask: (seq_q, seq_k) — True means MASKED (ignored)
    Returns:
        output: (seq_q, d_v)
        weights: (seq_q, seq_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # (seq_q, seq_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Softmax along last axis
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    output = weights @ V                       # (seq_q, d_v)
    return output, weights


# ---------------------------------------------------------------------------
# 2. Demo: Sentence Attention
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Scaled Dot-Product Attention — From Scratch")
    print("=" * 60)

    # Simulated token embeddings for "The cat sat on the mat"
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)
    d_model = 8

    np.random.seed(42)
    X = np.random.randn(seq_len, d_model)     # token embeddings

    # Learnable projection matrices (random init for demo)
    W_Q = np.random.randn(d_model, d_model) * 0.1
    W_K = np.random.randn(d_model, d_model) * 0.1
    W_V = np.random.randn(d_model, d_model) * 0.1

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # --- 2a. Full attention (bidirectional, encoder-style) ---
    output_full, weights_full = scaled_dot_product_attention(Q, K, V)

    print("\n[Full Attention Weights]")
    print(f"{'':>6s}", end="")
    for t in tokens:
        print(f"{t:>6s}", end="")
    print()
    for i, t in enumerate(tokens):
        print(f"{t:>6s}", end="")
        for j in range(seq_len):
            print(f"{weights_full[i, j]:6.3f}", end="")
        print()

    # --- 2b. Causal attention (decoder-style) ---
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    print("\n[Causal (Masked) Attention Weights]")
    print(f"{'':>6s}", end="")
    for t in tokens:
        print(f"{t:>6s}", end="")
    print()
    for i, t in enumerate(tokens):
        print(f"{t:>6s}", end="")
        for j in range(seq_len):
            print(f"{weights_causal[i, j]:6.3f}", end="")
        print()

    # --- 2c. Multi-Head Attention ---
    n_heads = 2
    d_k = d_model // n_heads

    print(f"\n[Multi-Head Attention] heads={n_heads}, d_k={d_k}")
    all_head_outputs = []
    for h in range(n_heads):
        W_Qh = np.random.randn(d_model, d_k) * 0.1
        W_Kh = np.random.randn(d_model, d_k) * 0.1
        W_Vh = np.random.randn(d_model, d_k) * 0.1

        Qh = X @ W_Qh
        Kh = X @ W_Kh
        Vh = X @ W_Vh

        out_h, w_h = scaled_dot_product_attention(Qh, Kh, Vh)
        all_head_outputs.append(out_h)

        print(f"\n  Head {h} attention (top pattern):")
        for i in range(seq_len):
            top_j = np.argmax(w_h[i])
            print(f"    {tokens[i]:>5s} → {tokens[top_j]:<5s} ({w_h[i, top_j]:.3f})")

    concat = np.concatenate(all_head_outputs, axis=-1)  # (seq_len, d_model)
    W_O = np.random.randn(d_model, d_model) * 0.1
    mha_output = concat @ W_O
    print(f"\n  MHA output shape: {mha_output.shape}")

    # --- 3. Visualization ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, weights, title in [
            (axes[0], weights_full, "Full (Bidirectional) Attention"),
            (axes[1], weights_causal, "Causal (Masked) Attention"),
        ]:
            im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45)
            ax.set_yticklabels(tokens)
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
            ax.set_title(title)

            for i in range(seq_len):
                for j in range(seq_len):
                    ax.text(j, i, f"{weights[i,j]:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if weights[i,j] > 0.5 else "black")

        fig.colorbar(im, ax=axes, shrink=0.8)
        plt.tight_layout()

        out_path = "attention_heatmap.png"
        plt.savefig(out_path, dpi=150)
        print(f"\n[Heatmap saved to {out_path}]")

    except ImportError:
        print("\n[Skipping visualization — install matplotlib]")


if __name__ == "__main__":
    main()
