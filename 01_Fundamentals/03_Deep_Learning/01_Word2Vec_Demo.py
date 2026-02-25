"""
01_Word2Vec_Demo.py — Word Embedding Training & Visualization

Demonstrates:
1. Training Word2Vec (Skip-gram) on a small corpus using Gensim
2. Exploring word similarity and analogy (king - man + woman = queen)
3. Visualizing embeddings in 2D via t-SNE

Prerequisites: pip install gensim matplotlib scikit-learn
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Corpus & Training
# ---------------------------------------------------------------------------

# Small demo corpus — in production, use Wikipedia / BookCorpus
corpus = [
    "the king ruled the kingdom with wisdom",
    "the queen ruled the kingdom with grace",
    "the prince trained with the knight in the castle",
    "the princess studied language and science",
    "a man and a woman walked through the village",
    "the knight defended the castle from invaders",
    "wisdom and grace are virtues of a good ruler",
    "the kingdom prospered under fair rule",
    "science and language open doors to knowledge",
    "the village celebrated the harvest festival",
    "a king and queen attended the royal banquet",
    "the prince and princess explored the forest",
    "knowledge is power said the wise scholar",
    "the castle stood tall above the green valley",
    "the knight rode his horse across the kingdom",
]

tokenized = [sentence.split() for sentence in corpus]

try:
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=tokenized,
        vector_size=50,      # embedding dimension
        window=3,            # context window
        min_count=1,         # include all words
        sg=1,                # 1 = Skip-gram, 0 = CBOW
        epochs=200,          # more epochs for small corpus
        seed=42,
    )

    # ------------------------------------------------------------------
    # 2. Similarity & Analogy
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Word2Vec Demo — Similarity & Analogy")
    print("=" * 60)

    print("\n[Most similar to 'king']:")
    for word, score in model.wv.most_similar("king", topn=5):
        print(f"  {word:15s} {score:.4f}")

    print("\n[Most similar to 'castle']:")
    for word, score in model.wv.most_similar("castle", topn=5):
        print(f"  {word:15s} {score:.4f}")

    # Analogy: king - man + woman ≈ queen
    print("\n[Analogy] king - man + woman = ?")
    try:
        results = model.wv.most_similar(
            positive=["king", "woman"], negative=["man"], topn=3
        )
        for word, score in results:
            print(f"  {word:15s} {score:.4f}")
    except KeyError as e:
        print(f"  Word not in vocabulary: {e}")

    # ------------------------------------------------------------------
    # 3. t-SNE Visualization
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        words = list(model.wv.key_to_index.keys())
        vectors = np.array([model.wv[w] for w in words])

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words) - 1))
        coords = tsne.fit_transform(vectors)

        plt.figure(figsize=(12, 8))
        plt.scatter(coords[:, 0], coords[:, 1], s=40, alpha=0.6)
        for i, word in enumerate(words):
            plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9)
        plt.title("Word2Vec Embeddings — t-SNE Projection")
        plt.tight_layout()

        out_path = "word2vec_tsne.png"
        plt.savefig(out_path, dpi=150)
        print(f"\n[Visualization saved to {out_path}]")

    except ImportError:
        print("\n[Skipping visualization — install matplotlib & scikit-learn]")

except ImportError:
    print("Gensim not installed. Run: pip install gensim")
    print("\nFalling back to manual Skip-gram illustration...\n")

    # ------------------------------------------------------------------
    # Fallback: Pure NumPy Skip-gram (educational)
    # ------------------------------------------------------------------
    vocab = sorted(set(w for s in tokenized for w in s))
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    D = 10  # small dimension for demo

    np.random.seed(42)
    W_in = np.random.randn(V, D) * 0.1   # input embeddings
    W_out = np.random.randn(D, V) * 0.1   # output embeddings

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    lr = 0.05
    window = 2

    for epoch in range(50):
        loss = 0.0
        for sentence in tokenized:
            for i, target in enumerate(sentence):
                for j in range(max(0, i - window), min(len(sentence), i + window + 1)):
                    if i == j:
                        continue
                    context = sentence[j]
                    t_idx = w2i[target]
                    c_idx = w2i[context]

                    hidden = W_in[t_idx]              # (D,)
                    scores = hidden @ W_out            # (V,)
                    probs = softmax(scores)            # (V,)

                    loss -= np.log(probs[c_idx] + 1e-9)

                    # Gradient
                    grad_out = probs.copy()
                    grad_out[c_idx] -= 1.0            # (V,)
                    W_out -= lr * np.outer(hidden, grad_out)
                    W_in[t_idx] -= lr * (W_out @ grad_out)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}  Loss: {loss:.2f}")

    # Cosine similarity
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    print("\n[Cosine similarities from pure NumPy Skip-gram]:")
    for pair in [("king", "queen"), ("king", "castle"), ("man", "woman")]:
        if pair[0] in w2i and pair[1] in w2i:
            sim = cosine(W_in[w2i[pair[0]]], W_in[w2i[pair[1]]])
            print(f"  {pair[0]:10s} ↔ {pair[1]:10s}  sim = {sim:.4f}")

if __name__ == "__main__":
    pass
