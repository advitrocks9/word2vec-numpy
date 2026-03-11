"""Evaluation utilities: nearest neighbours, word analogies, t-SNE, loss curve."""

import os
import urllib.request

import numpy as np
import numpy.typing as npt

from word2vec.vocab import Vocab

PROBE_WORDS: list[str] = [
    "king", "queen", "computer", "science",
    "france", "paris", "good", "bad", "river", "music",
]


def nearest_neighbors(
    word: str,
    W_in: npt.NDArray[np.float64],
    vocab: Vocab,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return the *top_k* nearest words by cosine similarity."""
    idx = vocab.word_to_idx.get(word)
    if idx is None:
        return []

    norms = np.linalg.norm(W_in, axis=1, keepdims=True)
    W_normed = W_in / np.maximum(norms, 1e-8)

    sims = W_normed @ W_normed[idx]
    sims[idx] = -np.inf

    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [(vocab.idx_to_word[int(i)], float(sims[i])) for i in top_indices]


def print_nearest_neighbors(
    W_in: npt.NDArray[np.float64],
    vocab: Vocab,
    words: list[str] | None = None,
    top_k: int = 10,
) -> None:
    """Print nearest neighbours for each word in *words*."""
    if words is None:
        words = PROBE_WORDS

    for word in words:
        neighbors = nearest_neighbors(word, W_in, vocab, top_k)
        if not neighbors:
            print(f"  '{word}' not in vocabulary")
            continue
        print(f"  {word:12s} -> {', '.join(f'{w} ({s:.3f})' for w, s in neighbors)}")


_ANALOGY_URL = (
    "https://raw.githubusercontent.com/tmikolov/word2vec/"
    "master/questions-words.txt"
)


def _download_analogies(path: str = "questions-words.txt") -> str:
    """Download Google's analogy test set if not present locally."""
    if not os.path.exists(path):
        print(f"Downloading analogy test set to {path} ...")
        urllib.request.urlretrieve(_ANALOGY_URL, path)  # noqa: S310
    return path


def word_analogy(
    W_in: npt.NDArray[np.float64],
    vocab: Vocab,
    questions_path: str = "questions-words.txt",
) -> dict[str, tuple[int, int]]:
    """Evaluate word analogies (a : b :: c : ?).

    For each quadruple ``(a, b, c, d)``, finds the word closest to
    ``b - a + c`` (excluding the query words) and checks if it equals *d*.

    Returns a dict mapping category name to ``(correct, total)`` counts.
    """
    questions_path = _download_analogies(questions_path)

    norms = np.linalg.norm(W_in, axis=1, keepdims=True)
    W_normed = W_in / np.maximum(norms, 1e-8)

    results: dict[str, tuple[int, int]] = {}
    current_category = ""
    correct = 0
    total = 0

    with open(questions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                if current_category and total > 0:
                    results[current_category] = (correct, total)
                current_category = line[2:]
                correct = 0
                total = 0
                continue

            parts = line.lower().split()
            if len(parts) != 4:
                continue
            a, b, c, d = parts

            if any(w not in vocab.word_to_idx for w in (a, b, c, d)):
                continue

            ia, ib, ic, id_target = (vocab.word_to_idx[w] for w in (a, b, c, d))

            query = W_normed[ib] - W_normed[ia] + W_normed[ic]
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query /= query_norm

            sims = W_normed @ query
            sims[ia] = -np.inf
            sims[ib] = -np.inf
            sims[ic] = -np.inf

            if int(np.argmax(sims)) == id_target:
                correct += 1
            total += 1

    if current_category and total > 0:
        results[current_category] = (correct, total)

    return results


def print_analogy_results(results: dict[str, tuple[int, int]]) -> None:
    """Print a formatted analogy accuracy report."""
    overall_correct = 0
    overall_total = 0

    print(f"  {'Category':<30s}  {'Correct':>7s} / {'Total':>5s}  {'Acc':>6s}")
    print("  " + "-" * 56)
    for cat, (c, t) in sorted(results.items()):
        acc = c / t * 100 if t > 0 else 0.0
        print(f"  {cat:<30s}  {c:>7d} / {t:>5d}  {acc:>5.1f}%")
        overall_correct += c
        overall_total += t

    if overall_total > 0:
        overall_acc = overall_correct / overall_total * 100
        print("  " + "-" * 56)
        print(f"  {'OVERALL':<30s}  {overall_correct:>7d} / {overall_total:>5d}  {overall_acc:>5.1f}%")


_WORDSIM353_URL = (
    "https://raw.githubusercontent.com/benathi/word2gm/"
    "master/evaluation_data/wordsim353/combined.tab"
)
_SIMLEX999_URL = (
    "https://raw.githubusercontent.com/benathi/word2gm/"
    "master/evaluation_data/SimLex-999/SimLex-999.txt"
)


def _download_file(url: str, path: str) -> str:
    """Download a file if it doesn't exist locally."""
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)  # noqa: S310
    return path


def _spearman_correlation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> float:
    """Spearman rank correlation between two 1-D arrays (pure NumPy)."""
    def _rankdata(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
        return ranks

    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def _load_similarity_dataset(
    path: str,
    word1_col: int,
    word2_col: int,
    score_col: int,
) -> list[tuple[str, str, float]]:
    """Parse a tab-separated word-similarity dataset (skip header, lowercase)."""
    pairs: list[tuple[str, str, float]] = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(word1_col, word2_col, score_col):
                continue
            pairs.append((
                parts[word1_col].lower(),
                parts[word2_col].lower(),
                float(parts[score_col]),
            ))
    return pairs


def word_similarity(
    W_in: npt.NDArray[np.float64],
    vocab: Vocab,
    dataset: str = "wordsim353",
) -> dict[str, object]:
    """Evaluate embeddings on a word similarity benchmark.

    Args:
        W_in: Center embedding matrix, shape ``(V, d)``.
        vocab: Vocabulary instance.
        dataset: One of ``"wordsim353"`` or ``"simlex999"``.

    Returns:
        Dict with ``dataset``, ``spearman``, ``covered``, ``total``, ``oov_pairs``.
    """
    configs = {
        "wordsim353": (_WORDSIM353_URL, "wordsim353.tab", 0, 1, 2),
        "simlex999": (_SIMLEX999_URL, "SimLex-999.txt", 0, 1, 3),
    }
    if dataset not in configs:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {list(configs)}")

    url, filename, w1_col, w2_col, score_col = configs[dataset]
    _download_file(url, filename)
    pairs = _load_similarity_dataset(filename, w1_col, w2_col, score_col)
    total = len(pairs)

    norms = np.linalg.norm(W_in, axis=1, keepdims=True)
    W_normed = W_in / np.maximum(norms, 1e-8)

    human_scores: list[float] = []
    model_scores: list[float] = []

    for w1, w2, score in pairs:
        idx1 = vocab.word_to_idx.get(w1)
        idx2 = vocab.word_to_idx.get(w2)
        if idx1 is None or idx2 is None:
            continue
        human_scores.append(score)
        model_scores.append(float(W_normed[idx1] @ W_normed[idx2]))

    covered = len(human_scores)
    oov_pairs = total - covered

    if covered < 2:
        spearman = 0.0
    else:
        spearman = _spearman_correlation(
            np.array(human_scores), np.array(model_scores)
        )

    return {
        "dataset": dataset,
        "spearman": spearman,
        "covered": covered,
        "total": total,
        "oov_pairs": oov_pairs,
    }


def print_similarity_results(results: list[dict[str, object]]) -> None:
    """Print a formatted word-similarity evaluation report."""
    print(f"  {'Dataset':<15s}  {'Spearman':>8s}  {'Covered':>7s} / {'Total':>5s}  {'OOV':>5s}")
    print("  " + "-" * 48)
    for r in results:
        print(
            f"  {r['dataset']:<15s}  {r['spearman']:>8.4f}"
            f"  {r['covered']:>7d} / {r['total']:>5d}  {r['oov_pairs']:>5d}"
        )


def plot_tsne(
    W_in: npt.NDArray[np.float64],
    vocab: Vocab,
    top_n: int = 500,
    path: str = "results/tsne.png",
) -> None:
    """Create a t-SNE scatter plot of the most frequent word embeddings.

    Requires ``scikit-learn`` and ``matplotlib``.
    """
    try:
        from sklearn.manifold import TSNE  # type: ignore[import-untyped]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Skipping t-SNE plot (missing dependency: {exc})")
        return

    n = min(top_n, vocab.vocab_size)
    embeddings = W_in[:n]
    words = [vocab.idx_to_word[i] for i in range(n)]

    print(f"  Running t-SNE on top {n} words ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    label_words = {
        "king", "queen", "man", "woman", "prince", "princess",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "january", "february", "march", "april", "may", "june",
        "france", "germany", "italy", "spain", "japan", "china", "india", "england",
        "paris", "london", "berlin", "rome", "tokyo",
        "war", "peace", "history", "science", "music", "art",
        "good", "bad", "great", "new", "old", "small", "large",
        "water", "river", "city", "world", "country",
    }

    categories = {
        "red": {"king", "queen", "man", "woman", "prince", "princess"},
        "blue": {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"},
        "green": {"january", "february", "march", "april", "may", "june"},
        "orange": {"france", "germany", "italy", "spain", "japan", "china", "india", "england"},
        "purple": {"paris", "london", "berlin", "rome", "tokyo"},
        "brown": {"good", "bad", "great", "new", "old", "small", "large"},
    }
    category_colors = {w: c for c, words in categories.items() for w in words}

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.3, c="grey")

    for i, word in enumerate(words):
        if word in label_words:
            ax.annotate(
                word,
                xy=(coords[i, 0], coords[i, 1]),
                fontsize=7,
                color=category_colors.get(word, "black"),
                alpha=0.85,
            )

    ax.set_title("t-SNE of word2vec embeddings (top-500 words)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved t-SNE plot to {path}")


def plot_loss_curve(
    losses: list[float],
    path: str = "results/loss_curve.png",
    log_interval: int = 1000,
) -> None:
    """Plot and save the training loss curve.

    Args:
        losses: Smoothed loss values recorded every *log_interval* steps.
        path: Output image path.
        log_interval: Steps between each recorded loss value.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Skipping loss curve (matplotlib not installed)")
        return

    steps = [i * log_interval for i in range(len(losses))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, linewidth=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Smoothed loss (EMA)")
    ax.set_title("SGNS Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved loss curve to {path}")
