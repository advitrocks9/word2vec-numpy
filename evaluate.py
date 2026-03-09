"""Evaluation utilities: nearest neighbours, word analogies."""

from __future__ import annotations

import os
import urllib.request

import numpy as np
import numpy.typing as npt

from word2vec.vocab import Vocab

# ---------------------------------------------------------------------------
# Nearest neighbours
# ---------------------------------------------------------------------------

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
    """Find the *top_k* nearest words by cosine similarity.

    Args:
        word: Query word.
        W_in: Center embedding matrix, shape ``(V, d)``.
        vocab: Vocabulary for ID ↔ string conversion.
        top_k: Number of neighbours to return.

    Returns:
        List of ``(word, similarity)`` tuples sorted by descending similarity.
    """
    idx = vocab.word_to_idx.get(word)
    if idx is None:
        return []

    norms = np.linalg.norm(W_in, axis=1, keepdims=True)  # (V, 1)
    W_normed = W_in / np.maximum(norms, 1e-8)

    sims: npt.NDArray[np.float64] = W_normed @ W_normed[idx]  # (V,)
    sims[idx] = -np.inf  # exclude the query word itself

    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [(vocab.idx_to_word[int(i)], float(sims[i])) for i in top_indices]


def print_nearest_neighbors(
    W_in: npt.NDArray[np.float64],
    vocab: Vocab,
    words: list[str] | None = None,
    top_k: int = 10,
) -> None:
    """Print a formatted table of nearest neighbours for probe words.

    Args:
        W_in: Center embedding matrix, shape ``(V, d)``.
        vocab: Vocabulary instance.
        words: Words to query (defaults to :data:`PROBE_WORDS`).
        top_k: Neighbours per word.
    """
    if words is None:
        words = PROBE_WORDS

    for word in words:
        neighbors = nearest_neighbors(word, W_in, vocab, top_k)
        if not neighbors:
            print(f"  '{word}' not in vocabulary")
            continue
        neighbour_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbors)
        print(f"  {word:12s} -> {neighbour_str}")


# ---------------------------------------------------------------------------
# Word analogy task
# ---------------------------------------------------------------------------

_ANALOGY_URL = (
    "https://raw.githubusercontent.com/tmikolov/word2vec/"
    "master/questions-words.txt"
)


def _download_analogies(path: str = "questions-words.txt") -> str:
    """Download Google's analogy test set if it doesn't exist locally.

    Args:
        path: Local destination.

    Returns:
        The local file path.
    """
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

    For each analogy quadruple ``(a, b, c, d)``, computes
    ``argmax_{w ∉ {a,b,c}} cos(w, b − a + c)`` and checks whether the
    result equals *d*.

    Args:
        W_in: Center embedding matrix, shape ``(V, d)``.
        vocab: Vocabulary instance.
        questions_path: Path to the analogies file.

    Returns:
        Dict mapping category name → ``(correct, total)`` counts.
    """
    questions_path = _download_analogies(questions_path)

    # Normalise embeddings once
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
                # Save previous category
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

            # Skip if any word is OOV
            if any(w not in vocab.word_to_idx for w in (a, b, c, d)):
                continue

            ia, ib, ic, id_ = (vocab.word_to_idx[w] for w in (a, b, c, d))

            # v = b - a + c
            query = W_normed[ib] - W_normed[ia] + W_normed[ic]
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query /= query_norm

            sims: npt.NDArray[np.float64] = W_normed @ query  # (V,)

            # Exclude input words
            sims[ia] = -np.inf
            sims[ib] = -np.inf
            sims[ic] = -np.inf

            predicted = int(np.argmax(sims))
            if predicted == id_:
                correct += 1
            total += 1

    # Save last category
    if current_category and total > 0:
        results[current_category] = (correct, total)

    return results


def print_analogy_results(results: dict[str, tuple[int, int]]) -> None:
    """Print a formatted analogy accuracy report.

    Args:
        results: Dict from :func:`word_analogy`.
    """
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


def plot_tsne(
    W_in: npt.NDArray[np.float64],
    vocab: Vocab,
    top_n: int = 500,
    path: str = "results/tsne.png",
) -> None:
    """Placeholder for t-SNE visualization."""
    pass


def plot_loss_curve(
    losses: list[float],
    path: str = "results/loss_curve.png",
    log_interval: int = 1000,
) -> None:
    """Placeholder for loss curve plotting."""
    pass
