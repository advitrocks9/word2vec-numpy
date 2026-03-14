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
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
        sorted_arr = arr[order]
        i = 0
        while i < len(sorted_arr):
            j = i + 1
            while j < len(sorted_arr) and sorted_arr[j] == sorted_arr[i]:
                j += 1
            if j > i + 1:
                avg_rank = 0.5 * (i + 1 + j)
                ranks[order[i:j]] = avg_rank
            i = j
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
) -> dict[str, str | int | float]:
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


def print_similarity_results(results: list[dict[str, str | int | float]]) -> None:
    """Print a formatted word-similarity evaluation report."""
    print(f"  {'Dataset':<15s}  {'Spearman':>8s}  {'Covered':>7s} / {'Total':>5s}  {'OOV':>5s}")
    print("  " + "-" * 48)
    for r in results:
        print(
            f"  {r['dataset']:<15s}  {r['spearman']:>8.4f}"
            f"  {r['covered']:>7d} / {r['total']:>5d}  {r['oov_pairs']:>5d}"
        )


def _setup_plot_style() -> None:
    """Apply a clean plot style for all figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def plot_tsne(
    W_in: npt.NDArray[np.float64],
    vocab: Vocab,
    top_n: int = 500,
    path: str = "results/tsne.png",
) -> None:
    """t-SNE projection of frequent word embeddings, coloured by semantic category."""
    try:
        from sklearn.manifold import TSNE  # type: ignore[import-untyped]
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Skipping t-SNE plot (missing dependency: {exc})")
        return

    _setup_plot_style()

    n = min(top_n, vocab.vocab_size)
    embeddings = W_in[:n]
    words = [vocab.idx_to_word[i] for i in range(n)]

    print(f"  Running t-SNE on top {n} words ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    categories: dict[str, tuple[str, list[str]]] = {
        "Royalty":    ("#d62728", ["king", "queen", "prince", "princess", "emperor"]),
        "Numbers":    ("#1f77b4", ["one", "two", "three", "four", "five", "six", "seven", "eight"]),
        "Months":     ("#2ca02c", ["january", "february", "march", "april", "may", "june"]),
        "Countries":  ("#ff7f0e", ["france", "germany", "italy", "spain", "japan", "china"]),
        "Cities":     ("#9467bd", ["paris", "london", "berlin", "rome", "tokyo"]),
        "Adjectives": ("#8c564b", ["good", "bad", "great", "small", "large", "old", "new"]),
        "Science":    ("#17becf", ["science", "physics", "mathematics", "theory", "research"]),
        "Music":      ("#e377c2", ["music", "song", "band", "rock", "jazz"]),
    }

    word_to_cat: dict[str, tuple[str, str]] = {}
    for cat_name, (colour, cat_words) in categories.items():
        for w in cat_words:
            word_to_cat[w] = (cat_name, colour)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.15, c="#cccccc", zorder=1)

    plotted_cats: dict[str, bool] = {}
    texts = []
    text_coords = []
    text_colors = []

    for i, word in enumerate(words):
        if word in word_to_cat:
            cat_name, colour = word_to_cat[word]
            label = cat_name if cat_name not in plotted_cats else None
            ax.scatter(coords[i, 0], coords[i, 1], s=40, c=colour,
                       edgecolors="white", linewidths=0.3, zorder=3, label=label)
            plotted_cats[cat_name] = True
            texts.append(word)
            text_coords.append((coords[i, 0], coords[i, 1]))
            text_colors.append(colour)

    from adjustText import adjust_text as _adjust
    annotations = [
        ax.text(x, y, t, fontsize=8, color=c, fontweight="bold", zorder=4)
        for t, (x, y), c in zip(texts, text_coords, text_colors)
    ]
    _adjust(annotations, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    ax.set_title("t-SNE projection of word2vec embeddings (text8, 200d)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved t-SNE plot to {path}")


def plot_loss_curve(
    losses: list[float],
    path: str = "results/loss_curve.png",
    log_interval: int = 1000,
    lr_start: float = 0.025,
    lr_end: float = 0.0001,
    total_steps: int = 0,
    epochs: int = 20,
) -> None:
    """Plot training loss with learning rate overlay and epoch markers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Skipping loss curve (matplotlib not installed)")
        return

    _setup_plot_style()

    steps = np.array([i * log_interval for i in range(len(losses))])
    if total_steps == 0:
        total_steps = int(steps[-1]) + log_interval

    fig, ax1 = plt.subplots(figsize=(10, 5))

    loss_colour = "#1f77b4"
    ax1.plot(steps, losses, linewidth=0.9, color=loss_colour, zorder=2)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Smoothed loss (EMA)", color=loss_colour)
    ax1.tick_params(axis="y", labelcolor=loss_colour)

    lr_colour = "#ff7f0e"
    ax2 = ax1.twinx()
    lr_values = lr_start + (lr_end - lr_start) * steps / total_steps
    ax2.plot(steps, lr_values, linewidth=1.2, color=lr_colour, linestyle="--", alpha=0.8, zorder=1)
    ax2.set_ylabel("Learning rate", color=lr_colour)
    ax2.tick_params(axis="y", labelcolor=lr_colour)

    if epochs > 0:
        steps_per_epoch = total_steps / epochs
        for e in range(1, epochs):
            boundary = e * steps_per_epoch
            ax1.axvline(boundary, color="#aaaaaa", linestyle=":", linewidth=0.6, zorder=0)
            ax1.text(boundary, ax1.get_ylim()[1], f" {e}", fontsize=7, color="#888888",
                     va="bottom", ha="center")

    ax1.set_title("SGNS Training Loss")
    ax1.grid(True, alpha=0.2)
    ax2.grid(False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved loss curve to {path}")


_SEMANTIC_CATEGORIES = {
    "capital-common-countries", "capital-world", "city-in-state", "currency", "family",
}


def plot_analogy_categories(
    results: dict[str, tuple[int, int]],
    path: str = "results/analogy_categories.png",
) -> None:
    """Horizontal bar chart of per-category analogy accuracy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Skipping analogy bar chart (matplotlib not installed)")
        return

    _setup_plot_style()

    overall_correct = sum(c for c, _ in results.values())
    overall_total = sum(t for _, t in results.values())
    overall_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0.0

    sem_cats = sorted([(k, v) for k, v in results.items() if k in _SEMANTIC_CATEGORIES])
    syn_cats = sorted([(k, v) for k, v in results.items() if k not in _SEMANTIC_CATEGORIES])

    names: list[str] = []
    accs: list[float] = []
    colors: list[str] = []
    sem_colour, syn_colour = "#2ca02c", "#1f77b4"

    for cat, (c, t) in syn_cats:
        names.append(cat)
        accs.append(c / t * 100 if t > 0 else 0.0)
        colors.append(syn_colour)

    names.append("")
    accs.append(0)
    colors.append("white")

    for cat, (c, t) in sem_cats:
        names.append(cat)
        accs.append(c / t * 100 if t > 0 else 0.0)
        colors.append(sem_colour)

    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(y_pos, accs, color=colors, edgecolor="white", height=0.7)

    ax.axvline(overall_acc, color="#d62728", linestyle="--", linewidth=1.2, zorder=3)
    ax.text(overall_acc + 0.5, len(names) - 0.5, f"overall {overall_acc:.1f}%",
            fontsize=8, color="#d62728", va="bottom")

    for i, (bar, acc) in enumerate(zip(bars, accs)):
        if names[i]:
            ax.text(acc + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.1f}%", va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"Word Analogy Accuracy by Category (overall: {overall_acc:.1f}%)")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=sem_colour, label="Semantic"),
                       Patch(facecolor=syn_colour, label="Syntactic")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_xlim(0, max(accs) + 8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved analogy bar chart to {path}")


def plot_similarity_heatmap(
    W_in: npt.NDArray[np.float64],
    vocab: Vocab,
    path: str = "results/similarity_heatmap.png",
) -> None:
    """Heatmap of pairwise cosine similarities across semantic groups."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Skipping similarity heatmap (matplotlib not installed)")
        return

    _setup_plot_style()

    groups: dict[str, list[str]] = {
        "Royalty":   ["king", "queen", "prince", "princess"],
        "Countries": ["france", "germany", "japan", "china"],
        "Tech":      ["computer", "software", "hardware"],
        "Nature":    ["river", "water", "ocean"],
        "Music":     ["music", "jazz", "song", "dance"],
    }

    ordered_words: list[str] = []
    group_boundaries: list[int] = []
    for group_words in groups.values():
        start = len(ordered_words)
        for w in group_words:
            if w in vocab.word_to_idx:
                ordered_words.append(w)
        if len(ordered_words) > start:
            group_boundaries.append(len(ordered_words))

    if len(ordered_words) < 4:
        print("  Skipping similarity heatmap (too few words in vocabulary)")
        return

    indices = [vocab.word_to_idx[w] for w in ordered_words]
    vecs = W_in[indices]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_normed = vecs / np.maximum(norms, 1e-8)
    sim_matrix = vecs_normed @ vecs_normed.T

    n = len(ordered_words)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.4, vmax=1.0, aspect="equal")

    for i in range(n):
        for j in range(n):
            val = sim_matrix[i, j]
            text_colour = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=text_colour)

    ax.set_xticks(range(n))
    ax.set_xticklabels(ordered_words, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ordered_words, fontsize=9)

    for b in group_boundaries[:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=1.0)
        ax.axvline(b - 0.5, color="black", linewidth=1.0)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
    ax.set_title("Pairwise Cosine Similarity of Word Embeddings")
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved similarity heatmap to {path}")


def plot_analogy_vectors(
    W_in: npt.NDArray[np.float64],
    vocab: Vocab,
    path: str = "results/analogy_vectors.png",
) -> None:
    """PCA projection of analogy pairs showing parallel vector structure."""
    try:
        from sklearn.decomposition import PCA  # type: ignore[import-untyped]
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Skipping analogy vector plot (missing dependency: {exc})")
        return

    _setup_plot_style()

    analogy_groups: dict[str, list[tuple[str, str]]] = {
        "Country-Capital": [("france", "paris"), ("germany", "berlin"),
                            ("italy", "rome"), ("japan", "tokyo")],
        "Gender":          [("king", "queen"), ("man", "woman"), ("prince", "princess")],
        "Comparative":     [("good", "better"), ("big", "bigger"), ("fast", "faster")],
    }

    group_colours = {
        "Country-Capital": "#1f77b4",
        "Gender":          "#d62728",
        "Comparative":     "#2ca02c",
    }

    all_words: list[str] = []
    valid_groups: dict[str, list[tuple[str, str]]] = {}
    for group_name, pairs in analogy_groups.items():
        valid_pairs = [(a, b) for a, b in pairs
                       if a in vocab.word_to_idx and b in vocab.word_to_idx]
        if valid_pairs:
            valid_groups[group_name] = valid_pairs
            for a, b in valid_pairs:
                if a not in all_words:
                    all_words.append(a)
                if b not in all_words:
                    all_words.append(b)

    if len(all_words) < 4:
        print("  Skipping analogy vector plot (too few words in vocabulary)")
        return

    indices = [vocab.word_to_idx[w] for w in all_words]
    vecs = W_in[indices]
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(vecs)
    word_to_coord = {w: coords_2d[i] for i, w in enumerate(all_words)}

    fig, ax = plt.subplots(figsize=(10, 8))

    for group_name, pairs in valid_groups.items():
        colour = group_colours.get(group_name, "black")
        for i, (a, b) in enumerate(pairs):
            ca, cb = word_to_coord[a], word_to_coord[b]
            label = group_name if i == 0 else None
            ax.annotate("", xy=cb, xytext=ca,
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.8))
            ax.scatter(*ca, c=colour, s=50, zorder=5, edgecolors="white", linewidths=0.5)
            ax.scatter(*cb, c=colour, s=50, zorder=5, edgecolors="white", linewidths=0.5,
                       label=label)

    for w in all_words:
        c = word_to_coord[w]
        ax.text(c[0], c[1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                w, fontsize=9, ha="center", va="bottom", fontweight="bold")

    ax.legend(fontsize=9, loc="best", framealpha=0.9, edgecolor="#cccccc")
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("PCA projection of analogy relationships")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved analogy vector plot to {path}")
