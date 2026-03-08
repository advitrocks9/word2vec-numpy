"""Vectorized data pipeline: subsampling, dynamic windowing, batch generation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from word2vec.vocab import Vocab


class DataLoader:
    """Produces training batches of (center, context+negatives, labels).

    All inner loops are implemented as vectorized NumPy operations — no
    Python-level iteration over individual words or pairs.

    Args:
        vocab: Built :class:`Vocab` instance (provides the negative-sampling CDF).
        corpus: Encoded corpus as a 1-D ``int32`` array of word IDs.
        window: Maximum context window half-size.
        batch_size: Number of (center, context) pairs per batch.
        n_negatives: Number of negative samples per positive pair.
        subsample_t: Threshold *t* for Mikolov's subsampling of frequent words.
    """

    def __init__(
        self,
        vocab: Vocab,
        corpus: npt.NDArray[np.int32],
        window: int = 5,
        batch_size: int = 512,
        n_negatives: int = 5,
        subsample_t: float = 1e-5,
    ) -> None:
        self.vocab = vocab
        self.corpus = corpus.copy()
        self.window = window
        self.batch_size = batch_size
        self.n_negatives = n_negatives
        self.subsample_t = subsample_t

        self._subsample()

    # ------------------------------------------------------------------
    # Subsampling
    # ------------------------------------------------------------------

    def _subsample(self) -> None:
        """Apply Mikolov's subsampling to discard frequent words.

        Each word *w* is kept with probability::

            P(keep) = min(1,  sqrt(t / f(w)) + t / f(w))

        where *f(w)* is the word's relative frequency and *t* is the
        subsampling threshold (typically 1e-5).  The entire operation is
        vectorized over the corpus array.
        """
        total = float(self.vocab.counts.sum())
        freqs = self.vocab.counts.astype(np.float64) / total  # (V,)

        # Per-vocab keep probability
        ratio = self.subsample_t / np.maximum(freqs, 1e-20)
        p_keep = np.where(freqs > 0, np.minimum(1.0, np.sqrt(ratio) + ratio), 1.0)

        # Look up keep probability for every token in the corpus
        token_keep_probs = p_keep[self.corpus]  # (corpus_len,)

        # Bernoulli mask
        mask = np.random.rand(len(self.corpus)) < token_keep_probs
        self.corpus = self.corpus[mask]

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64]]]:
        """Yield vectorized training batches.

        Each batch is a tuple of three arrays:

        * **centers** — center word IDs, shape ``(B,)``
        * **context_and_negs** — context word (column 0) concatenated with
          *K* negative samples, shape ``(B, 1+K)``
        * **labels** — ``1.0`` for the positive pair, ``0.0`` for negatives,
          shape ``(B, 1+K)``

        Yields:
            ``(centers, context_and_negs, labels)`` per batch.
        """
        corpus = self.corpus
        corpus_len = len(corpus)
        window = self.window
        B = self.batch_size
        K = self.n_negatives

        # All valid center positions, shuffled for this epoch
        positions = np.arange(window, corpus_len - window)
        np.random.shuffle(positions)

        # All possible non-zero offsets: [-window, ..., -1, 1, ..., window]
        all_offsets = np.concatenate([np.arange(-window, 0), np.arange(1, window + 1)])

        # --- Expand centers into ALL (center, context) pairs (chunked) ---
        CHUNK = 200_000
        pair_centers: list[npt.NDArray[np.int32]] = []
        pair_contexts: list[npt.NDArray[np.int32]] = []

        for chunk_start in range(0, len(positions), CHUNK):
            chunk_pos = positions[chunk_start : chunk_start + CHUNK]
            n = len(chunk_pos)

            # Dynamic window: reduction ∈ [0, window) → eff ∈ [1, window]
            reductions = np.random.randint(0, window, size=n)
            eff_windows = window - reductions  # (n,), values in [1, window]

            # Keep all offsets within [-eff, +eff] for each center
            offset_mat = np.tile(all_offsets, (n, 1))          # (n, 2*window)
            mask = np.abs(offset_mat) <= eff_windows[:, None]  # (n, 2*window)

            # Flatten valid (center, context) position pairs
            expanded_pos = np.repeat(chunk_pos, mask.sum(axis=1))
            context_pos = expanded_pos + offset_mat[mask]

            pair_centers.append(corpus[expanded_pos])
            pair_contexts.append(corpus[context_pos])

        all_centers = np.concatenate(pair_centers)
        all_contexts = np.concatenate(pair_contexts)

        # Shuffle pair ordering
        perm = np.random.permutation(len(all_centers))
        all_centers = all_centers[perm]
        all_contexts = all_contexts[perm]

        # --- Yield batches ---
        n_batches = len(all_centers) // B
        for i in range(n_batches):
            s = i * B
            e = s + B
            centers = all_centers[s:e]
            contexts = all_contexts[s:e]

            # Draw K negatives per pair from the smoothed unigram CDF
            neg_uniform = np.random.rand(B, K)
            negatives = np.searchsorted(self.vocab.neg_cdf, neg_uniform).astype(np.int32)

            # Assemble output
            context_and_negs = np.concatenate(
                [contexts[:, None], negatives], axis=1,
            )  # (B, 1+K)

            labels = np.zeros((B, 1 + K), dtype=np.float64)
            labels[:, 0] = 1.0

            yield centers, context_and_negs, labels

    def __len__(self) -> int:
        """Expected number of batches per epoch.

        Each valid center produces on average ``(window + 1)`` context pairs
        (dynamic window eff ∈ [1, window] gives E[2*eff] = window + 1).
        """
        n_valid = max(0, len(self.corpus) - 2 * self.window)
        return max(1, (n_valid * (self.window + 1)) // self.batch_size)