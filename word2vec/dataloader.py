"""Data pipeline: batch generation for Skip-Gram training."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from word2vec.vocab import Vocab


class DataLoader:
    """Produces training batches of (center, context+negatives, labels).

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

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64]]]:
        """Yield training batches.

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
        n_batches = corpus_len // B

        for _ in range(n_batches):
            center_pos = np.random.randint(window, corpus_len - window, size=B)

            # Sample one context offset per centre
            offsets = np.random.randint(1, window + 1, size=B)
            signs = 2 * np.random.randint(0, 2, size=B) - 1
            context_pos = center_pos + offsets * signs

            centers = corpus[center_pos]
            contexts = corpus[context_pos]

            neg_uniform = np.random.rand(B, K)
            negatives = np.searchsorted(self.vocab.neg_cdf, neg_uniform).astype(np.int32)

            context_and_negs = np.concatenate(
                [contexts[:, None], negatives], axis=1,
            )  # (B, 1+K)

            labels = np.zeros((B, 1 + K), dtype=np.float64)
            labels[:, 0] = 1.0

            yield centers, context_and_negs, labels

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return len(self.corpus) // self.batch_size
