"""Vocabulary management: string-integer mapping, frequency counts, negative sampling."""

import pickle
from collections import Counter

import numpy as np
import numpy.typing as npt


class Vocab:
    """Tokenization, frequency counting, and negative sampling distribution."""

    def __init__(self) -> None:
        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}
        self.counts: npt.NDArray[np.int64] = np.array([], dtype=np.int64)
        self.vocab_size: int = 0
        self.neg_cdf: npt.NDArray[np.float64] = np.array([], dtype=np.float64)

    def build(self, tokens: list[str], min_count: int = 5) -> None:
        """Build vocabulary from a list of tokens.

        Words below ``min_count`` are collapsed into ``<UNK>``. The vocabulary
        is sorted by descending frequency so the most common words get the lowest IDs.
        """
        if not tokens:
            raise ValueError("Cannot build vocabulary from an empty corpus.")

        raw_counts = Counter(tokens)

        kept: list[tuple[str, int]] = []
        unk_count = 0
        for word, count in raw_counts.items():
            if count >= min_count:
                kept.append((word, count))
            else:
                unk_count += count

        kept.sort(key=lambda x: x[1], reverse=True)

        self.word_to_idx = {}
        self.idx_to_word = {}
        count_list: list[int] = []

        for idx, (word, count) in enumerate(kept):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            count_list.append(count)

        unk_idx = len(count_list)
        self.word_to_idx["<UNK>"] = unk_idx
        self.idx_to_word[unk_idx] = "<UNK>"
        count_list.append(max(unk_count, 1))

        self.counts = np.array(count_list, dtype=np.int64)
        self.vocab_size = len(count_list)

        self._build_neg_sampling_table()

    def _build_neg_sampling_table(self) -> None:
        """Smoothed unigram CDF for negative sampling (Mikolov et al.)."""
        powered = self.counts.astype(np.float64) ** 0.75
        cdf = np.cumsum(powered)
        cdf /= cdf[-1]
        self.neg_cdf = cdf

    def sample_negatives(self, n: int) -> npt.NDArray[np.int32]:
        """Draw *n* negative-sample word IDs from the smoothed unigram distribution."""
        uniform_samples = np.random.rand(n)
        return np.searchsorted(self.neg_cdf, uniform_samples).astype(np.int32)

    def encode(self, tokens: list[str]) -> npt.NDArray[np.int32]:
        """Map a list of tokens to integer IDs, with unknown words as ``<UNK>``."""
        unk_id = self.word_to_idx["<UNK>"]
        return np.array(
            [self.word_to_idx.get(w, unk_id) for w in tokens], dtype=np.int32
        )

    def save(self, path: str) -> None:
        """Serialize the vocabulary to disk (pickle)."""
        state = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "counts": self.counts,
            "vocab_size": self.vocab_size,
            "neg_cdf": self.neg_cdf,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        """Load a vocabulary from a pickle file."""
        with open(path, "rb") as f:
            state: dict[str, object] = pickle.load(f)  # noqa: S301

        vocab = cls()
        vocab.word_to_idx = state["word_to_idx"]  # type: ignore[assignment]
        vocab.idx_to_word = state["idx_to_word"]  # type: ignore[assignment]
        vocab.counts = state["counts"]  # type: ignore[assignment]
        vocab.vocab_size = state["vocab_size"]  # type: ignore[assignment]
        vocab.neg_cdf = state["neg_cdf"]  # type: ignore[assignment]
        return vocab
