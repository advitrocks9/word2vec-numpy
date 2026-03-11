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
        self.neg_table: npt.NDArray[np.int32] = np.array([], dtype=np.int32)

    def build(self, tokens: list[str], min_count: int = 5) -> None:
        """Build vocab from tokens; words below min_count collapse into <UNK>."""
        if not tokens:
            raise ValueError("Cannot build vocabulary from an empty corpus.")

        raw_counts = Counter(tokens)

        kept = []
        unk_count = 0
        for word, count in raw_counts.items():
            if count >= min_count:
                kept.append((word, count))
            else:
                unk_count += count

        kept.sort(key=lambda x: x[1], reverse=True)

        self.word_to_idx = {}
        self.idx_to_word = {}
        count_list = []

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
        """Pre-computed flat lookup table for O(1) negative sampling (Mikolov et al.).

        100M slots, each word occupying a number of slots proportional to count^0.75.
        Sampling reduces to a single randint index into this array.
        """
        TABLE_SIZE = 100_000_000
        powered = self.counts.astype(np.float64) ** 0.75
        probs = powered / powered.sum()
        slots = np.floor(probs * TABLE_SIZE).astype(np.int64)
        # floor truncates; give leftover slots to whichever words lost the most
        remaining = TABLE_SIZE - int(slots.sum())
        frac = probs * TABLE_SIZE - slots.astype(np.float64)
        slots[np.argsort(frac)[::-1][:remaining]] += 1
        self.neg_table = np.repeat(np.arange(self.vocab_size, dtype=np.int32), slots)

    def sample_negatives(self, n: int) -> npt.NDArray[np.int32]:
        """Draw n word IDs from the smoothed unigram distribution."""
        return self.neg_table[np.random.randint(0, len(self.neg_table), size=n)]

    def encode(self, tokens: list[str]) -> npt.NDArray[np.int32]:
        """Map tokens to integer IDs; unknown words become <UNK>."""
        unk_id = self.word_to_idx["<UNK>"]
        return np.array(
            [self.word_to_idx.get(w, unk_id) for w in tokens], dtype=np.int32
        )

    def save(self, path: str) -> None:
        """Pickle the vocabulary to disk."""
        state = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "counts": self.counts,
            "vocab_size": self.vocab_size,
            "neg_table": self.neg_table,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        """Load a pickled vocabulary from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301

        vocab = cls()
        vocab.word_to_idx = state["word_to_idx"]
        vocab.idx_to_word = state["idx_to_word"]
        vocab.counts = state["counts"]
        vocab.vocab_size = state["vocab_size"]
        vocab.neg_table = state["neg_table"]
        return vocab
