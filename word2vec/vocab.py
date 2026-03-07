"""Vocabulary management: string-integer mapping, frequency counts."""

from __future__ import annotations

from collections import Counter

import numpy as np
import numpy.typing as npt


class Vocab:
    """Handles tokenization and frequency counting.

    Attributes:
        word_to_idx: Mapping from word strings to integer IDs.
        idx_to_word: Reverse mapping from integer IDs to word strings.
        counts: Frequency count for each word ID, shape ``(vocab_size,)``.
        vocab_size: Number of words in the vocabulary (including ``<UNK>``).
    """

    def __init__(self) -> None:
        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}
        self.counts: npt.NDArray[np.int64] = np.array([], dtype=np.int64)
        self.vocab_size: int = 0

    def build(self, tokens: list[str], min_count: int = 5) -> None:
        """Build vocabulary from a list of tokens.

        Words with count below ``min_count`` are collapsed into a single
        ``<UNK>`` token.  The vocabulary is sorted by descending frequency
        so that the most common words receive the lowest IDs.

        Args:
            tokens: Raw corpus as a list of word strings.
            min_count: Minimum frequency threshold.  Words below this count
                are mapped to ``<UNK>``.
        """
        if not tokens:
            raise ValueError("Cannot build vocabulary from an empty corpus.")

        raw_counts = Counter(tokens)

        # Separate kept words from rare words
        kept: list[tuple[str, int]] = []
        unk_count = 0
        for word, count in raw_counts.items():
            if count >= min_count:
                kept.append((word, count))
            else:
                unk_count += count

        # Sort by frequency descending (stable by insertion order for ties)
        kept.sort(key=lambda x: x[1], reverse=True)

        # Build mappings — <UNK> gets the last ID
        self.word_to_idx = {}
        self.idx_to_word = {}
        count_list: list[int] = []

        for idx, (word, count) in enumerate(kept):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            count_list.append(count)

        # Add <UNK>
        unk_idx = len(count_list)
        self.word_to_idx["<UNK>"] = unk_idx
        self.idx_to_word[unk_idx] = "<UNK>"
        count_list.append(max(unk_count, 1))  # ensure non-zero

        self.counts = np.array(count_list, dtype=np.int64)
        self.vocab_size = len(count_list)

    def encode(self, tokens: list[str]) -> npt.NDArray[np.int32]:
        """Map a list of word strings to their integer IDs.

        Unknown words are mapped to the ``<UNK>`` ID.

        Args:
            tokens: Words to encode.

        Returns:
            1-D array of integer word IDs, shape ``(len(tokens),)``.
        """
        unk_id = self.word_to_idx["<UNK>"]
        return np.array(
            [self.word_to_idx.get(w, unk_id) for w in tokens], dtype=np.int32
        )
