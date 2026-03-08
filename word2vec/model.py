"""SGNS model: forward, backward, SGD update, gradient checking, persistence."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class SGNSModel:
    """Skip-Gram with Negative Sampling — pure NumPy.

    Maintains two embedding matrices of shape ``(V, d)``:

    * **W_in** (center / input embeddings) — the usable embeddings after
      training.
    * **W_out** (context / output embeddings) — auxiliary; discarded
      post-training.

    The score for a (center *c*, context *o*) pair is simply
    ``W_out[o] · W_in[c]``.

    Args:
        vocab_size: Number of words in the vocabulary (*V*).
        embed_dim: Dimensionality of each embedding vector (*d*).
    """

    def __init__(self, vocab_size: int, embed_dim: int = 100) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Xavier-style init scaled to keep initial dot products small
        scale = 0.5 / embed_dim
        self.W_in: npt.NDArray[np.float64] = np.random.uniform(
            -scale, scale, (vocab_size, embed_dim)
        )
        self.W_out: npt.NDArray[np.float64] = np.random.uniform(
            -scale, scale, (vocab_size, embed_dim)
        )

        # Cached forward-pass tensors used by backward()
        self._cache: dict[str, Any] = {}
        # Gradients computed by backward()
        self._grad_v_in: npt.NDArray[np.float64] = np.empty(0)
        self._grad_v_out: npt.NDArray[np.float64] = np.empty(0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        center_idx: npt.NDArray[np.int32],
        context_and_neg_idx: npt.NDArray[np.int32],
        labels: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Compute SGNS scores and binary cross-entropy loss for a batch.

        Args:
            center_idx: Center word indices, shape ``(B,)``.
            context_and_neg_idx: Context (column 0) + negative indices,
                shape ``(B, 1+K)``.
            labels: Ground-truth labels (1 for positive, 0 for negative),
                shape ``(B, 1+K)``.

        Returns:
            A tuple ``(scores, loss)`` where *scores* has shape ``(B, 1+K)``
            (raw dot products) and *loss* is a scalar (mean BCE over the batch).
        """
        v_in = self.W_in[center_idx]              # (B, d)
        v_out = self.W_out[context_and_neg_idx]    # (B, 1+K, d)

        # Batched dot product
        scores: npt.NDArray[np.float64] = np.einsum("bd,bkd->bk", v_in, v_out)  # (B, 1+K)

        # Numerically stable BCE via logaddexp
        #   log σ(x)     = -logaddexp(0, -x)
        #   log(1-σ(x))  = -logaddexp(0,  x)
        log_sig_pos = -np.logaddexp(0.0, -scores)  # log σ(s)
        log_sig_neg = -np.logaddexp(0.0, scores)    # log(1 - σ(s))

        bce = -(labels * log_sig_pos + (1.0 - labels) * log_sig_neg)  # (B, 1+K)
        loss = float(np.mean(bce))

        # Cache for backward
        self._cache = {
            "center_idx": center_idx,
            "context_and_neg_idx": context_and_neg_idx,
            "v_in": v_in,
            "v_out": v_out,
            "scores": scores,
            "labels": labels,
        }

        return scores, loss

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Compute analytical gradients from the cached forward pass.

        After calling this method the gradients are available in
        ``self._grad_v_in`` (shape ``(B, d)``) and ``self._grad_v_out``
        (shape ``(B, 1+K, d)``).

        The key identity is that the gradient of the BCE w.r.t. scores is
        simply ``sigmoid(scores) - labels`` — identical to logistic regression.
        """
        v_in: npt.NDArray[np.float64] = self._cache["v_in"]
        v_out: npt.NDArray[np.float64] = self._cache["v_out"]
        scores: npt.NDArray[np.float64] = self._cache["scores"]
        labels: npt.NDArray[np.float64] = self._cache["labels"]

        B, K_plus_1 = labels.shape

        sigmoid = 1.0 / (1.0 + np.exp(-scores))                 # (B, 1+K)
        grad_scores = (sigmoid - labels) / (B * K_plus_1)        # (B, 1+K)

        # ∂L/∂W_out  for context+neg embeddings
        self._grad_v_out = np.einsum("bk,bd->bkd", grad_scores, v_in)   # (B, 1+K, d)

        # ∂L/∂W_in   for center embeddings
        self._grad_v_in = np.einsum("bk,bkd->bd", grad_scores, v_out)   # (B, d)

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------

    def update(self, lr: float) -> None:
        """Apply SGD with scatter-add for duplicate index accumulation.

        ``np.add.at`` is used instead of fancy-indexed subtraction because
        the latter silently overwrites (rather than accumulates) when the
        same word index appears more than once in a batch.

        Args:
            lr: Current learning rate.
        """
        center_idx: npt.NDArray[np.int32] = self._cache["center_idx"]
        context_and_neg_idx: npt.NDArray[np.int32] = self._cache["context_and_neg_idx"]
        d = self.embed_dim

        B, K_plus_1 = self._cache["labels"].shape
        effective_lr = lr * B * K_plus_1

        np.add.at(self.W_in, center_idx, -effective_lr * self._grad_v_in)
        np.add.at(
            self.W_out,
            context_and_neg_idx.ravel(),
            (-effective_lr * self._grad_v_out).reshape(-1, d),
        )

    # ------------------------------------------------------------------
    # Gradient check
    # ------------------------------------------------------------------

    def gradient_check(
        self,
        center_idx: npt.NDArray[np.int32],
        context_and_neg_idx: npt.NDArray[np.int32],
        labels: npt.NDArray[np.float64],
        epsilon: float = 1e-5,
    ) -> float:
        """Verify the analytical backward pass against finite differences.

        Computes centered finite-difference approximations for a random
        subset of parameters and returns the maximum relative error.

        Args:
            center_idx: Center word indices, shape ``(B,)``.
            context_and_neg_idx: Context + negative indices, shape ``(B, 1+K)``.
            labels: Labels, shape ``(B, 1+K)``.
            epsilon: Perturbation magnitude for finite differences.

        Returns:
            Maximum relative error across all checked parameters.
        """
        # Analytical gradients
        self.forward(center_idx, context_and_neg_idx, labels)
        self.backward()

        # Accumulate per-batch-item gradients into full weight-matrix shapes.
        # This is necessary because multiple batch items may reference the same
        # weight index; the numerical gradient perturbs the shared weight, so
        # the analytical gradient must sum all contributions.
        d = self.embed_dim
        grad_W_in = np.zeros_like(self.W_in)
        np.add.at(grad_W_in, center_idx, self._grad_v_in)

        grad_W_out = np.zeros_like(self.W_out)
        np.add.at(
            grad_W_out,
            context_and_neg_idx.ravel(),
            self._grad_v_out.reshape(-1, d),
        )

        max_rel_error = 0.0
        n_checks = 5  # dimensions to probe per index

        # --- Check W_in gradients ---
        unique_centers = list(set(int(x) for x in center_idx))
        for idx in unique_centers[:3]:
            for j in range(min(d, n_checks)):
                old = self.W_in[idx, j].copy()

                self.W_in[idx, j] = old + epsilon
                _, loss_plus = self.forward(center_idx, context_and_neg_idx, labels)

                self.W_in[idx, j] = old - epsilon
                _, loss_minus = self.forward(center_idx, context_and_neg_idx, labels)

                self.W_in[idx, j] = old  # restore

                num_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
                ana_grad = float(grad_W_in[idx, j])

                denom = max(abs(num_grad), abs(ana_grad), 1e-8)
                rel_err = abs(ana_grad - num_grad) / denom
                max_rel_error = max(max_rel_error, rel_err)

        # --- Check W_out gradients ---
        unique_out = list(set(int(x) for x in context_and_neg_idx.ravel()))
        for idx in unique_out[:3]:
            for j in range(min(d, n_checks)):
                old = self.W_out[idx, j].copy()

                self.W_out[idx, j] = old + epsilon
                _, loss_plus = self.forward(center_idx, context_and_neg_idx, labels)

                self.W_out[idx, j] = old - epsilon
                _, loss_minus = self.forward(center_idx, context_and_neg_idx, labels)

                self.W_out[idx, j] = old  # restore

                num_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
                ana_grad = float(grad_W_out[idx, j])

                denom = max(abs(num_grad), abs(ana_grad), 1e-8)
                rel_err = abs(ana_grad - num_grad) / denom
                max_rel_error = max(max_rel_error, rel_err)

        return max_rel_error

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to a ``.npz`` file.

        Args:
            path: Destination file path.
        """
        np.savez(
            path,
            W_in=self.W_in,
            W_out=self.W_out,
            vocab_size=np.array(self.vocab_size),
            embed_dim=np.array(self.embed_dim),
        )

    @classmethod
    def load(cls, path: str) -> SGNSModel:
        """Load model weights from a ``.npz`` file.

        Args:
            path: Source file path.

        Returns:
            A fully initialised :class:`SGNSModel` instance.
        """
        data = np.load(path)
        model = cls(int(data["vocab_size"]), int(data["embed_dim"]))
        model.W_in = data["W_in"]
        model.W_out = data["W_out"]
        return model
