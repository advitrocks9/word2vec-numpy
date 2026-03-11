"""SGNS model: forward, backward, SGD update, gradient checking, persistence."""

from typing import Any

import numpy as np
import numpy.typing as npt


class SGNSModel:
    """Skip-Gram with Negative Sampling, pure NumPy.

    W_in (V, d): center embeddings — the output after training.
    W_out (V, d): context embeddings — auxiliary, discarded post-training.
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

        self._cache: dict[str, Any] = {}
        self._grad_v_in: npt.NDArray[np.float64] = np.empty(0)
        self._grad_v_out: npt.NDArray[np.float64] = np.empty(0)

    def forward(
        self,
        center_idx: npt.NDArray[np.int32],
        ctx_neg_idx: npt.NDArray[np.int32],
        labels: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """SGNS forward pass; returns (scores, bce_loss)."""
        v_in = self.W_in[center_idx]      # (B, d)
        v_out = self.W_out[ctx_neg_idx]   # (B, 1+K, d)

        scores = np.einsum("bd,bkd->bk", v_in, v_out)  # (B, 1+K)

        log_sig_pos = -np.logaddexp(0.0, -scores)
        log_sig_neg = -np.logaddexp(0.0, scores)

        bce = -(labels * log_sig_pos + (1.0 - labels) * log_sig_neg)
        loss = float(np.mean(bce))

        self._cache = {
            "center_idx": center_idx,
            "ctx_neg_idx": ctx_neg_idx,
            "v_in": v_in,
            "v_out": v_out,
            "scores": scores,
            "labels": labels,
        }

        return scores, loss

    def backward(self) -> None:
        """Backprop through the cached forward pass."""
        v_in = self._cache["v_in"]
        v_out = self._cache["v_out"]
        scores = self._cache["scores"]
        labels = self._cache["labels"]

        B, nk = labels.shape

        sigmoid = 1.0 / (1.0 + np.exp(-scores))
        grad_scores = (sigmoid - labels) / (B * nk)

        self._grad_v_out = np.einsum("bk,bd->bkd", grad_scores, v_in)
        self._grad_v_in = np.einsum("bk,bkd->bd", grad_scores, v_out)

    def update(self, lr: float) -> None:
        """Apply SGD with scatter-add for duplicate index accumulation.

        np.add.at is used instead of fancy-indexed assignment because the
        latter silently overwrites when the same index appears more than once.
        """
        center_idx = self._cache["center_idx"]
        ctx_neg_idx = self._cache["ctx_neg_idx"]
        d = self.embed_dim

        B, nk = self._cache["labels"].shape
        effective_lr = lr * B * nk

        np.add.at(self.W_in, center_idx, -effective_lr * self._grad_v_in)
        np.add.at(
            self.W_out,
            ctx_neg_idx.ravel(),
            (-effective_lr * self._grad_v_out).reshape(-1, d),
        )

    def _check_grad_block(
        self,
        W: npt.NDArray[np.float64],
        grad_W: npt.NDArray[np.float64],
        unique_indices: list[int],
        center_idx: npt.NDArray[np.int32],
        ctx_neg_idx: npt.NDArray[np.int32],
        labels: npt.NDArray[np.float64],
        epsilon: float,
        n_checks: int,
    ) -> float:
        max_rel_err = 0.0
        d = self.embed_dim
        for idx in unique_indices[:3]:
            for j in range(min(d, n_checks)):
                old = W[idx, j].copy()

                W[idx, j] = old + epsilon
                _, loss_plus = self.forward(center_idx, ctx_neg_idx, labels)

                W[idx, j] = old - epsilon
                _, loss_minus = self.forward(center_idx, ctx_neg_idx, labels)

                W[idx, j] = old

                num_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
                ana_grad = float(grad_W[idx, j])

                denom = max(abs(num_grad), abs(ana_grad), 1e-8)
                rel_err = abs(ana_grad - num_grad) / denom
                max_rel_err = max(max_rel_err, rel_err)
        return max_rel_err

    def gradient_check(
        self,
        center_idx: npt.NDArray[np.int32],
        ctx_neg_idx: npt.NDArray[np.int32],
        labels: npt.NDArray[np.float64],
        epsilon: float = 1e-5,
    ) -> float:
        """Finite-difference gradient check; returns max relative error."""
        self.forward(center_idx, ctx_neg_idx, labels)
        self.backward()

        d = self.embed_dim
        n_checks = 5

        grad_w_in = np.zeros_like(self.W_in)
        np.add.at(grad_w_in, center_idx, self._grad_v_in)

        grad_w_out = np.zeros_like(self.W_out)
        np.add.at(
            grad_w_out,
            ctx_neg_idx.ravel(),
            self._grad_v_out.reshape(-1, d),
        )

        unique_centers = list(set(int(x) for x in center_idx))
        err_in = self._check_grad_block(
            self.W_in, grad_w_in, unique_centers,
            center_idx, ctx_neg_idx, labels, epsilon, n_checks,
        )

        unique_out = list(set(int(x) for x in ctx_neg_idx.ravel()))
        err_out = self._check_grad_block(
            self.W_out, grad_w_out, unique_out,
            center_idx, ctx_neg_idx, labels, epsilon, n_checks,
        )

        return max(err_in, err_out)

    def save(self, path: str) -> None:
        """Save weights to a .npz file."""
        np.savez(
            path,
            W_in=self.W_in,
            W_out=self.W_out,
            vocab_size=np.array(self.vocab_size),
            embed_dim=np.array(self.embed_dim),
        )

    @classmethod
    def load(cls, path: str) -> "SGNSModel":
        """Load weights from a .npz file."""
        data = np.load(path)
        model = cls(int(data["vocab_size"]), int(data["embed_dim"]))
        model.W_in = data["W_in"]
        model.W_out = data["W_out"]
        return model
