"""SGNS model with Numba-accelerated SGD"""

from typing import Any

import numba
import numpy as np
import numpy.typing as npt


@numba.njit(fastmath=True)
def _scatter_update(W_in, W_out, center_idx, ctx_neg_idx, grad_v_in, grad_v_out, effective_lr):
    """Accumulate gradient updates for W_in and W_out (handles duplicate indices)."""
    B = center_idx.shape[0]
    nk = ctx_neg_idx.shape[1]
    d = W_in.shape[1]
    for i in range(B):
        ci = center_idx[i]
        for j in range(d):
            W_in[ci, j] -= effective_lr * grad_v_in[i, j]
    for i in range(B):
        for k in range(nk):
            oi = ctx_neg_idx[i, k]
            for j in range(d):
                W_out[oi, j] -= effective_lr * grad_v_out[i, k, j]


class SGNSModel:
    """Skip-Gram with Negative Sampling."""

    def __init__(self, vocab_size: int, embed_dim: int = 100, n_negatives: int = 5,
                 batch_size: int = 4096) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_negatives = n_negatives
        self.batch_size = batch_size
        self.hparams: dict[str, int | float] = {}

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
        nk = 1 + n_negatives
        self._scores: npt.NDArray[np.float64] = np.empty((batch_size, nk), dtype=np.float64)
        self._sigmoid: npt.NDArray[np.float64] = np.empty((batch_size, nk), dtype=np.float64)

    def forward(
        self,
        center_idx: npt.NDArray[np.int32],
        ctx_neg_idx: npt.NDArray[np.int32],
        labels: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Forward pass, returns (scores, mean BCE loss)."""
        v_in = self.W_in[center_idx]      # (B, d)
        v_out = self.W_out[ctx_neg_idx]   # (B, 1+K, d)

        B = len(center_idx)
        nk = ctx_neg_idx.shape[1]
        scores = self._scores[:B, :nk]
        np.einsum("bd,bkd->bk", v_in, v_out, out=scores)  # (B, 1+K)

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
        """Compute gradients from cached forward pass."""
        v_in = self._cache["v_in"]
        v_out = self._cache["v_out"]
        scores = self._cache["scores"]
        labels = self._cache["labels"]

        B, nk = labels.shape

        sig = self._sigmoid[:B, :nk]
        np.exp(-scores, out=sig)
        np.add(1.0, sig, out=sig)
        np.divide(1.0, sig, out=sig)
        grad_scores = (sig - labels) / (B * nk)

        self._grad_v_out = np.einsum("bk,bd->bkd", grad_scores, v_in)
        self._grad_v_in = np.einsum("bk,bkd->bd", grad_scores, v_out)

    def update(self, lr: float) -> None:
        """SGD step via Numba scatter kernel (np.add.at is too slow)."""
        center_idx = self._cache["center_idx"]
        ctx_neg_idx = self._cache["ctx_neg_idx"]

        B, nk = self._cache["labels"].shape
        effective_lr = lr * B * nk

        _scatter_update(
            self.W_in, self.W_out,
            center_idx, ctx_neg_idx,
            self._grad_v_in, self._grad_v_out,
            effective_lr,
        )

    def _check_grad_block(self, W, grad_W, unique_indices, center_idx, ctx_neg_idx,
                          labels, epsilon, n_checks):
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
        """Finite-difference gradient check. Returns max relative error."""
        # TODO: this is slow for large batches, only use small B for checking
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
        arrays: dict[str, npt.NDArray[np.float64]] = {
            "W_in": self.W_in,
            "W_out": self.W_out,
            "vocab_size": np.array(self.vocab_size),
            "embed_dim": np.array(self.embed_dim),
            "n_negatives": np.array(self.n_negatives),
            "batch_size": np.array(self.batch_size),
        }
        for k, v in self.hparams.items():
            arrays[f"hp_{k}"] = np.array(v)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "SGNSModel":
        data = np.load(path)
        model = cls(
            int(data["vocab_size"]),
            int(data["embed_dim"]),
            n_negatives=int(data["n_negatives"]),
            batch_size=int(data["batch_size"]),
        )
        model.W_in = data["W_in"]
        model.W_out = data["W_out"]
        model.hparams = {
            k[3:]: float(data[k]) for k in data.files if k.startswith("hp_")
        }
        return model
