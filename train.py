#!/usr/bin/env python3
"""CLI entry point: download text8, run gradient check, train SGNS, evaluate."""

from __future__ import annotations

import os
import time
import urllib.request
import zipfile

import numpy as np

from word2vec.vocab import Vocab
from word2vec.dataloader import DataLoader
from word2vec.model import SGNSModel
from evaluate import (
    print_nearest_neighbors,
    word_analogy,
    print_analogy_results,
    plot_tsne,
    plot_loss_curve,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EMBED_DIM = 100
WINDOW = 5
N_NEGATIVES = 5
MIN_COUNT = 5
SUBSAMPLE_T = 1e-5
LR_START = 0.025
LR_END = 0.0001
BATCH_SIZE = 512
EPOCHS = 5
LOG_INTERVAL = 1000
SEED = 42


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_text8(path: str = "text8") -> str:
    """Download and extract the text8 corpus if not already present.

    Args:
        path: Destination file path for the extracted text.

    Returns:
        The file path to the extracted corpus.
    """
    if os.path.exists(path):
        return path

    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = path + ".zip"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)  # noqa: S310

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")
    os.remove(zip_path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full training and evaluation pipeline."""
    np.random.seed(SEED)
    os.makedirs("results", exist_ok=True)

    # ---- 1. Data --------------------------------------------------------
    text8_path = download_text8()
    print("Loading corpus ...")
    with open(text8_path) as f:
        tokens = f.read().strip().split()
    print(f"  Raw corpus: {len(tokens):,} tokens")

    # ---- 2. Vocabulary --------------------------------------------------
    print("Building vocabulary ...")
    vocab = Vocab()
    vocab.build(tokens, min_count=MIN_COUNT)
    print(f"  Vocabulary size: {vocab.vocab_size:,}")

    # ---- 3. Encode ------------------------------------------------------
    print("Encoding corpus ...")
    corpus = vocab.encode(tokens)
    print(f"  Encoded corpus: {corpus.shape[0]:,} int32 IDs")

    # ---- 4. DataLoader (subsampling inside) -----------------------------
    print("Creating DataLoader (subsampling) ...")
    loader = DataLoader(
        vocab, corpus,
        window=WINDOW,
        batch_size=BATCH_SIZE,
        n_negatives=N_NEGATIVES,
        subsample_t=SUBSAMPLE_T,
    )
    print(f"  Corpus after subsampling: {len(loader.corpus):,} tokens")
    print(f"  Batches per epoch: {len(loader):,}")

    # ---- 5. Model -------------------------------------------------------
    model = SGNSModel(vocab.vocab_size, embed_dim=EMBED_DIM)
    print(f"  Model: {vocab.vocab_size:,} x {EMBED_DIM} embeddings "
          f"({model.W_in.nbytes / 1e6:.1f} MB per matrix)")

    # ---- 6. Gradient check ----------------------------------------------
    print("\nGradient check ...")
    # Grab a tiny batch for the check
    grad_check_iter = iter(loader)
    gc_centers, gc_ctx_neg, gc_labels = next(grad_check_iter)
    max_err = model.gradient_check(
        gc_centers[:4], gc_ctx_neg[:4], gc_labels[:4]
    )
    status = "PASS" if max_err < 1e-5 else "FAIL"
    print(f"  Max relative error: {max_err:.2e}  [{status}]")
    assert max_err < 1e-5, f"Gradient check FAILED (max rel error = {max_err:.2e})"

    # ---- 7. Training loop -----------------------------------------------
    total_steps = EPOCHS * len(loader)
    print(f"\nTraining for {EPOCHS} epochs ({total_steps:,} steps) ...")

    losses: list[float] = []
    smoothed_loss = 0.0
    global_step = 0
    t_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        for centers, context_and_negs, labels in loader:
            # Linear LR decay
            progress = global_step / total_steps
            lr = LR_START + (LR_END - LR_START) * progress

            # Forward
            _, loss = model.forward(centers, context_and_negs, labels)

            # Backward
            model.backward()

            # Update
            model.update(lr)

            # Logging
            if global_step == 0:
                smoothed_loss = loss
            else:
                smoothed_loss = 0.95 * smoothed_loss + 0.05 * loss

            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                tokens_per_sec = (global_step + 1) * BATCH_SIZE / max(elapsed, 1e-8)
                print(
                    f"  Epoch {epoch + 1}/{EPOCHS} | "
                    f"Step {global_step:>7,}/{total_steps:,} | "
                    f"Loss {smoothed_loss:.4f} | "
                    f"LR {lr:.6f} | "
                    f"{tokens_per_sec:,.0f} tok/s"
                )
                losses.append(smoothed_loss)

            global_step += 1

        epoch_time = time.time() - epoch_start
        print(f"  -- Epoch {epoch + 1} done in {epoch_time:.1f}s")

        # Checkpoint
        model.save(f"results/model_epoch{epoch + 1}.npz")
        vocab.save(f"results/vocab_epoch{epoch + 1}.pkl")

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s")

    # Final save
    model.save("results/model_final.npz")
    vocab.save("results/vocab_final.pkl")

    # ---- 8. Evaluation --------------------------------------------------
    print("\n=== Nearest Neighbours ===")
    print_nearest_neighbors(model.W_in, vocab)

    print("\n=== Word Analogies ===")
    analogy_results = word_analogy(model.W_in, vocab)
    print_analogy_results(analogy_results)

    print("\n=== Visualisation ===")
    plot_loss_curve(losses, log_interval=LOG_INTERVAL)
    plot_tsne(model.W_in, vocab)

    print("\nDone.")


if __name__ == "__main__":
    main()
