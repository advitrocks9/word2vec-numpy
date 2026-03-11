#!/usr/bin/env python3
"""CLI entry point: download text8, run gradient check, train SGNS, evaluate."""

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
    word_similarity,
    print_similarity_results,
    plot_tsne,
    plot_loss_curve,
)

EMBED_DIM    = 200
WINDOW       = 5
N_NEGATIVES  = 10
MIN_COUNT    = 5
SUBSAMPLE_T  = 1e-4
LR_START     = 0.025
LR_END       = 0.0001
BATCH_SIZE   = 4096
EPOCHS       = 20
LOG_INTERVAL = 1000
SEED         = 42
PATIENCE     = 5
MIN_DELTA    = 0.0005


def download_text8(path: str = "text8") -> str:
    """Download and extract the text8 corpus if not already present."""
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


def _find_latest_checkpoint() -> int:
    """Scan results/ for model_epoch{N}.npz and return the highest N, or 0."""
    if not os.path.isdir("results"):
        return 0
    latest = 0
    for fname in os.listdir("results"):
        if fname.startswith("model_epoch") and fname.endswith(".npz"):
            try:
                n = int(fname[len("model_epoch"):-len(".npz")])
                latest = max(latest, n)
            except ValueError:
                pass
    return latest


def main() -> None:
    """Run the full training and evaluation pipeline."""
    np.random.seed(SEED)
    os.makedirs("results", exist_ok=True)

    text8_path = download_text8()
    print("Loading corpus ...")
    with open(text8_path) as f:
        tokens = f.read().strip().split()
    print(f"  Raw corpus: {len(tokens):,} tokens")

    print("Building vocabulary ...")
    vocab = Vocab()
    vocab.build(tokens, min_count=MIN_COUNT)
    print(f"  Vocabulary size: {vocab.vocab_size:,}")

    print("Encoding corpus ...")
    corpus = vocab.encode(tokens)
    print(f"  Encoded corpus: {corpus.shape[0]:,} int32 IDs")

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

    model = SGNSModel(vocab.vocab_size, embed_dim=EMBED_DIM, n_negatives=N_NEGATIVES)
    print(f"  Model: {vocab.vocab_size:,} x {EMBED_DIM} embeddings "
          f"({model.W_in.nbytes / 1e6:.1f} MB per matrix)")

    losses: list[float] = []
    smoothed_loss = 0.0
    global_step = 0
    best_loss = float("inf")
    stale_epochs = 0
    start_epoch = 0

    resume_epoch = _find_latest_checkpoint()
    if resume_epoch > 0:
        ckpt_path = f"results/model_epoch{resume_epoch}.npz"
        state_path = f"results/train_state_epoch{resume_epoch}.npz"
        print(f"\nResuming from epoch {resume_epoch} checkpoint ...")
        model = SGNSModel.load(ckpt_path)
        if os.path.exists(state_path):
            state = np.load(state_path)
            global_step = int(state["global_step"])
            smoothed_loss = float(state["smoothed_loss"])
            best_loss = float(state["best_loss"])
            stale_epochs = int(state["stale_epochs"])
            losses = list(state["losses"])
        else:
            global_step = resume_epoch * len(loader)
        start_epoch = resume_epoch
        print(f"  global_step={global_step:,}, smoothed_loss={smoothed_loss:.4f}, "
              f"stale_epochs={stale_epochs}")

    print("\nGradient check ...")
    grad_check_iter = iter(loader)
    gc_centers, gc_ctx_neg, gc_labels = next(grad_check_iter)
    max_err = model.gradient_check(
        gc_centers[:4], gc_ctx_neg[:4], gc_labels[:4]
    )
    status = "PASS" if max_err < 1e-5 else "FAIL"
    print(f"  Max relative error: {max_err:.2e}  [{status}]")
    assert max_err < 1e-5, f"Gradient check FAILED (max rel error = {max_err:.2e})"

    total_steps = EPOCHS * len(loader)
    remaining_epochs = EPOCHS - start_epoch
    print(f"\nTraining for {remaining_epochs} epoch(s) ({total_steps:,} total steps) ...")

    t_start = time.time()
    initial_step = global_step

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        for centers, ctx_and_negs, labels in loader:
            progress = global_step / total_steps
            lr = LR_START + (LR_END - LR_START) * progress

            _, loss = model.forward(centers, ctx_and_negs, labels)
            model.backward()
            model.update(lr)

            if global_step == 0:
                smoothed_loss = loss
            else:
                smoothed_loss = 0.95 * smoothed_loss + 0.05 * loss

            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                session_steps = global_step - initial_step
                tokens_per_sec = (session_steps + 1) * BATCH_SIZE / elapsed

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
        print(f"  -- Epoch {epoch + 1} done in {epoch_time:.1f}s  (loss {smoothed_loss:.4f})")

        if best_loss - smoothed_loss > MIN_DELTA:
            best_loss = smoothed_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            print(f"  ** No improvement for {stale_epochs}/{PATIENCE} epochs")

        model.save(f"results/model_epoch{epoch + 1}.npz")
        vocab.save(f"results/vocab_epoch{epoch + 1}.pkl")
        np.savez(
            f"results/train_state_epoch{epoch + 1}.npz",
            global_step=np.array(global_step),
            smoothed_loss=np.array(smoothed_loss),
            best_loss=np.array(best_loss),
            stale_epochs=np.array(stale_epochs),
            losses=np.array(losses),
        )

        if stale_epochs >= PATIENCE:
            print(f"  ** Early stopping -- loss plateaued at {smoothed_loss:.4f}")
            break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s")

    model.save("results/model_final.npz")
    vocab.save("results/vocab_final.pkl")

    print("\n=== Nearest Neighbours ===")
    print_nearest_neighbors(model.W_in, vocab)

    print("\n=== Word Analogies ===")
    analogy_results = word_analogy(model.W_in, vocab)
    print_analogy_results(analogy_results)

    print("\n=== Word Similarity ===")
    sim_results = []
    for dataset in ("wordsim353", "simlex999"):
        sim_results.append(word_similarity(model.W_in, vocab, dataset=dataset))
    print_similarity_results(sim_results)

    print("\n=== Visualisation ===")
    plot_loss_curve(losses, log_interval=LOG_INTERVAL)
    plot_tsne(model.W_in, vocab)

    print("\nDone.")


if __name__ == "__main__":
    main()
