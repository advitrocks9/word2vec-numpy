# word2vec-numpy

Pure NumPy implementation of Skip-Gram with Negative Sampling (SGNS), trained on the text8 corpus. Implements the core training loop (forward pass, loss, hand-derived gradients, parameter updates) without PyTorch, TensorFlow, or other ML frameworks.

## Running

```bash
pip install -r requirements.txt
python train.py
```

This downloads text8 (~100MB), builds vocabulary, runs a gradient check, trains for 20 epochs with early stopping, then runs the full evaluation suite and generates all plots. Takes roughly 2.25 hours on CPU. Trained weights aren't committed (too large) but are reproducible from the fixed seed.

## How it works

Two embedding matrices, W_in (center words) and W_out (context words), both of shape (V, 200). For each (center, context) pair, K=10 negative samples are drawn from a smoothed unigram distribution (frequency^0.75). The loss is binary cross-entropy: push the positive pair's dot product up through the sigmoid, push negative pairs down.

The update step uses a Numba-compiled scatter kernel because NumPy's `np.add.at` is correct but painfully slow, and fancy indexing silently drops duplicate updates.
Other details worth noting:
- Subsampling of frequent words before training (threshold 1e-4), following Mikolov et al.
- Dynamic window sizes sampled uniformly per position, matching the original C implementation
- Linear LR decay from 0.025 to 0.0001
- The backward pass normalises gradients by B*(1+K) to make the loss a proper mean; the update rescales LR by the same factor so the effective per-pair step size equals the base LR

| Parameter | Value |
|-----------|-------|
| Embedding dim | 200 |
| Window size | 5 |
| Negative samples | 10 |
| Min count | 5 |
| Subsampling | 1e-4 |
| LR | 0.025 → 0.0001 |
| Batch size | 4096 |
| Epochs | 20 (patience 5) |

## Results

Trained on text8 (~17M tokens after subsampling). Final smoothed loss: 0.2343. Throughput around 140k tokens/sec.

### Word Analogies

Accuracy on the Google analogy test set (19,544 questions, 14 categories):

| Embedding | Overall Accuracy |
|-----------|-----------------|
| W_in | **36.2%** (6460 / 17827) |
| W_in + W_out | 35.4% (6308 / 17827) |

Semantic categories do well (capital-common-countries: 78.7%, nationality-adjective: 79.5%). Syntactic ones like opposites (4.8%) and adjective-to-adverb (12.0%) are much weaker, which makes sense given the small corpus and lack of morphological features.

![Per-category analogy accuracy](results/analogy_categories.png)

### Word Similarity

| Dataset | W_in (Spearman ρ) | W_in + W_out (Spearman ρ) | Coverage |
|---------|-------------------|--------------------------|----------|
| WordSim-353 | 0.693 | **0.712** | 351 / 353 |
| SimLex-999 | **0.298** | 0.297 | 992 / 999 |

Combining W_in + W_out helps on WordSim-353, consistent with Levy et al. (2015). SimLex is harder because it tests strict similarity rather than relatedness.

### Nearest Neighbours

| Query | Top-5 Neighbours (cosine similarity) |
|-------|--------------------------------------|
| king | kings (0.66), queen (0.61), elessar (0.59), crowned (0.59), fortinbras (0.57) |
| computer | computers (0.78), hardware (0.71), computing (0.67), software (0.65), bresenham (0.60) |
| france | spain (0.64), belgium (0.63), vexin (0.62), italy (0.62), french (0.62) |
| river | rivers (0.74), tributaries (0.71), murrumbidgee (0.70), sutlej (0.68), ziibi (0.67) |

### Plots

![Training loss](results/loss_curve.png)
*Training loss (EMA, α=0.05) with linear LR decay and epoch boundaries.*

![Cosine similarity heatmap](results/similarity_heatmap.png)
*Block-diagonal structure shows the embedding space clusters semantically related words.*

![PCA analogy vectors](results/analogy_vectors.png)
*Approximately parallel displacement vectors for country-capital, gender, and comparative relationships.*

![t-SNE](results/tsne.png)
*t-SNE projection of the 500 most frequent words, coloured by semantic category.*

For reference, production word2vec trained on billions of tokens typically hits 60-75% on the analogy task, largely due to the difference in corpus size (1B+ vs 17M tokens).

## Project structure

```
word2vec/
  vocab.py        - vocabulary, frequency counts, negative sampling table
  dataloader.py   - subsampling, dynamic windowing, batch generation
  model.py        - SGNS forward/backward, Numba SGD kernel, gradient checking
evaluate.py       - analogies, word similarity, nearest neighbours, all plots
train.py          - training loop, checkpointing, evaluation
```

## Dependencies

Python 3.10+, NumPy, Numba, matplotlib, scikit-learn, adjustText. See `requirements.txt`.

## References

- Mikolov et al. (2013). *Efficient estimation of word representations in vector space.* arXiv:1301.3781
- Mikolov et al. (2013). *Distributed representations of words and phrases and their compositionality.* NeurIPS 2013
- Levy & Goldberg (2014). *Neural word embedding as implicit matrix factorization.* NeurIPS 2014
- Levy, Goldberg & Dagan (2015). *Improving distributional similarity with lessons learned from word embeddings.* TACL 3