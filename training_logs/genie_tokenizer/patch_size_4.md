# VQ-VAE Tokenizer Training Logs (Patch Size 4)

Tokenizer trained on procgen heist dataset generated using `scripts/gen_procgen_heist.py`.

**Optmizations:** 

 In order to fit all the activations at patch size 4 in memory without running out of memory, we add gradient checkpointing after each transformer layer in both the encoder and decoder. This results in a faster wall clock time than gradient accumulation based on a few quick sanity checks. 

## Fixed Hyperparameters

We select the fixed hyperparameters from the appendix of the Genie paper.

We use AdamW as our optimizer, with default betas 0.9, 0.9 (unless mentioned otherwise)

| Parameter | Value |
|-----------|-------|
| Codebook Size | 1024 |
| Codebook Dim | 32 |
| Num Heads | 8 |
| Num Layers | 4 (unless noted) |
| Frame Size | 64x64 |
| Patch Size | 4 |
| Tokens/Frame | 256 (16x16) |

---

## All Experiments

| Exp ID | Num Layers | Batch Size | Quantizer | LR | Warmup | Dead Code Thresh | Beta Schedule | Min Val LPIPS | Min Val Recon | Last Val Usage | Last Val Perplexity | Duration | Motivation | Results |
|--------|------------|------------|-----------|-----|--------|------------------|---------------|---------------|---------------|----------------|---------------------|----------|------------|---------|
| **P4-E00** | 4 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.01 (drop @10k) | 0.00460 | 0.000116 | **0.9987** | **808.3** | Crashed @ epoch 87 | Post-d_k fix baseline to revalidate low-beta drop behavior. Guided by experiments with patch size 8. | Crashed (storage) @ epoch 87. Trained well up until that point, far exceeding results from patch size 8 experiments. |
| **P4-E01** | 4 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.15->0.05 (warmup 10k, decay 10k) | 0.00516 | 0.000117 | 0.9919 | 505.2 | Crashed @ epoch 86 | Test higher beta sensitivity with decay back to 0.05. | Crashed (storage) @ epoch 86. Slightly worse than low-beta baseline so far. |
| **P4-E02** | 8 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.01 (drop @10k) | 0.00223 | 0.0000444 | 0.9972 | 218.2 | Crashed @ epoch 76 | 8-layer variant with gradient checkpointing to test low-beta transfer at higher depth. | Crashed (storage) @ epoch 76. Best LPIPS/recon so far; continued improving late. |
| **P4-E03** | 8 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.15->0.05 (warmup 10k, decay 10k) | 0.00224 | 0.0000428 | 0.9888 | 8.5 | Crashed @ epoch 76 | 8-layer variant with gradient checkpointing to test higher beta sensitivity. | Crashed (storage) @ epoch 76. Similar LPIPS to P4-E02, slightly better recon; codebook perplexity much lower (soft collapse hidden potentially due to a strong decoder). |
| **P4-E04** | 8 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.03->0.01 (drop @10k) | **0.00183** | **0.0000351** | 0.9973 | 259.6 | Completed 100 epochs | Lower beta plateau (0.03) to test lower commit weight. | Best LPIPS/recon so far; perplexity moderate. |
| **P4-E05** | 8 | 48 | EMA | 7e-5 | 10k | 1e-4 | 0.05->0.01 (drop @10k) | 0.00912 | 0.000251 | 0.9982 | 750.0 | Completed 100 epochs | LR sweep down (7e-5) while keeping beta schedule. | Significantly worse LPIPS/recon; perplexity high. |
| **P4-E06** | 8 | 48 | EMA | 3e-4 | 10k | 1e-3 | 0.05->0.01 (drop @10k) | 0.00198 | 0.0000379 | 0.9955 | 115.6 | Completed 100 epochs | Increase dead-code threshold (1e-3). | Strong metrics; lower perplexity suggests usage skew. |
| **P4-E07** | 8 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.01 (drop @10k) | 0.00409 | 0.0000982 | 0.9967 | 776.6 | Completed 100 epochs | Higher EMA decay (0.99) to test stability. | Worse LPIPS/recon; perplexity high. |

---

## Key Findings

1. **Very low beta schedule remains critical (early hypothesis)** — 0.05→0.01 at ~10k steps continues to work; lowering the plateau to 0.03 improves LPIPS/recon further.
2. **Higher capacity converges better** — 8-layer runs now complete and reach the best minima (P4-E04: 0.00183 LPIPS, 3.51e-05 recon).
3. **Dead-code threshold trades quality for usage** — bumping to 1e-3 (P4-E06) keeps usage high but drops perplexity, suggesting more skewed code usage despite strong metrics.
4. **EMA decay sensitivity** — higher decay (0.99 in P4-E07) degrades LPIPS/recon while keeping perplexity high.
5. **LR sweep down to 7e-5 underperforms** — P4-E05 is substantially worse than 3e-4, so lower LR is not automatically better at patch size 4.
6. **10M ordering matches 1M ordering** — the run with higher perplexity (P4-E07/decay 0.99) also has worse LPIPS, consistent with the 1M-scale differences vs P4-E04.

## Planned 10M Run

For the 10M-step tokenizer run, the candidate configurations are:
- **P4-E04** (best val LPIPS/recon)
- **P4-E07** (decent val metrics with higher perplexity/usage)

If either run shows instability or collapse at 10M, we can do a small sweep at that scale; the
point of earlier sweeps was to conserve compute.

---

## 10M Step Runs

| Run ID | Config | Min Val LPIPS | Min Val Recon | Last Val Usage | Last Val Perplexity | Duration | Status | Notes |
|--------|--------|---------------|---------------|----------------|---------------------|----------|--------|-------|
| i9o9pcjj | P4-E04 @ 10M (β 0.03→0.01, decay 0.985) | 0.00196 | 4.66e-05 | 0.99826 | 421.6 | 134.95h | Completed | Best of the 10M runs. |
| 61zybkcy | P4-E07 @ 10M (β 0.05→0.01, decay 0.99) | 0.00319 | 9.15e-05 | 0.99830 | 551.4 | 136.00h | Completed | Higher perplexity and worse LPIPS vs i9o9pcjj. |

---

## Ablations / Ideas to Try

- does the beta schedule matter? or is it that we just need a low beta value? 
- learning rate sweep, one higher and one lower lr 
- how do we ensure better usage / perplexity ? 

After the above experiments, we will select the best performing set of hyperparameters and run a final tokenizer training job with 10M procgen environment steps as opposed to 1M as used for the experiments so far. Then, we will move onto training the latent action model. 
