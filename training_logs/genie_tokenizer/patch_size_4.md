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
| **P4-E00** | 4 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.01 (drop @10k) | 0.00460 | 0.000116 | 0.9987 | 808.3 | Crashed @ epoch 87 | Post-d_k fix baseline to revalidate low-beta drop behavior. Guided by experiments with patch size 8. | Crashed (storage) @ epoch 87. Trained well up until that point, far exceeding results from patch size 8 experiments. |
| **P4-E01** | 4 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.15->0.05 (warmup 10k, decay 10k) | 0.00516 | 0.000117 | 0.9919 | 505.2 | Crashed @ epoch 86 | Test higher beta sensitivity with decay back to 0.05. | Crashed (storage) @ epoch 86. Slightly worse than low-beta baseline so far. |
| **P4-E02** | 8 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.01 (drop @10k) | **0.00223** | 0.0000444 | 0.9972 | 218.2 | Crashed @ epoch 76 | 8-layer variant with gradient checkpointing to test low-beta transfer at higher depth. | Crashed (storage) @ epoch 76. Best LPIPS/recon so far; continued improving late. |
| **P4-E03** | 8 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.15->0.05 (warmup 10k, decay 10k) | 0.00224 | **0.0000428** | 0.9888 | 8.5 | Crashed @ epoch 76 | 8-layer variant with gradient checkpointing to test higher beta sensitivity. | Crashed (storage) @ epoch 76. Similar LPIPS to P4-E02, slightly better recon; codebook perplexity much lower (soft collapse hidden potentially due to a strong decoder). |

---

## Key Findings

1. **Very low beta schedule remains critical (early hypothesis)** — 0.05→0.01 at ~10k steps appears to transfer from pre-fix patch-size-8 runs and is consistent with early 4/8-layer behavior.
2. **Higher capacity learns slower but converges better** — with the same hyperparameters, 8 layers improves more slowly early on but continues to converge later and reaches better minima (expected).
3. **8-layer runs already outperform 4-layer** — even with early crashes, 8 layers reached ~0.0022 LPIPS and ~4.3e-05 recon vs ~0.0046 LPIPS and ~1.16e-04 recon for 4 layers.
4. **Runs stopped by storage limits** — all four experiments crashed before full convergence; minima likely improve with clean reruns.
5. **Perplexity collapse can hide behind good recon** — P4-E03 matches LPIPS/recon but has very low perplexity, indicating code usage skew; for tokenizer training, encoder stability matters more than decoder strength.

---

## Ablations / Ideas to Try

- does the beta schedule matter? or is it that we just need a low beta value? 
- learning rate sweep, one higher and one lower lr 
- how do we ensure better usage / perplexity ? 

After the above experiments, we will select the best performing set of hyperparameters and run a final tokenizer training job with 10M procgen environment steps as opposed to 1M as used for the experiments so far. Then, we will move onto training the latent action model. 
