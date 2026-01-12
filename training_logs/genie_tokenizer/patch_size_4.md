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

| Exp ID | Num Layers | Batch Size | Quantizer | LR | Warmup | Dead Code Thresh | Beta Schedule | Min Val LPIPS | Min Val Recon | Duration | Motivation | Results |
|--------|------------|------------|-----------|-----|--------|------------------|---------------|---------------|---------------|----------|------------|---------|
| **P4-E00** | 4 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.01 (drop @10k) | TBD | TBD | Ongoing | Post-d_k fix baseline to revalidate low-beta drop behavior. Guided by experiments with patch size 8. | TBD (ongoing). |
| **P4-E01** | 4 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.15->0.05 (warmup 10k, decay 10k) | TBD | TBD | Ongoing | Test higher beta sensitivity with decay back to 0.05. | TBD (ongoing). |
| **P4-E02** | 8 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.01 (drop @10k) | TBD | TBD | Ongoing | 8-layer variant with gradient checkpointing to test low-beta transfer at higher depth. | TBD (ongoing). |
| **P4-E03** | 8 | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.15->0.05 (warmup 10k, decay 10k) | TBD | TBD | Ongoing | 8-layer variant with gradient checkpointing to test higher beta sensitivity. | TBD (ongoing). |

---

## Successful / Ongoing Runs

| Exp ID | Num Layers | Key Config | Beta Schedule | Min Val LPIPS | Min Val Recon | Duration | Status | Motivation | Results |
|--------|------------|------------|---------------|---------------|---------------|----------|--------|------------|---------|
| **P4-E00** | 4 | LR 3e-4 | 0.05->0.01 (drop @10k) | TBD | TBD | Ongoing | **Ongoing** | Post-fix baseline to revalidate low-beta drop behavior. | TBD (ongoing). |
| **P4-E01** | 4 | LR 3e-4 | 0.05->0.15->0.05 (warmup 10k, decay 10k) | TBD | TBD | Ongoing | **Ongoing** | Test higher beta sensitivity with decay back to 0.05. | TBD (ongoing). |
| **P4-E02** | 8 | LR 3e-4 + grad ckpt | 0.05->0.01 (drop @10k) | TBD | TBD | Ongoing | **Ongoing** | 8-layer variant with gradient checkpointing to test low-beta transfer at higher depth. | TBD (ongoing). |
| **P4-E03** | 8 | LR 3e-4 + grad ckpt | 0.05->0.15->0.05 (warmup 10k, decay 10k) | TBD | TBD | Ongoing | **Ongoing** | 8-layer variant with gradient checkpointing to test higher beta sensitivity. | TBD (ongoing). |

---

## Key Findings

1. **Very low beta schedule remains critical** — 0.05→0.01 at ~10k steps appears to transfer from pre-fix patch-size-8 runs and is consistent with early 4/8-layer behavior.

---

## Ideas to Try
