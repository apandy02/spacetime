# VQ-VAE Tokenizer Training Logs (Patch Size 4)

Tokenizer trained on procgen heist dataset generated using `scripts/gen_procgen_heist.py`.

**Note:** Initial hyperparameters for these experiments are guided by findings from the patch size 8 experiments (see `patch_size_8.md`). Key insights carried forward:

## Fixed Hyperparameters

| Parameter | Value |
|-----------|-------|
| Codebook Size | 1024 |
| Codebook Dim | 32 |
| Num Heads | 8 |
| Num Layers | 4 |
| Frame Size | 64x64 |
| Patch Size | 4 |
| Tokens/Frame | 256 (16x16) |

---

## All Experiments

| Exp ID | Batch Size | Quantizer | LR | Warmup | Dead Code Thresh | Min Val LPIPS | Min Val Recon | Duration | Notes |
|--------|------------|-----------|-----|--------|------------------|---------------|---------------|----------|-------|
| **P4-E00** | 12 | EMA | 5e-5 | 10k | 1e-4 | 0.0223* | 0.00055* | 12h* | Baseline from patch 8 optimal config. Epoch 10. **Ongoing.** |

*Asterisk (*) indicates run is still ongoing; metrics shown are current values, not final.

---

## Successful / Ongoing Runs (so far)

| Exp ID | Key Config | Min Val LPIPS | Min Val Recon | Duration | Status | Notes |
|--------|------------|---------------|---------------|----------|--------|-------|
| **P4-E00** | Batch 12, LR 5e-5, β drop | 0.0223* | 0.00055* | 12h* | **Ongoing** | 5x better than patch 8 at same epoch (expected). Slow training with batch 12, though. |

---

## Key Findings (so far)

1. **Patch 4 converges much faster than patch 8 (expected)** — at epoch 9, already 5.5x better LPIPS than patch 8 at same epoch
3. **Memory constrained** — batch 48 OOMs on A100 80GB without optimization; using batch 12 as baseline (will experiment with gradient accumulation/checkpointing to raise batch size)

---

## Ideas to Try

- **8 layers** — more capacity and matches Genie paper's encoder/decoder depth
