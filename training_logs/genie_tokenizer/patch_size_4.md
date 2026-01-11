# VQ-VAE Tokenizer Training Logs (Patch Size 4)

Tokenizer trained on procgen heist dataset generated using `scripts/gen_procgen_heist.py`.

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

| Exp ID | Batch Size | Quantizer | LR | Warmup | Dead Code Thresh | Beta Schedule | Min Val LPIPS | Min Val Recon | Duration | Motivation | Results |
|--------|------------|-----------|-----|--------|------------------|---------------|---------------|---------------|----------|------------|---------|
| **P4-E00** | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.01 (drop @10k) | TBD | TBD | Ongoing | Post-d_k fix baseline to revalidate low-beta drop behavior. Guided by experiments with patch size 8. | TBD (ongoing). |
| **P4-E01** | 48 | EMA | 3e-4 | 10k | 1e-4 | 0.05->0.15->0.05 (warmup 10k, decay 10k) | TBD | TBD | Ongoing | Test higher beta sensitivity with decay back to 0.05. | TBD (ongoing). |

---

## Successful / Ongoing Runs

| Exp ID | Key Config | Beta Schedule | Min Val LPIPS | Min Val Recon | Duration | Status | Motivation | Results |
|--------|------------|---------------|---------------|---------------|----------|--------|------------|---------|
| **P4-E00** | LR 3e-4 | 0.05->0.01 (drop @10k) | TBD | TBD | Ongoing | **Ongoing** | Post-fix baseline to revalidate low-beta drop behavior. | TBD (ongoing). |
| **P4-E01** | LR 3e-4 | 0.05->0.15->0.05 (warmup 10k, decay 10k) | TBD | TBD | Ongoing | **Ongoing** | Test higher beta sensitivity with decay back to 0.05. | TBD (ongoing). |

---

## Key Findings

---

## Ideas to Try
