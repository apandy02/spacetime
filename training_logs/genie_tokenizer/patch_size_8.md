# VQ-VAE Tokenizer Training Logs (Patch Size 8)

## Fixed Hyperparameters

| Parameter | Value |
|-----------|-------|
| Codebook Size | 1024 |
| Codebook Dim | 32 |
| Num Heads | 8 |
| Num Layers | 4 (unless noted) |
| Frame Size | 64x64 |
| Patch Size | 8 |

---

## All Experiments

| Exp ID | Quantizer | β (commit) | EMA Decay | LR enc/dec | Warmup | Dead Code Thresh | Entropy Wt | Min Val LPIPS | Min Val Recon | Notes |
|--------|-----------|------------|-----------|------------|--------|------------------|------------|---------------|---------------|-------|
| E00 | Vanilla | 0.25 const | 0.99 | 3e-4 | 1k | N/A | N/A | | | Learns well early, then collapses. **Terminated.** |
| E01 | EMA | 0.30 const | 0.95 | 5e-5 | 500 | N/A | N/A | | | Perplexity ~7.5 @ 10k. **Terminated (storage).** |
| E02 | EMA | 0.05→0.35 (10k) | 0.985 | 3e-4 | 1k | N/A | N/A | 0.607 | 0.0447 | Collapsed early. **Terminated.** |
| E02.5 | EMA | 0.05→0.35 (10k) | 0.95 | 3e-4 | 1k | N/A | N/A | | | Val perplexity <2 over 5k steps. |
| E02.7 | EMA | 0.05→0.35 (10k) | 0.95 | 3e-4 | 10k | N/A | N/A | 0.128 | 0.00229 | Collapsed after 11k. |
| E03 | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | 10k | N/A | N/A | 0.364 | 0.0138 | **Collapsed 18–19k.** |
| E03.5lin | EMA | 0.02→0.20 (10k) | 0.985 | 3e-4 | 10k | N/A | N/A | 0.114 | 0.00137 | Slightly better FID, collapsed. |
| E04 | EMA | 0.05→0.25 (10k) | 0.995 | 3e-4 | 10k | N/A | N/A | 0.103 | 0.00161 | Earlier collapse (12k). |
| **E05** | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | 10k | 0.01 | N/A | 0.174 | 0.00195 | Perplexity 500, usage 75%. **Collapsed ~6k.** |
| E06 | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | 10k | 0.001 | N/A | 0.162 | 0.00369 | Same as E05 — collapsed early. |
| E07 | EMA | 0.05→0.25 (10k) | 0.985 | 1e-4 | 20k | 1e-4 | N/A | 0.158 | 0.00341 | Temp collapse ~20k, crashed ~21k. |
| E08 | EMA | 0.05→0.25 (10k) | 0.985 | 1e-4 | 20k | 1e-4 | N/A | 0.145 | 0.00273 | FP32 EMA + grad clip. Flip-flopping ~18–19k. |
| E09 | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | N/A | 0.147 | 0.00277 | **L2 norm.** Angular collapse @ 10.9k. |
| E10 | EMA | 0.05→0.15 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | 0.1 | 0.150 | 0.00286 | **L2 norm + entropy.** Collapsed @ 11k. |
| E11 (v18) | EMA | 0.05→0.10 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | 0.5 | 0.076 | 0.00145 | **Collapsed ~10–11k.** |
| E12 (v19) | EMA | 0.05→0.10 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | 1.0 | 0.070 | 0.00132 | **Collapsed ~10.4k.** |
| E13 (v20) | EMA | 0.05 const | 0.985 | 3e-4 | 10k | 1e-4 | 1.0 | 0.077 | 0.00154 | **Collapsed ~11.5k.** |
| E14 (v21) | EMA | 0.05→0.10 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | 0.5 | 0.098 | 0.00501 | LPIPS train. **Collapsed ~4.7k.** |
| E15 (v23) | EMA | 0.05 const | 0.985 | 3e-4 | 10k | 1e-4 | 0.0 | 0.073 | 0.00137 | **Collapsed ~10k.** |
| E16 (v27) | EMA | 0.05 const | 0.985 | 3e-4 | 10k | 1e-4 | 0.0 | 0.095 | 0.00202 | **8 layers.** Collapsed ~18k. |
| E17 (v28) | EMA | 0.05 const | 0.9 | 3e-4 | 10k | N/A | 0.0 | 0.097 | 0.00210 | **8 layers.** **Collapsed ~22k.** |
| E18 (v29) | EMA | 0.25→0.01 (10k) | 0.985 | 3e-4 | 10k | N/A | N/A | 0.087 | 0.00150 | 8 layers. **Collapsed ~10–12k.** |
| **E19 (v30)** | EMA | 0.05→0.01 drop | 0.985 | 1e-4 | 10k | 1e-4 | 0.0 | **0.023** | **0.00043** | **BEST RUN.** Temp collapse ~70k, recovered. |
| E20 | EMA | 0.05→0.01 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | 0.0 | | | **Collapsed; stopped.** |
| E21 | EMA | 0.05→0.01 (10k) | 0.985 | 3e-4 | 15k | 1e-4 | 0.0 | | | **Collapsed; stopped.** |
| **E22 (v35)** | EMA | 0.05 flat | 0.985 | 1e-4 | 10k | 1e-4 | 0.0 | 0.033 | 0.00059 | **Ongoing.** Temp collapse ~55k. |

---

## Successful / Ongoing Runs

| Exp ID | Key Config | Min Val LPIPS | Min Val Recon | Status | Notes |
|--------|------------|---------------|---------------|--------|-------|
| **E19 (v30)** | β: 0.05→0.01 drop, LR: 1e-4 | **0.023** | **0.00043** | **Best run** | Temp collapse ~70k recovered. |
| **E22 (v35)** | β: 0.05 flat, LR: 1e-4 | 0.033 | 0.00059 | **Ongoing** | Temp collapse ~55k, recovered but didn't match pre-collapse. |

### E19 vs E22 Comparison
- Both use LR 1e-4 with cosine annealing, same warmup (10k), same dead code threshold (1e-4)
- **E19**: β drops to 0.01 post-warmup → temp collapse at ~70k, **LPIPS 0.023**
- **E22**: β stays flat at 0.05 → temp collapse at ~55k (earlier), **LPIPS 0.033**
- **Hypothesis**: The β drop in E19 may provide additional stability, delaying temp collapse by ~15k steps

---

## Proposed Experiments (3 GPUs available)

| Exp ID | β Schedule | LR | Warmup | Hypothesis |
|--------|------------|-----|--------|------------|
| **E23** | 0.05 flat (40k) → 0.01 drop | 1e-4 (cosine) | 10k | Delay β drop to give encoder more time to stabilize |
| **E24** | 0.05 → 0.01 cosine (20k) | 1e-4 (cosine) | 10k | Gradual β decay instead of sudden drop |
| **E25** | 0.05 flat → 0.01 drop | 5e-5 (cosine) | 10k | Test LR-β interaction with lower LR |

### Rationale
- **E23 vs E19**: E19 drops β immediately post-warmup (10k). E23 delays to 40k to test if longer flat phase helps.
- **E24**: Tests whether gradual decay is better than sudden drop
- **E25**: Tests LR-β interaction — if lower LR compensates for higher β, maybe even lower LR + drop = more stable

---

## Key Findings (So far subject to results from further ablations)

1. **Codebook collapse** is the primary failure mode across almost all experiments
2. **Lower learning rate (1e-4 vs 3e-4)** appears critical for stability
3. **Beta schedule:** Flat low β (0.05) with post-warmup drop to 0.01 works best (E19)
4. **EMA decay:** Standard 0.985 works; very low decay (0.9) delayed collapse to ~22k but didn't prevent it
5. **Dead code refresh:** Improved metrics dramatically (E05: perplexity 500, usage 75%) but didn't prevent eventual collapse
6. **L2 normalization:** Caused angular collapse on unit sphere (commit loss ~5e-8)
7. **Entropy regularization:** Delayed but didn't prevent collapse
8. **LPIPS loss:** Accelerated collapse significantly (E14 collapsed @ 4.7k)
9. **Increased capacity (8 layers):** Delayed collapse but didn't prevent it alone
