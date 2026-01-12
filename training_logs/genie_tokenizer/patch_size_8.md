# VQ-VAE Tokenizer Training Logs (Patch Size 8)

Tokenizer trained on procgen heist dataset generated using `scripts/gen_procgen_heist.py`.

**Important (pre-fix note):** All experiments in this log were run before fixing a bug where `d_k = d_model` (instead of `d_model // num_heads`). These results reflect a much larger attention head dimension and are not directly comparable to post-fix runs, though the qualitative findings (e.g., codebook hyperparameters, collapse mitigation) still transfer. After fixing the bug (model size 147M -> 29M), new sweeps moved to patch size 4 for compute efficiency since at the lower param count the increased attention activations were easier to deal with in memory.

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
| Patch Size | 8 |

---

## All Experiments

| Exp ID | Quantizer | β (commit) | EMA Decay | LR enc/dec | Warmup | Dead Code Thresh | Entropy Wt | Min Val LPIPS | Min Val Recon | Duration | Notes |
|--------|-----------|------------|-----------|------------|--------|------------------|------------|---------------|---------------|----------|-------|
| E00 | Vanilla | 0.25 const | 0.99 | 3e-4 | 1k | N/A | N/A | | | | Learns well early, then collapses. **Terminated.** |
| E01 | EMA | 0.30 const | 0.95 | 5e-5 | 500 | N/A | N/A | | | | Perplexity ~7.5 @ 10k. **Terminated (storage).** |
| E02 | EMA | 0.05→0.35 (10k) | 0.985 | 3e-4 | 1k | N/A | N/A | 0.607 | 0.0447 | | Collapsed early. **Terminated.** |
| E02.5 | EMA | 0.05→0.35 (10k) | 0.95 | 3e-4 | 1k | N/A | N/A | | | | Val perplexity <2 over 5k steps. |
| E02.7 | EMA | 0.05→0.35 (10k) | 0.95 | 3e-4 | 10k | N/A | N/A | 0.128 | 0.00229 | | Collapsed after 11k. |
| E03 | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | 10k | N/A | N/A | 0.364 | 0.0138 | | **Collapsed 18–19k.** |
| E03.5lin | EMA | 0.02→0.20 (10k) | 0.985 | 3e-4 | 10k | N/A | N/A | 0.114 | 0.00137 | | Slightly better FID, collapsed. |
| E04 | EMA | 0.05→0.25 (10k) | 0.995 | 3e-4 | 10k | N/A | N/A | 0.103 | 0.00161 | | Earlier collapse (12k). |
| **E05** | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | 10k | 0.01 | N/A | 0.174 | 0.00195 | | Perplexity 500, usage 75%. **Collapsed ~6k.** |
| E06 | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | 10k | 0.001 | N/A | 0.162 | 0.00369 | | Same as E05 — collapsed early. |
| E07 | EMA | 0.05→0.25 (10k) | 0.985 | 1e-4 | 20k | 1e-4 | N/A | 0.158 | 0.00341 | | Temp collapse ~20k, crashed ~21k. |
| E08 | EMA | 0.05→0.25 (10k) | 0.985 | 1e-4 | 20k | 1e-4 | N/A | 0.145 | 0.00273 | | FP32 EMA + grad clip. Flip-flopping ~18–19k. |
| E09 | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | N/A | 0.147 | 0.00277 | | **L2 norm.** Angular collapse @ 10.9k. |
| E10 | EMA | 0.05→0.15 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | 0.1 | 0.150 | 0.00286 | | **L2 norm + entropy.** Collapsed @ 11k. |
| E11 (v18) | EMA | 0.05→0.10 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | 0.5 | 0.076 | 0.00145 | 6h 15m | **Collapsed ~10–11k.** |
| E12 (v19) | EMA | 0.05→0.10 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | 1.0 | 0.070 | 0.00132 | 13h 50m | **Collapsed ~10.4k.** |
| E13 (v20) | EMA | 0.05 const | 0.985 | 3e-4 | 10k | 1e-4 | 1.0 | 0.077 | 0.00154 | 13h 14m | **Collapsed ~11.5k.** |
| E14 (v21) | EMA | 0.05→0.10 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | 0.5 | 0.098 | 0.00501 | 9h 22m | LPIPS train. **Collapsed ~4.7k.** |
| E15 (v23) | EMA | 0.05 const | 0.985 | 3e-4 | 10k | 1e-4 | 0.0 | 0.073 | 0.00137 | 9h 9m | **Collapsed ~10k.** |
| E16 (v27) | EMA | 0.05 const | 0.985 | 3e-4 | 10k | 1e-4 | 0.0 | 0.095 | 0.00202 | 7h 15m | **8 layers.** Collapsed ~18k. |
| E17 (v28) | EMA | 0.05 const | 0.9 | 3e-4 | 10k | N/A | 0.0 | 0.097 | 0.00210 | 7h 14m | **8 layers.** **Collapsed ~22k.** |
| E18 (v29) | EMA | 0.25→0.01 (10k) | 0.985 | 3e-4 | 10k | N/A | N/A | 0.087 | 0.00150 | 4h 38m | 8 layers. **Collapsed ~10–12k.** |
| **E19 (v30)** | EMA | 0.05→0.01 drop | 0.985 | 1e-4 | 10k | 1e-4 | 0.0 | 0.023 | 0.00043 | 21h 8m | **Previous best.** Temp collapse ~70k, recovered. |
| E20 | EMA | 0.05→0.01 (10k) | 0.985 | 3e-4 | 10k | 1e-4 | 0.0 | | | | **Collapsed; stopped.** |
| E21 | EMA | 0.05→0.01 (10k) | 0.985 | 3e-4 | 15k | 1e-4 | 0.0 | | | | **Collapsed; stopped.** |
| E22 (v35) | EMA | 0.05 flat | 0.985 | 1e-4 | 10k | 1e-4 | 0.0 | 0.033 | 0.00059 | 26h 39m | Temp collapse ~55k. **Stopped.** |
| E23 | EMA | 0.05 flat (40k) → 0.01 | 0.985 | 1e-4 | 10k | 1e-4 | 0.0 | 0.025 | 0.00046 | | Delayed β drop didn't help. **Stopped @ 67k.** |
| E24 | EMA | 0.05→0.01 (20k decay) | 0.985 | 1e-4 | 10k | 1e-4 | 0.0 | 0.021 | 0.00039 | 54h | Gradual decay works. Completed 100 epochs. |
| **E25** | EMA | 0.05→0.01 drop | 0.985 | **5e-5** | 10k | 1e-4 | 0.0 | **0.014** | **0.00030** | 53h | **NEW BEST.** Completed 100 epochs. |
| **E26** | EMA | 0.05→0.01 drop | 0.985 | **2.5e-5** | 10k | 1e-4 | 0.0 | 0.038* | 0.00073* | 36h* | Even lower LR. Epoch 67. **Ongoing.** |
| **E27** | EMA | 0.05→0.01 (20k decay) | 0.985 | **5e-5** | 10k | 1e-4 | 0.0 | 0.018* | 0.00036* | 36h* | Best LR + gradual decay. Epoch 67. **Ongoing.** |

*Asterisk (*) indicates run is still ongoing; metrics shown are current values, not final.

---

## Successful / Ongoing Runs

| Exp ID | Key Config | Min Val LPIPS | Min Val Recon | Duration | Status | Notes |
|--------|------------|---------------|---------------|----------|--------|-------|
| **E25** | β: 0.05→0.01 drop, **LR: 5e-5** | **0.014** | **0.00030** | 53h | **Completed** | **NEW BEST!** ~40% better LPIPS than E19. |
| **E27** | β: 0.05→0.01 (20k decay), **LR: 5e-5** | 0.018* | 0.00036* | 36h* | **Ongoing** | On track to match E25. Epoch 67. |
| **E26** | β: 0.05→0.01 drop, **LR: 2.5e-5** | 0.038* | 0.00073* | 36h* | **Ongoing** | Slower convergence than 5e-5. Epoch 67. |
| E24 | β: 0.05→0.01 (20k decay), LR: 1e-4 | 0.021 | 0.00039 | 54h | Completed | Gradual decay works, but higher LR limits quality. |
| E19 (v30) | β: 0.05→0.01 drop, LR: 1e-4 | 0.023 | 0.00043 | 21h 8m | Completed | Previous best before LR discoveries. |

---

## Key Findings (Updated Jan 10)

1. **Codebook collapse** is the primary failure mode across almost all experiments
2. **LR 5e-5 is optimal** — E25 achieved best results (LPIPS 0.014). LR 2.5e-5 converges slower. LR 1e-4+ prone to instability.
3. **Beta schedule:** Flat low β (0.05) with drop to 0.01 works; immediate drop slightly better than gradual
4. **EMA decay:** Standard 0.985 works best
5. **Dead code refresh:** Improved metrics dramatically but didn't prevent collapse alone
6. **L2 normalization:** Caused angular collapse on unit sphere
7. **Entropy regularization:** Delayed but didn't prevent collapse
8. **LPIPS loss:** Accelerated collapse significantly
9. **Increased capacity (8 layers):** Delayed collapse but didn't prevent it alone
10. **LR-β interaction:** Lower LR allows more aggressive β schedules to work
11. **Best config (E25):** LR 5e-5, β 0.05→0.01 immediate drop, 100 epochs → LPIPS 0.014, Recon 0.00030
