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

| Exp ID | Quantizer | β (commit) | EMA Decay | LR enc/dec | LR codebook | Warmup | Dead Code Thresh | Entropy Weight | Notes |
|--------|-----------|------------|-----------|------------|-------------|--------|------------------|----------------|-------|
| E00 | Vanilla | 0.25 const | 0.99 | 3e-4 | 3e-4 | 1k | N/A | N/A | Learns well early (500–1500 steps), then codebook collapses. Static tokens learned well, moving character not learned. Loss spikes then plateaus. **Terminated.** |
| E01 | EMA | 0.30 const | 0.95 | 5e-5 | N/A | 500 | N/A | N/A | Starts slowly, codebook looks collapsed but perplexity increases logarithmically. After 10k steps perplexity ~7.5. Loss still decreasing. **Terminated (storage issues).** |
| E02 | EMA | 0.05→0.35 (10k) | 0.985 | 3e-4 | N/A | 1k | N/A | N/A | Codebook collapsed early (perplexity 1), didn't recover for 1k steps. **Terminated.** |
| E02.5 | EMA | 0.05→0.35 (10k) | 0.95 | 3e-4 | N/A | 1k | N/A | N/A | Codebook improved vs prior collapses, but val perplexity <2 over 5k steps. Reconstruction loss noisy. Adam betas (0.9, 0.9). |
| E02.7 | EMA | 0.05→0.35 (10k) | 0.95 | 3e-4 | N/A | 10k | N/A | N/A | Trained well till ~13k steps (perplexity up to 20). Static elements learned well, moving character not learned. Collapsed after 11k. Hypothesis: final β too high. |
| E03 | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | N/A | 10k | N/A | N/A | Trained well till ~18k steps (best val perceptual loss so far). Static elements good, temporal dynamics poor. **Collapsed 18–19k.** |
| E03.5lin | EMA | 0.02→0.20 (10k) | 0.985 | 3e-4 | N/A | 10k | N/A | N/A | Similar to above, slightly better FID, slightly longer till collapse. β schedule improvements hitting limit. |
| E04 | EMA | 0.05→0.25 (10k) | 0.995 | 3e-4 | N/A | 10k | N/A | N/A | Earlier collapse (12k). Hypothesis rejected: higher decay doesn't improve stability. |
| **E05** | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | N/A | 10k | 0.01 | N/A | **Best metrics by far:** perplexity up to 500, code usage >75%. **Sudden collapse ~6k.** |
| E06 | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | N/A | 10k | 0.001 | N/A | Same as E05 — collapsed early. |
| E07 | EMA | 0.05→0.25 (10k) | 0.985 | 1e-4 | N/A | 20k | 1e-4 | N/A | Cosine anneal after warmup. Temp collapse ~20k, recovered, crashed ~21k. |
| E08 | EMA | 0.05→0.25 (10k) | 0.985 | 1e-4 | N/A | 20k | 1e-4 | N/A | Cosine anneal + FP32 EMA + grad clip 1.0. Perplexity/usage/loss flip-flopping ~18–19k. |
| E09 | EMA | 0.05→0.25 (10k) | 0.985 | 3e-4 | N/A | 10k | 1e-4 | N/A | **L2 normalized encoder.** Peak perplexity 439 @ 6.9k. Collapsed @ 10.9k (β max). Commit loss ~5e-8: angular collapse on unit sphere. |
| E10 | EMA | 0.05→0.15 (10k) | 0.985 | 3e-4 | N/A | 10k | 1e-4 | 0.1 | **L2 norm + entropy reg.** Collapsed @ 11k. Entropy loss rose 0.21→1.0 but couldn't prevent angular collapse. |
| E11 (v18) | EMA | 0.05→0.10 (10k) | 0.985 | 3e-4 | N/A | 10k | 1e-4 | 0.5 | 4 layers. **Collapsed ~10–11k.** Entropy reg delayed but didn't prevent collapse. |
| E12 (v19) | EMA | 0.05→0.10 (10k) | 0.985 | 3e-4 | N/A | 10k | 1e-4 | 1.0 | 4 layers. **Collapsed ~10.4k.** Higher entropy weight still insufficient. |
| E13 (v20) | EMA | 0.05 const | 0.985 | 3e-4 | N/A | 10k | 1e-4 | 1.0 | 4 layers. **Collapsed ~11.5k.** Lower β didn't help. |
| E14 (v21) | EMA | 0.05→0.10 (10k) | 0.985 | 3e-4 | N/A | 10k | 1e-4 | 0.5 | 4 layers, LPIPS=0.1. **Collapsed ~4.7k.** LPIPS accelerated collapse significantly. |
| E15 (v22/23) | EMA | 0.05 const | 0.985 | 3e-4 | N/A | 10k | 1e-4 | 0.0 | 4 layers. **Collapsed ~10k.** Isolation test: low β alone doesn't prevent collapse. |
| E16 (v27) | EMA | 0.05 const | 0.985 | 3e-4 | N/A | 10k | 1e-4 | 0.0 | **8 layers.** Collapsed ~18k. More capacity delayed collapse. |
| E17 (v28) | EMA | 0.05 const | 0.9 | 3e-4 | 1e-4 | 10k | N/A | 0.0 | **8 layers.** Perplexity 10–24, usage 6–13%. Lower EMA decay delayed collapse. Loss stagnant, reconstructions not great. **Collapsed ~22k.** |
| E18 (v29) | EMA | 0.25→0.01 (10k) | 0.985 | 3e-4 | N/A | 10k | N/A | N/A | 8 layers. **Collapsed ~10–12k.** β decay didn't help. |
| **E19** | EMA | 0.05 flat → 0.01 drop | 0.985 | 1e-4 | N/A | 10k | 1e-4 | 0.0 | **BEST RUN — only one that doesn't collapse.** Temp collapse ~70k, recovered. Learned some temporal reasoning (character + movement). |
| E20 | EMA | 0.05 flat → 0.01 (10k) | 0.985 | 3e-4 | N/A | 10k | 1e-4 | 0.0 | **Collapsed; stopped.** |
| E21 | EMA | 0.05 flat → 0.01 (10k) | 0.985 | 3e-4 | N/A | 15k | 1e-4 | 0.0 | **Collapsed; stopped.** |
| **E22** | EMA | 0.05 flat | 0.985 | 1e-4 | N/A | 10k | 1e-4 | 0.0 | **Ongoing; training well.** |

---

## Successful / Ongoing Runs

| Exp ID | Key Config | Status | Notes |
|--------|------------|--------|-------|
| **E19** | β: 0.05→0.01 drop, LR: 1e-4 | **Best run** | Only run that didn't collapse. Temp collapse ~70k recovered. Learned temporal reasoning. |
| **E22** | β: 0.05 flat, LR: 1e-4 | **Ongoing** | Training well. |

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
