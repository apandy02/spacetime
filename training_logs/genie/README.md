# Genie Training Logs (Heist)

Genie trained on procgen heist shards generated with `scripts/gen_procgen_heist.py`.
These runs train the latent action model (LAM) and MaskGIT dynamics model jointly
using a fixed, pretrained tokenizer.

## Fixed Hyperparameters

Defaults below reflect the current Genie config and are updated when the
experiment defaults change.

### LAM Defaults

| Parameter | Value |
|-----------|-------|
| Num Layers | 8 |
| Num Heads | 8 |
| d_model | 512 |
| d_linear | 2048 |
| Codebook Size | 6 |
| Codebook Dim | 32 |
| Quantizer | EMA |
| Beta | 0.05 |

### Dynamics Defaults

| Parameter | Value |
|-----------|-------|
| Num Layers | 12 |
| Num Heads | 8 |
| d_model | 512 |
| d_linear | 2048 |
| MaskGIT Steps | 25 |
| Sampling Temperature | 1.0 |
| p_sample | 0.2 |

### Optimizer Defaults

| Parameter | Value |
|-----------|-------|
| max_lr | 3e-5 |
| min_lr | 3e-6 |
| betas | (0.9, 0.9) |
| weight_decay | 1e-4 |
| warmup_steps | 5k |

---

## All Experiments

| Exp ID | Tokenizer | LAM (layers/heads/d_model/codes/dim) | Dynamics (layers/heads/d_model/steps/temp/p) | Loss Weights (lambda_recon, beta) | Min Val Dyn Loss | LAM Perplexity | Min Val LPIPS | Epochs | Status | Notes |
|--------|-----------|--------------------------------------|---------------------------------------------|-----------------------------------|------------------|----------------|---------------|--------|--------|-------|
| 1v10p5qh | i9o9pcjj (P4-E04, ppl=421.6) | 8/8/512/6/32, VANILLA | 12/8/512/25/1.0/0.2 | 1.0, 0.25 | 0.757 | **1.0** ❌ | 0.00063 | 7/10 | Failed | Decoder could see target frame; codebook collapsed @ step ~200 |
| zehhdgq5 | 61zybkcy (P4-E07, ppl=551.4) | 8/8/512/6/32, VANILLA | 12/8/512/25/1.0/0.2 | 1.0, 0.25 | 0.342 | **1.0** ❌ | 0.00066 | 7/10 | Failed | Decoder could see target frame; codebook collapsed @ step ~200 |
| an9471sd | i9o9pcjj (P4-E04) | 8/8/512/6/32, VANILLA | 12/8/512/25/1.0/0.2 | 1.0, 0.25 | - | **1.0** ❌ | - | - | Failed | Architecture fix only; beta too high, still collapsed |
| zbn6i2np | i9o9pcjj (P4-E04) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 1.0, 0.05 | 0.836 | ~3.0 ✓ | 0.0262 | 1/10 | Stopped | Architecture fix + EMA + low beta; no collapse. Used as eval baseline. |
| bcl9gigk | 61zybkcy (P4-E07) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 1.0, 0.05 | 0.373 | ~2.6 ✓ | 0.0257 | 1/10 | Stopped | Architecture fix + EMA + low beta; strongest early baseline. |
| 4q0usk1i | 61zybkcy (P4-E07) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 1.0, 0.05 | - | ~2.16 | - | 2+ | Interrupted | Phase 1 of ongoing chain; VM crash; resumed as `wdxomlz6`. |
| cj2tuclk | i9o9pcjj (P4-E04) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 1.0, 0.05 | - | ~2.81 | - | 2+ | Interrupted | Phase 1 of ongoing chain; VM crash; resumed as `rqi23ryz`. |
| wdxomlz6 | 61zybkcy (P4-E07) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 1.0, 0.05 | **0.307** | ~1.33 | **0.01646** | 4+ | Running | Phase 2 of split run (`4q0usk1i -> wdxomlz6`), then resumed again from `lightning_logs/wdxomlz6/version_0/checkpoints/last.ckpt` after VM crash. |
| rqi23ryz | i9o9pcjj (P4-E04) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 1.0, 0.05 | 0.700 | ~2.74 | **0.01639** | 4+ | Running | Phase 2 of split run (`cj2tuclk -> rqi23ryz`), then resumed again from `lightning_logs/rqi23ryz/version_0/checkpoints/last.ckpt` after VM crash. |

---

## Key Findings

1. **LAM decoder architecture bug (fixed)**: Original decoder used `[a,f,a,f,...]` interleaving with causal mask, allowing frame positions to see their own embeddings. Fixed by changing to `[f,a,f,a,...]` and predicting from action positions with `is_causal=True`.

2. **EMA + low beta prevents collapse**: Matching tokenizer settings (EMA quantizer, beta=0.05) prevents codebook collapse. VANILLA + beta=0.25 collapsed even with architecture fix.

3. **LAM perplexity ~2.5-3.0 is reasonable**: With 6 codes and only 4 game actions (some more frequent than others), using ~3 codes actively makes sense. Full uniform usage (ppl=6) isn't expected.

4. **Dynamics loss differs by tokenizer**: P4-E07 gives lower dynamics loss (0.37 vs 0.84) due to different codebook distributions. Not directly comparable across tokenizers.

5. **LAM continues learning**: Recon loss keeps improving (5.3e-4 → 4.7e-4 from step 10k-20k), confirming LAM learns through reconstruction signal even after dynamics loss plateaus.

---

## Current Eval Snapshot (2026-02-22)

Evaluated checkpoints:

- `bcl9gigk` @ `epoch=1-step=27778` with tokenizer `61zybkcy` (`epoch=22-step=239591`)
- `zbn6i2np` @ `epoch=1-step=27778` with tokenizer `i9o9pcjj` (`epoch=22-step=239591`)

Observations:

- **Single-step quality is acceptable** for both runs, with `bcl9gigk` appearing modestly stronger.
- **Multi-step autoregressive rollouts degrade quickly** for both runs, with increasing tile-local visual artifacts over horizon.
- Failure mode is most consistent with **compounding dynamics token errors under autoregressive feedback**: one-step predictions are usable, but error accumulation rapidly destabilizes longer rollouts.
- At this stage, results support a **relative** conclusion (`bcl9gigk` > `zbn6i2np`), but not a claim of robust long-horizon world-model quality yet.

---

## Ablations / Ideas to Try

- ~~EMA quantizer vs VANILLA for LAM codebook~~ (EMA works, VANILLA collapses)
- ~~LAM decoder architecture fix~~ (implemented)
- try separate LAM/dynamics optimizers
- mask ratio schedule vs fixed p_sample
- dynamics temperature sweep
- post-success: build latent-action → key mapping via one-step causal probe (latent index ↔ keyboard key) for control
