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
| 4q0usk1i -> wdxomlz6 -> fvg686wc | 61zybkcy (P4-E07) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 1.0, 0.05 | **0.285** | ~1.04 | **0.01392** | 9+ | Interrupted | Single continuous experiment split across VM crashes. Phase 1 (`4q0usk1i`, step `27799 -> 33999`): train loss ended ~`0.356`. Phase 2 (`wdxomlz6`, step `34049 -> 56049`): val loss improved `0.3215 -> 0.3071` (val LPIPS `0.01700 -> 0.01646`). Phase 3 (`fvg686wc`, step `56049 -> 128292`): best val loss/LPIPS reached `0.2847` / `0.01392`, train loss stayed in the low `0.31-0.38` band, and action perplexity drifted near `1.04`. Run ended via `KeyboardInterrupt` after saving `epoch=9-step=128292`. |
| cj2tuclk -> rqi23ryz -> plzyddbz | i9o9pcjj (P4-E04) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 1.0, 0.05 | 0.660 | ~1.05 | 0.01429 | 9+ | Interrupted | Single continuous experiment split across VM crashes. Phase 1 (`cj2tuclk`, step `27799 -> 34049`): train loss ended ~`0.887`. Phase 2 (`rqi23ryz`, step `34099 -> 56099`): val loss improved `0.7320 -> 0.7004` (val LPIPS `0.01798 -> 0.01639`). Phase 3 (`plzyddbz`, step `56049 -> 128493`): best val loss/LPIPS reached `0.65998` / `0.01429`, train loss mostly stayed in the `0.68-0.85` range, and action perplexity drifted down to ~`1.05`. Run ended via `KeyboardInterrupt` after saving `epoch=9-step=128493`. |
| m8wskkhz | i9o9pcjj (P4-E04) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 0.03, 0.05 | 0.5997 | 1.040 | 0.01007 | 12/12 | Finished | Fresh `i9o9pcjj` restart with lower reconstruction weight. Outperforms `plzyddbz` on val dynamics and LPIPS, but still ends in a near-single-code regime. |
| 19bvqxaq | i9o9pcjj (P4-E04) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 0.03, 0.03 | 0.5866 | 1.046 | 0.01015 | 12/12 | Finished | Same sweep family as `m8wskkhz`, but lowering `beta` from `0.05` to `0.03` improves finished validation dynamics loss. |
| aed3ufss | i9o9pcjj (P4-E04) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 0.05, 0.03 | **0.5732** | 1.067 | **0.00943** | 12/12 | Finished | Best finished Genie run in the repo so far. Raising `lambda_reconstruction` to `0.05` while keeping `beta=0.03` gives the strongest finished `i9o9pcjj` metrics, but perplexity is still only ~`1.07`. |

---

## Key Findings

1. **LAM decoder architecture bug (fixed)**: Original decoder used `[a,f,a,f,...]` interleaving with causal mask, allowing frame positions to see their own embeddings. Fixed by changing to `[f,a,f,a,...]` and predicting from action positions with `is_causal=True`.

2. **EMA + reduced beta avoids immediate collapse, but does not solve long-run code collapse by itself**: VANILLA + beta=0.25 collapsed quickly even with the architecture fix. EMA with beta in `{0.03, 0.05}` trains stably, but long runs can still drift toward perplexity ~`1.0`.

3. **LAM perplexity ~2.5-3.0 is reasonable**: With 6 codes and only 4 game actions (some more frequent than others), using ~3 codes actively makes sense. Full uniform usage (ppl=6) isn't expected.

4. **Dynamics loss differs by tokenizer**: P4-E07 gives lower dynamics loss (0.37 vs 0.84) due to different codebook distributions. Not directly comparable across tokenizers.

5. **LAM continues learning through reconstruction**: Recon loss keeps improving as training proceeds, confirming that the LAM still benefits from reconstruction signal even after the dynamics curve slows down.
6. **Dynamics can keep improving even with weak action-code usage**: In the >100k-step runs and the finished February 26 sweep, validation and train dynamics loss continue to improve while action perplexity stays low, so low perplexity alone does not imply immediate dynamics failure.
7. **Fresh `i9o9pcjj` restarts beat the older resumed chain**: All three February 26 sweeps (`m8wskkhz`, `19bvqxaq`, `aed3ufss`) outperform `plzyddbz` on validation dynamics loss and LPIPS.
8. **Within the February 26 sweep, `beta=0.03` beat `0.05`, and `lambda_reconstruction=0.05` beat `0.03`**: `aed3ufss` (`lambda_reconstruction=0.05`, `beta=0.03`) is the best finished `i9o9pcjj` run so far.

---

## Latest Finished Sweep (2026-03-04)

All three February 26 runs reached `max_epochs=12` on March 4, 2026.

- `m8wskkhz`: `lambda_reconstruction=0.03`, `beta=0.05`, `val_dynamics_loss=0.5997`, `val_lpips=0.01007`, `val_lam_codebook_perplexity=1.040`.
- `19bvqxaq`: `lambda_reconstruction=0.03`, `beta=0.03`, `val_dynamics_loss=0.5866`, `val_lpips=0.01015`, `val_lam_codebook_perplexity=1.046`.
- `aed3ufss`: `lambda_reconstruction=0.05`, `beta=0.03`, `val_dynamics_loss=0.5732`, `val_lpips=0.00943`, `val_lam_codebook_perplexity=1.067`.

Takeaway: the best finished Genie checkpoint is now `aed3ufss`, but its LAM still uses the codebook very weakly. The manual rollout evaluation below has not yet been rerun on these newer checkpoints.

---

## Manual Eval Snapshot (2026-02-22)

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
- rerun manual rollout evaluation on `aed3ufss` vs `fvg686wc`
- try separate LAM/dynamics optimizers
- mask ratio schedule vs fixed p_sample
- dynamics temperature sweep
- post-success: build latent-action → key mapping via one-step causal probe (latent index ↔ keyboard key) for control
