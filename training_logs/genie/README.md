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
| Quantizer | VANILLA (learned codebook) |
| Beta | 0.25 |

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
| 1v10p5qh | i9o9pcjj (P4-E04, ppl=421.6) | 8/8/512/6/32 | 12/8/512/25/1.0/0.2 | 1.0, 0.25 | 0.757 | **1.0** ❌ | 0.00063 | 7/10 | Failed | Decoder could see target frame; codebook collapsed @ step ~200 |
| zehhdgq5 | 61zybkcy (P4-E07, ppl=551.4) | 8/8/512/6/32 | 12/8/512/25/1.0/0.2 | 1.0, 0.25 | 0.342 | **1.0** ❌ | 0.00066 | 7/10 | Failed | Decoder could see target frame; codebook collapsed @ step ~200 |
| an9471sd | i9o9pcjj (P4-E04) | 8/8/512/6/32 | 12/8/512/25/1.0/0.2 | 1.0, 0.25 | - | - | - | 0/10 | Running | Fixed: [f,a,f,a] ordering + predict from action positions |
| TBD | 61zybkcy (P4-E07) | 8/8/512/6/32, EMA | 12/8/512/25/1.0/0.2 | 1.0, 0.25 | - | - | - | - | Planned | Same fix + EMA quantizer |

---

## Ablations / Ideas to Try

- EMA quantizer vs VANILLA for LAM codebook
- try separate LAM/dynamics optimizers
- mask ratio schedule vs fixed p_sample
- dynamics temperature sweep
- post-success: build latent-action → key mapping via one-step causal probe (latent index ↔ keyboard key) for control
