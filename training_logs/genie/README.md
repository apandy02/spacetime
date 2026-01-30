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

| Exp ID | Tokenizer | LAM (layers/heads/d_model/codes/dim) | Dynamics (layers/heads/d_model/steps/temp/p) | Loss Weights (lambda_recon, beta) | Min Val Dyn Loss | Min Val Recon | Min Val Commit | Min Val Total | Min Val LPIPS | Duration | Notes |
|--------|-----------|--------------------------------------|---------------------------------------------|-----------------------------------|------------------|---------------|----------------|---------------|---------------|----------|-------|
| 2sqwg4wo | i9o9pcjj (P4-E04) | 8/8/512/6/32 | 12/8/512/25/1.0/0.2 | 0.01, 0.25 | 0.1356 | 7.98e-05 | 2.81e-07 | 0.1356 | 0.00446 | Ongoing | bs=36, compile, grad_ckpt; val @ step 13888 |
| kovejwoy | 61zybkcy (P4-E07) | 8/8/512/6/32 | 12/8/512/25/1.0/0.2 | 0.01, 0.25 | 0.1140 | 3.34e-05 | 2.63e-07 | 0.1140 | 0.00367 | Ongoing | bs=36, compile, grad_ckpt; val @ step 13888 |

---

## Successful / Ongoing Runs

| Exp ID | Tokenizer | Key Config | Min Val Dyn Loss | Min Val Recon | Min Val LPIPS | Duration | Status | Notes |
|--------|-----------|------------|------------------|---------------|---------------|----------|--------|-------|
| 2sqwg4wo | i9o9pcjj (P4-E04) | bs=36, compile, grad_ckpt | 0.1356 | 7.98e-05 | 0.00446 | Ongoing | Ongoing | val @ step 13888 |
| kovejwoy | 61zybkcy (P4-E07) | bs=36, compile, grad_ckpt | 0.1140 | 3.34e-05 | 0.00367 | Ongoing | Ongoing | val @ step 13888 |

---

## Key Findings

1. TBD

---

## Planned Runs

- TBD

---

## Ablations / Ideas to Try

- try separate LAM/dynamics optimizers
- mask ratio schedule vs fixed p_sample
- dynamics temperature sweep
- post-success: build latent-action → key mapping via one-step causal probe (latent index ↔ keyboard key) for control
