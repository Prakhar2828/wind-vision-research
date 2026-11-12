# Claims, quantitative findings, and caveats

## A. Frozen one-time held-out reconstruction, 872 simulator fields — [F872]

| Method | RMSE on unobserved sensors (m/s) | Temporal anomaly R² | Step RMS / reference |
|---|---:|---:|---:|
| Observed-estimated vertical mean + full residual IDW | 0.681607 | 0.88438 | 0.73039 |
| Known-mean full residual IDW (privileged simulator parameters) | 0.684121 | 0.88438 | 0.73039 |
| Flow V3 + full residual IDW | 0.684632 | 0.88485 | 0.73045 |
| VAE + full residual IDW | 0.686786 | 0.88472 | 0.73044 |
| Flow V3 uncorrected | 1.550589 | 0.31723 | 0.09837 |
| VAE uncorrected | 1.571237 | 0.30849 | 0.09720 |
| Direct wind IDW | 1.726863 | 0.88438 | 0.73039 |

Selected result: **0.681607 m/s**. Difference versus uncorrected Flow V3 **56.04% reduction**. Compared with Flow V3 + residual IDW, absolute advantage **0.003025 m/s**, paired field-bootstrap 95% interval **[0.002620, 0.003416] m/s**; winning on **612/872** fields. The selected method estimates mean vertical profiles from measurements and interpolates residual wind; it is **not a trained deep model**. The 128-field development subset and 872-field frozen final subset come from the same synthetic simulator, geometry and distribution. Claim only simulator-family validation; cannot claim real-world operational performance. [F872]

## B. Four-arm matched-step benchmark — [SPEED]

| Arm | Median full VAE optimizer-step (ms) |
|---|---:|
| A original PyTorch | 268.195 |
| B unoptimized PhysicsNeMo | 266.880 |
| C optimized PyTorch | 48.577 |
| D optimized PhysicsNeMo | 48.360 |

- A/D **5.5458×**, A/B **1.0049×**, A/C **5.5210×**, B/D **5.5187×**.
- Input `(2,40,256)`, coordinate `(2,10240,4)`, NVIDIA L4, 4 warmup and 16 measured steps. Twenty matched optimizer updates, weights, gradients, and Adam moments were reported exact under the tested protocol. This is **not** a multi-GPU scaling study, full training campaign, or full-generation throughput gain. Full peak memory includes four resident models. [SPEED]

## C. Generator realism caveat — [PHYSICAL]

| Source | Temporal variance | Step RMS (m/s) |
|---|---:|---:|
| Simulator reference | 3.2106 | 0.24172 |
| Original PyTorch generation | 0.0407 | 0.00551 |
| PhysicsNeMo Flow V3 generation | 0.8395 | 0.02573 |

Each model used 16 fresh generated samples; reference pool consisted of 512 earlier-used simulated fields. This is **descriptive**, not blinded final validation; the model architectures and trained checkpoints differ, so it is **not framework parity**. [PHYSICAL]

## D. Physical-fidelity research gates — [S33] [S43] [S44] [S45]

- **Stage 33** analytical power-band ratios around unity did not rescue a spatial correlation MAE of **0.5203**. The `spatial_correlation_mae_le_0p02` gate was false; entire stage gate was false. [S33]
- **Stage 43** frozen 40-observed-sensor posterior means had paired RMSE **1.5912 m/s**; the upper-frequency spectral ratios were extremely small. Training-pool diagnostic only. [S43]
- **Stage 44** best listed oracle trunk LS RMSE **1.2129 m/s**, but high-band ratio **0.00002322**; oracle access to complete ground truth invalidates it as an attainable predictor. [S44]
- **Stage 45** control RMSE **1.55594**, adapter RMSE **1.56976 m/s**; adapter relative step RMS **0.48967** versus frozen baseline **0.10412**. Two RMSE criteria false, overall `pilot_pass = false`. Frozen VAE and latent flow not modified. [S45]

## E. Separate W1 condition-surrogate study — [W74] [W75]

- W1-E74: report claims its own precommitted criterion passed across fresh parameter-combination draws, with power, ACF and spatial-correlation improvements counted on five conditions; these are within the known E71 surrogate family, not original NASA/UQpy or real wind. [W74]
- W1-E75: more stringent comparison against non-neural matched parametric models yields **`precommitted_matched_baseline_challenge_success = false`**. Mean ACF MAE over six conditions: neural **0.004546**, log-linear **0.003133**. Mean spatial-correlation MAE: neural **0.003679**, log-linear **0.002182**. Neural had somewhat better log-power RMSE (**0.09485** vs **0.10291**), so results are mixed rather than a blanket failure. Oracle remains best/known by construction. [W75]

## Recommended publication framing

**Strongest defensible claim:** systematic, auditable study of temporal/spectral fidelity failure modes in PhysicsNeMo-compatible NASA-inspired wind reconstruction, with a rigorously controlled computational optimization and a held-out synthetic reconstruction comparison showing benefits of measurement assimilation over uncorrected neural generation.

**Do not claim:** NASA has deployed this; superiority on measured atmospheric wind; native PhysicsNeMo causes a 5.5× speedup; Fourier adapter passed; Stage 33 is original c-LFM; learned surrogate unconditionally beats parametric methods; any Stage 46 result; or publication acceptance is assured.
