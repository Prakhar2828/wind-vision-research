# Wind VISION — portable results replay v1

By Prakhar Pandey. A small, independently runnable route into the Wind VISION research record. This release packages the actual frozen predictions, the complete 1,000-field synthetic test dataset, locked protocol, original reports, and new CPU evaluation code. It is not a retraining release.

## Run it

Python 3.11 or later; about 2 GB free memory and 1.5 GB free disk recommended. No GPU, notebook account or API key needed. Tested on macOS with Python 3.12.14, NumPy 2.3.5 and h5py 3.15.1.

```sh
python3 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python replay.py --verify-only
python -m unittest discover -s tests
python replay.py --output my-replay-receipt.json
```

The program fails with a nonzero exit status if a file digest, split, prediction, metric or bootstrap interval does not match. It writes a machine-readable receipt; it never executes a checkpoint or downloads code.

## What this reproduces

- SHA-256 validation of the bundled files, matching the originally frozen prediction and test-data digests.
- All eight methods' RMSE, 872 per-field errors, MAE, bias, temporal anomaly R², variance, adjacent-step and endpoint metrics, and four Hann-window spectral bands.
- Three classical predictions **regenerated from the 40 observed sensors and geometry only**, then compared to the frozen predictions. The baseline function does not receive hidden labels.
- Seven paired, 5,000-resample bootstrap intervals with the original seed.
- Four VAE optimizer-step timing medians and the 5.545842× A/D speedup, **recomputed from archived raw timings**, not newly benchmarked.

The selected observation-only estimated-mean + four-neighbor inverse-distance estimator obtains **0.681607188 m/s RMSE** on 872 held-out fields × 60 hidden sensors × 256 time samples (13,393,920 targets). It slightly outperforms the Flow + IDW hybrid (0.684631944 m/s). Reporting that simpler comparator is part of the result.

## What this does not reproduce

Neural predictions are rescored from frozen arrays, not regenerated. No VAE/flow retraining, CUDA timing, gradient parity or new simulator generation is run by this CPU tool. Training-set files are not required for results replay and are not included here. The optional, unchanged `WIND_VISION_RESEARCH_RC2.zip` release asset preserves the original model/phase-decoder candidate, its own per-file manifest, pinned NASA source, checkpoints and CUDA replay instructions. Its `verify_release.py` has been run for integrity; CUDA execution has not been rerun for this release. Historical development notebooks and full archive history remain separate from this portable bundle.

## Scope and provenance

This is one **synthetic scalar wind-velocity component**, a fixed 100-location geometry and a 40-observed/60-hidden spatial reconstruction task. Observed sensors supply complete time histories; this is not future-time weather forecasting. It does not establish flight suitability, real-weather accuracy or credible turbulence generation. The selected estimator still suppresses high-frequency variation.

Upstream constrained latent flow-matching research and the simulated benchmark originate with [NASA random_field_c-lfm](https://github.com/nasa/random_field_c-lfm), source commit `921e666c8d6060ae13bfd901dcc7f74f819d0eea`, and [NASA research record 20250007145](https://ntrs.nasa.gov/citations/20250007145). Prakhar's work covers reproduction, PhysicsNeMo integration, analytical-target vectorization, numerical checks, and reconstruction/physical-quality comparisons. The original generative algorithm is credited to the upstream authors.

Test HDF5 SHA-256: `e9c755f4d42e271a4bad6b82558d5a043835212dcdca8742a78889b93d035d31`.
Frozen predictions SHA-256: `23bd9fc7d5d0345d4a715b73a52094c4b5cd29398a2e759a96dfd501f6bd6372`.
The original evaluator ingested values/coordinates as float32 before float64 metric computation; this replay preserves that convention. Frozen predictions are float32; metric comparisons allow 2e-6 absolute tolerance, bootstrap intervals 1e-9.

[Research notebooks and scientific caveats](https://github.com/Prakhar2828/wind-vision-research) · [Portfolio research note](https://prakhar-pandey-portfolio.yoofis.chatgpt.site/wind-vision.html)

## Rights

The newly authored `replay.py` and `tests/test_release.py` are MIT licensed, copyright 2026 Prakhar Pandey. Upstream code, datasets, notebooks, historical artifacts and third-party components retain their existing terms and attribution. This permission does not relicense those materials or imply NASA endorsement.

Permission is granted, free of charge, to any person obtaining a copy of the newly authored files to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies, subject to inclusion of this copyright and permission notice. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY ARISING FROM USE OF THE SOFTWARE.
