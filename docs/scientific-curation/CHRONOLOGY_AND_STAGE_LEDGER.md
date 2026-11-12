# Research chronology and Stage 33–45 evidence ledger

## Research tracks — do not conflate numbering systems

**October 28 baseline/source audit:** recovered NASA original code, checkpoint, reference reproduction reports, architecture and decoder ablation artifacts. The `MAC/projects/random_field_c-lfm_reference/research/` tree contains multiple preserved independent JSON audit records from October 28. These reports contain historical analyses and are not freshly rerun by this curation.

**November 3 P-numbered work:** the `Wind_VISION_P32_P43_Live_Snapshot_20261103.ipynb` notebook is included, along with other notebook revisions. `P32–P43` is a notebook milestone sequence, not the later `Stage 33–45` sequence. The older P29 evidence freeze's acceleration and conditional-MSE results must be scoped to their individual experiments; do not quote them as contemporary full-field generation or native framework gains without rebuilding their benchmarks.

**November 4–5 W1-E44 to E75:** 31 selected `scientific_report.json` files cover W1-E44 and W1-E46–E75 (E45 lacks a selected report). Their `status: PASS` generally records successful completion/validation, not universal model superiority. See `W1_REPORT_LEDGER.csv` for every exact path and hash. Important subtrends: VAE/decoder forensic work, spatial reconstruction, independent spectral generators, and parameter-conditioned neural-statistical surrogates. W1-E71 defines a separate, NASA-inspired PSD-projected surrogate, W1-E72/73 fit neural-statistical conditional models, W1-E74 shows improvements relative to specified baselines on fresh surrogate parameter combinations, and W1-E75 challenges the method against matched parametric alternatives and **fails its precommitted neural-success rule**. [W74] [W75]

**November 8 final completion/research release:** refreshed parity and VAE-step benchmarking, physical-quality diagnostics, phase-model tests and release acceptance are documented by `cell02`–`cell32` reports. Some early cells are negative or diagnostic, not end-to-end scientific success. The one-time 872-field evaluation is in the separate `matched_flow_v3.../final_test_872_one_time_v1` directory. It must not be treated as Stage 45's independent test. [F872] [SPEED] [REPLAY]

**November 10 Stages 33–45:** distinct set of learned spectral and decoder-capacity research checkpoints, ending with failed Stage 45 adapter pilot. Some stages have reports in the parent manifest that were not selected into this compact review package; do not invent their outcomes.

## Stage-by-stage documented results

| Stage | Evidence in reviewed package | Finding / status | Evidence |
|---|---|---|---|
| 33 | Full-scale learned-eigenspectrum hybrid report; 4,500 train / 500 internal development | Mean spectral-band ratios near 1 and step RMS near 1, **but spatial correlation off-diagonal MAE 0.5203**, exceeding 0.02 screen. Overall screening **failed**. This is not original NASA c-LFM or learned latent flow. | [S33] |
| 34 | Two forensic report snapshots, one indexed in [S34] | Spectral-allocation α ablation diagnosed worsening spatial correlation as learned eigenvalue allocation increased (α=1 analytical MAE ≈ 0.5167 versus training-only empirical analytical baseline). No independent test. | [S34] |
| 35 | Saved extension evaluation | Descriptive replay of Stage 34 extension; internal development HDF5 not available in that execution. Does not overturn failed Stage 33 screen. | [S35] |
| 36 | Two entries named Stage 36 in master manifest, but no direct Stage 36 report selected | **Not independently reviewed here.** Stage 37 explicitly says its ablation does not replace the Stage 36 screen. | [S37] |
| 37 | Four-seed innovation-law ablation | Fixed-amplitude phase shell produced essentially zero between-field total-energy CV. Complex Gaussian coefficient sampling restored energy CV ≈0.437 but did not prove realistic wind variability. Diagnostic only. | [S37] |
| 38 | No selected Stage 38 JSON report | Stage 39 says it transcribed Stage 38 stdout reference statistics; **independent detailed Stage 38 review remains open**. | [S39] |
| 39 | Training-only spectral-band variability audit | Fixed-shell spectral-band CVs fit training reference more closely than Gaussian alternatives, but total energy remains artificially constant; Gaussian variability exceeds training reference in multiple bands. Source training statistics transcribed, not newly loaded. | [S39] |
| 40–41 | No explicitly stage-named reports in reviewed evidence/parent manifest | **Unresolved as separate experimental milestones.** Search full notebooks/scripts before assigning status. | — |
| 42 | Stage 42 reference-input NPZ present in parent manifest, no Stage 42 experiment report in review pack | Hash-pinned training subset exists; not sufficient to infer Stage 42 outcome. | — |
| 43 | Frozen sparse-encoder VAE/Flow diagnostic | On 64 training-pool reconstructions, the posterior mean generated very weak high-/mid-band power (≈4×10⁻⁷ / 10⁻⁵ relative), and observed 40-sensor paired RMSE ≈1.591 m/s. No independent test. | [S43] |
| 44 | Frozen trunk capacity/oracle | 512-feature decoder trunk oracle reduces paired RMSE to ≈1.213 m/s under lenient regularization but high-band power remains ≈2.32×10⁻⁵ of target. Oracle coefficients use complete field targets and aren't realizable from branch(z) by assumption. | [S44] |
| 45 | Trained Fourier residual adapter versus raw-time control | Adjacent-step RMS ratio 0.10412→0.48967, high-band ratio 3.93×10⁻⁷→0.2038, but RMSE 1.55594→1.56976 m/s and both RMSE gates failed. `pilot_pass: false`. | [S45] |
| 46 | **Never executed in this Stage sequence** | No Stage 46 results exist to report. Historical W1-E46 is a different experiment. | — |

## Reproducibility implication

Stage 33's frequency-level preservation does **not** establish appropriate spatial covariance or generative success. Stage 37 shows that restoring stochastic energy variability alone can worsen band-wise variability. Stages 43–44 support a decoder/information-bottleneck diagnosis, while Stage 45 shows that a Fourier residual can restore variation but does not automatically improve paired field accuracy. These are evidence-supported *diagnoses and negative controls*, not a validated improved physical generator.

## Source paths

All bracketed keys above refer to master-relative files and their exact SHA-256 hashes in `SOURCE_KEY.md`. Use `STAGE_REPORT_LEDGER.csv` for the path/hash of each reviewed direct stage report.
