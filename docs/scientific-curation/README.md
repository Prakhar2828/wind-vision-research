# Wind VISION / PhysicsNeMo — scientific evidence curation v1

**Evidence cutoff:** reported work through Stage 45 (10 November 2026); **Stage 46 was not executed**. This curation was prepared from the 393 scientific-evidence files in the uploaded review ZIP and its 3,855-file parent manifest. It is an **evidence review**, not fresh model execution, independent reproduction or archive-member audit.

## Validation performed here

- Verified the uploaded ZIP's CRC integrity; 395 members (393 research files, the full master manifest and a review-pack index).
- Recomputed SHA-256 of **all 393 selected research files**, checked file sizes and exact match to both index and parent manifest: **all passed**.
- Parsed the seven Colab notebooks for code/output/error metadata; this is **not** notebook reexecution. See `NOTEBOOK_STATUS.csv`.
- Extracted staged result metrics from saved JSON and publication summary; these numbers are **reported measurements**, not independently recalculated from raw HDF5 arrays or model checkpoints.
- Left both the source ZIP and evidence files unmodified. Generated only new review documents.

## Main conclusions

1. **Final-test reconstruction**: A non-neural observed-mean + full-residual IDW method achieved **0.681607 m/s** on 872 one-time held-out synthetic fields; uncorrected Flow V3 recorded **1.550589 m/s**, a **56.04%** reduction in RMSE. This is a sparse-observation reconstruction finding, not proof of generative atmospheric realism. [F872]
2. **Computation**: On an L4, the 4-arm VAE optimizer-step comparison measured A original PyTorch **268.195 ms**, B unoptimized PhysicsNeMo **266.880 ms**, C optimized PyTorch **48.577 ms**, D optimized PhysicsNeMo **48.360 ms**. A/D is **5.5458×**, but A/B is **1.0049×**, so optimization is attributable to shared frequency vectorization, **not to framework choice alone**. [SPEED]
3. **Physical fidelity remains poor for native generative outputs**: fresh 16-field comparison gives simulator adjacent-step RMS **0.24172 m/s**, original PyTorch **0.00551 m/s**, and Flow V3 **0.02573 m/s**. These are descriptive small-sample numbers; reference fields were previously used in development. [PHYSICAL]
4. **Stage 45 Fourier adapter is a recorded negative result**: step-RMS retention improves **0.10412 → 0.48967**, but paired RMSE worsens **1.55594 → 1.56976 m/s**. The stated acceptance gate failed; no new VAE/flow production checkpoint was accepted. [S45]
5. **W1-E75 matched parametric-baseline challenge failed its neural-success rule**: the log-linear baseline had lower ACF and spatial correlation errors than the frozen neural model, while the neural model had slightly lower average logpower RMSE. `precommitted_matched_baseline_challenge_success = false`. [W75]

## What's included

- `CHRONOLOGY_AND_STAGE_LEDGER.md`: chronology and Stage 33–45 evidence including missing direct reports.
- `SCIENTIFIC_CLAIMS_AND_CAVEATS.md`: defensible claims, failed screens, key metrics, publication positioning.
- `REPRODUCIBILITY_GAPS_AND_CONTINUATION.md`: exact stopping point, archive gaps and bounded next-stage requirements.
- `SOURCE_KEY.md`: source-path+hash provenance for primary scientific claims.
- `REVIEWED_EVIDENCE_INDEX.csv`: source paths, hashes, sizes for all 393 reviewed files.
- `STAGE_REPORT_LEDGER.csv`, `W1_REPORT_LEDGER.csv`: mechanically generated report indexes.
- `NOTEBOOK_STATUS.csv`: notebook metadata, including historical embedded error outputs.
- `VERIFY_OTHER_ARCHIVES.py`: read-only optional local ZIP/TAR checks and dataset-structure inspection; run explicitly on Mac as documented.

## Critical limits

- Selected evidence excludes raw 35.378-GiB contents and most large checkpoint and prediction arrays; the parent manifest records them, but they were **not rehashed here**.
- Original NASA-inspired data are **synthetic**. Stage 71–75 research uses a further PSD-projected Fourier surrogate distinct from original UQpy generation. No claims of validation on measured atmospheric wind are justified.
- The numbered **W1-E46** experiment in the Nov 4 archive is **not** the newer unexecuted **Stage 46**.
- Neither a `PASS` status in a W1 experiment report nor a notebook containing outputs implies scientific superiority, unbiased holdout, or clean reproducibility.
- The official source lineage recorded in many reports is [NASA random_field_c-lfm](https://github.com/nasa/random_field_c-lfm/tree/921e666c8d6060ae13bfd901dcc7f74f819d0eea), commit `921e666c8d6060ae13bfd901dcc7f74f819d0eea`. Recorded attribution does not imply NASA endorsement.
