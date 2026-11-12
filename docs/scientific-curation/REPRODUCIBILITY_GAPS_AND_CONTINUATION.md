# Reproducibility gaps and bounded continuation

## Current stopping point

- Most recent documented **Stage 45** on 10 Nov 2026 was an internal Fourier adapter pilot, not an accepted training or inference model. `pilot_pass = false`, while original VAE and flow were frozen. Evidence key **[S45]**.
- **Stage 46 unexecuted**. The W1-E46 numbered experiment from November 4 is unrelated.
- Do not choose a new production checkpoint just because its spectral metrics improved: Stage 45 RMSE gates failed.

## Unresolved material questions before reproducibility certification

1. **Multiple files called `wind_test_data.hdf5` differ significantly.** The original repository copy and the recovered-runtime copies are different byte sets, not aliases. Their parent-manifest entries are:
- `11_RAW_SOURCE_SNAPSHOTS/MAC/projects/random_field_c-lfm_reference/data/wind/wind_test_data.hdf5` — **1,031,040,136 bytes**, SHA-256 `ab56534536f7ae209e5a7b47f6f46ac5dc4743a3ae488f402e0aa846e90dc943`
- `11_RAW_SOURCE_SNAPSHOTS/MAC/projects/wind_vision_storage/WIND_VISION_MASTER_20261109/mac_only/recovered_original_test/wind_test_data.hdf5` — **206,217,920 bytes**, SHA-256 `e9c755f4d42e271a4bad6b82558d5a043835212dcdca8742a78889b93d035d31`
- `11_RAW_SOURCE_SNAPSHOTS/MAC/projects/wind_vision_storage/WIND_VISION_MASTER_20261109/runtime_inputs/wind_test_data.hdf5` — **206,217,920 bytes**, SHA-256 `e9c755f4d42e271a4bad6b82558d5a043835212dcdca8742a78889b93d035d31`
- `11_RAW_SOURCE_SNAPSHOTS/MAC/projects/wind_vision_storage/recovered_original_test/wind_test_data.hdf5` — **206,217,920 bytes**, SHA-256 `e9c755f4d42e271a4bad6b82558d5a043835212dcdca8742a78889b93d035d31`

   The original repository version is ~983 MiB, recovered version ~197 MiB. We have **not opened the raw HDF5 datasets** in this 9.29-MiB evidence package. Determine dataset keys, shape, dtype, coordinates, field IDs, train/test split, provenance, and whether any transpose/order transforms are intentional before claiming exact baseline dataset parity. Do not label a file corrupted solely on size/hash.

2. **Archive-member verification remains separate.** The 35.378-GiB master audit hashes outer ZIP/TAR files; it does not establish that all archive members are valid or free of path traversal. Validate members read-only; record CRC/checksum failures and blocked unsafe names. The 9.29-MiB uploaded review ZIP itself passed `testzip()`.
3. **Report vs full model replay.** This curation verified all 393 selected files against their SHA-256 index but did **not** load checkpoint tensors, rerun seeds, or recompute evaluation arrays. The `cell32` saved report says acceptance passed, but `clean_python_environment_created=false` and original VAE/flow were not replayed in that cell. [REPLAY]
4. **Four of the seven notebooks contain historic embedded error outputs (13 error records in total)**, detailed by `NOTEBOOK_STATUS.csv`. An error output can remain in an otherwise successful research notebook. Inspect cell context before cleaning anything; preserve original notebook copies.
5. **Experiment chronology completeness.** No direct Stage 38 or Stages 40–41 reports were selected. Stage 36 and Stage 42 data are in the parent manifest but weren't incorporated as full reports. Find history in notebooks, control manifests and long source reports; do not infer execution or success from stage names alone.
6. **Data leakage and test scoping.** Keep original 872 one-time final-test protocol frozen. Stage 45's reported 16 evaluation fields were withheld from adapter optimization but came from the original training pool used to pretrain the frozen VAE; this is *not independent-test validation*. W1-E71–E75 surrogate tests are a different scientific distribution and should never be pooled with original NASA-simulator final-test statistics. [F872] [S45] [W75]
7. **Deployment gaps.** No measured atmospheric wind, alternate sites/geometries, uncertainty calibration on real observations, real-time end-to-end budget or downstream mission operations deployment verified. Claims should be explicitly limited to synthetic research models.

## Proposed Stage 46 design brief — NOT an executed result

**Hypothesis:** the Stage 45 Fourier adapter boosts high-frequency energy and step variation but compromises paired accuracy because the adaptation objective underweights field reconstruction/conditional constraints. Test whether a reconstruction-regularized and spatially informed spectral residual can satisfy **both** predictive and temporal-fidelity metrics without leaking withheld field values.

**Pre-registration before running:**

- Freeze source commit, dataset hash/splits and Stage 45 control checkpoint; document full sensor geometry and preprocessing. Keep the 872-field one-time final test sealed for exploratory work.
- Reproduce Stage 45 matched control with identical field IDs, seeds, 160-step budget, and measurement protocol before any new model training.
- Limit to a small factorial ablation chosen in advance: Fourier residual alone, Fourier residual with MSE regularization, Fourier residual with observation-consistency penalty and a small-capacity matched raw-time control. Same optimization budget/parameter-count accounting.
- Primary gate: require **lower paired RMSE than the frozen model and the matched raw-time control** (target should be fixed *a priori*, e.g. ≥2% vs frozen and ≥1% vs control as in Stage 45), not merely increased step RMS.
- Co-primary quality gates: no >3× upper-band power overshoot, no excess adjacent-step RMS, and specified retained lower/mid-frequency power ranges. Treat gate failure as negative result.
- Measure paired reconstruction errors only on proper internal held-out examples. After a successful internal gate and frozen method, plan a **new untouched dataset / geometry / synthetic seed family**, not reuse the 872-field historical one-time final test as fresh.
- Separate PhysicsNeMo-vs-PyTorch framework overhead from any algorithmic vectorization gain; matched baselines and GPU timing protocol required.
- Archive runnable code, exact environment, RNG seeds, model/source hashes, logs, checkpoints, figures, test IDs, and failures; do not overwrite old snapshots.

**Stop rule:** If controlled reconstruction RMSE gate fails, do not promote candidate or run an independent final test; preserve failed run as evidence and revise hypothesis. Do not assert this suggested Stage 46 plan is the previously agreed experimental specification; it is a review-driven recommendation.

## Transfer curated documents into master directory

After downloading `WIND_VISION_SCIENTIFIC_CURATION_V1.zip`, extract its folder under the master archive's documentation area (not over `11_RAW_SOURCE_SNAPSHOTS`). For example on Mac, with the ZIP in Downloads:

```bash
mkdir -p "$HOME/Documents/WIND_VISION_RESEARCH_MASTER/07_SCIENTIFIC_CURATION"
unzip -n "$HOME/Downloads/WIND_VISION_SCIENTIFIC_CURATION_V1.zip" -d "$HOME/Documents/WIND_VISION_RESEARCH_MASTER/07_SCIENTIFIC_CURATION"
```

`unzip -n` skips any existing file; this protects earlier drafts. This curation is an additional documentation snapshot, not a replacement for the full original-file archive.
