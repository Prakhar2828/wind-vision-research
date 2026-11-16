# Wind VISION Research

Prakhar Pandey's public research record for synthetic wind-field reconstruction, constrained latent flow matching, and NVIDIA PhysicsNeMo reproduction/optimization experiments. Upstream constrained latent flow-matching research is credited to [NASA random_field_c-lfm](https://github.com/nasa/random_field_c-lfm) and its authors.

## Start here: reproducible public results

**[Download the v1 results replay release](https://github.com/Prakhar2828/wind-vision-research/releases/tag/results-replay-v1)**. The self-contained portable ZIP includes the actual test data, frozen predictions, original protocol/reports and CPU replay code. No GPU, notebook account or API key is required.

```sh
gh release download results-replay-v1 --repo Prakhar2828/wind-vision-research --pattern WIND_VISION_PORTABLE_RESULTS_V1.zip
unzip WIND_VISION_PORTABLE_RESULTS_V1.zip
cd wind-vision-portable-v1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python replay.py --verify-only
python -m unittest discover -s tests
python replay.py --output my-replay-receipt.json
```

Windows: activate with `.venv\Scripts\Activate.ps1`. See [portable replay instructions](releases/portable-v1/README.md) for scope, memory, tolerances, provenance and attribution. The repository copy omits the large `data/` files; use the release ZIP for execution.

### Checked in a clean extraction

- All eight methods' reconstruction metrics and 872 per-field errors match the saved experiment.
- Three classical baselines regenerate from observed sensors only.
- All seven predeclared paired bootstrap intervals reproduce (5,000 resamples).
- Four timing medians and the 5.545842× ratio are recomputed from archived raw timing samples.
- Corrupted files and escaping manifest paths are rejected by tests.

The selected observation-only estimated-mean + IDW method achieves **0.681607188 m/s RMSE**, slightly ahead of the Flow + IDW hybrid (**0.684631944 m/s**) on 872 synthetic fields / 13,393,920 hidden targets. The simpler comparator wins; this does not imply the neural model beats classical reconstruction.

**Limits:** this is fixed-geometry, single-component, synthetic spatial reconstruction, not weather forecasting. CPU replay evaluates frozen neural predictions; it does not regenerate them or rerun GPU benchmarks, parity checks or model training. The release separately includes the unchanged original RC2 model/phase-decoder candidate, whose 108-file integrity check passes. CUDA replay requirements remain in that archive and were not rerun for this publication.

## Inspect the work

- [Portfolio research note, figures and interpretations](https://prakhar-pandey-portfolio.yoofis.chatgpt.site/wind-vision.html)
- [Portable evaluator and tests](releases/portable-v1)
- [Scientific claims and caveats](docs/scientific-curation/SCIENTIFIC_CLAIMS_AND_CAVEATS.md)
- [Chronology and stage ledger](docs/scientific-curation/CHRONOLOGY_AND_STAGE_LEDGER.md)
- [Seven notebooks with saved outputs](notebooks)
- [Research implementation](src)
- [Archive and restoration status](https://github.com/Prakhar2828/wind-vision-archive)

The new portable release supersedes earlier **results replay** availability gaps. Historical curation documents are retained as dated research records; their private-repository/upload-pending language should not be read as the current status of this public CPU release. The complete historical development archive is a different artifact and is not required to reproduce the released results.

No internal NASA systems, workforce data, maintenance evidence or intern exit deck is included in this release. Synthetic research and upstream public-source materials only. No NASA endorsement or flight qualification is implied.
