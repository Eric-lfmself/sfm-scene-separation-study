# Getting started

I keep a lightweight validation path separate from reconstruction runs. Commands assume the repository root and an authorized use under [LICENSE](../LICENSE).

## Inspect and validate without datasets

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python src/run_experiments.py --help
python -m unittest discover -s tests -v
python tools/validate_repository.py
```

`--help` needs only Python's standard library. The numerical regression suite uses NumPy and synthetic fixtures. It tests calculations and failure handling; it does not run pretrained models or reproduce the published tables.

## Reconstruction environment

```bash
python -m pip install -r requirements.txt
```

I recommend a separate Linux/CUDA environment for learned-front-end runs. Install a PyTorch build appropriate for that environment, following [PyTorch's installation selector](https://pytorch.org/get-started/locally/). A CPU run is supported but can be slow. Dependencies may fetch model weights on their first use.

PyTorch CUDA and COLMAP CUDA are separate capabilities. Explicit `--device cuda` in a SIFT arm requires both; a CPU-only pycolmap wheel cannot provide SIFT GPU execution. With `--device auto`, DINOv2 may use the GPU while SIFT uses CPU; the runner announces that split and records the capabilities.

The requirements describe the supported implementation range; they are not a complete lock of the historical Colab environment. The exact old LightGlue commit was not retained. Each new run records resolved package/model information where available. The recorded historical versions remain listed in [RESULTS](../RESULTS.md) and [porting notes](porting-notes.md), without a claim that every old wheel remains installable.

## Obtain ETH3D data

Install a system `7z` or `7zz` executable before the fetch step. The runner does not install system packages. Check the dataset's current terms at the [ETH3D source](https://www.eth3d.net/datasets).

```bash
python src/run_experiments.py --fetch eth3d \
  --scenes courtyard terrace pipes relief relief_2 \
  --data-root ./data/eth3d
```

This downloads images and calibration files. For manual preparation or Mill 19 data, see [Data sources](DATA_SOURCES.md).

## First reconstruction

Start with a small scene and the stock-setting SIFT baseline:

```bash
python src/run_experiments.py --experiment pipes --pipeline colmap-default \
  --data-root ./data/eth3d --work-root ./work --device cpu --seed 0
```

For the learned arm in a CUDA environment:

```bash
python src/run_experiments.py --experiment pipes --pipeline learned \
  --data-root ./data/eth3d --work-root ./work --device cuda --seed 0
```

Each run creates a distinct output directory beneath `work/<experiment>/`. I retain `report.json`, feature/match artifacts, the database, and reconstruction files together. Reports begin after input preparation succeeds; a preparation failure may only appear in the console. A successful lightweight check is not a successful reconstruction; inspect the report's completion status and registration coverage.

## Rebuild documentation figures

```bash
python tools/make_figures.py
python tools/validate_repository.py
```

The generator reads the committed historical summaries. It does not execute experiments. Editable SVGs and PNG previews are written to `figures/`. [Figure provenance](../figures/README.md)

## Common problems

| Symptom | Next check |
| :--- | :--- |
| `7z` / `7zz` missing | Install an extractor, then retry the explicit fetch command |
| No images or reference poses | Check the exact directory structure in [Data sources](DATA_SOURCES.md) |
| CUDA requested but unavailable | Check PyTorch CUDA and, for SIFT, `pycolmap.has_cuda`; or use `--device cpu` |
| Model download unavailable | Prepare the relevant Hugging Face / torch cache in the intended runtime |
| Very slow mapping | Inspect verified pairs, registration, and mapper options; a low model cap can also truncate the search |
| Historical table does not reproduce | Compare input manifests, shortlist policy, library versions, camera setup, and [known corrections](AUDIT.md) |

I do not overwrite the historical tables with a new run until its artifacts and interpretation have been checked.
