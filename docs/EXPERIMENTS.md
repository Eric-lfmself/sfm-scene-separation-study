# Experiment guide

[Setup](GETTING_STARTED.md) · [Method](METHOD.md) · [Historical results](../RESULTS.md)

## Presets

| Preset | Inputs | Intended grouping | Historical summary |
| :--- | :--- | :--- | :--- |
| `pipes` | 14 ETH3D images | One place | Three arms |
| `terrace` | 23 ETH3D images | One place | Three arms |
| `courtyard` | 38 ETH3D images | One place | Three arms |
| `mixed3` | The three scenes above, 75 images | Three places | Three arms |
| `revisit` | `relief` + `relief_2`, 62 images | One place, two reference frames | Learned and shortlist SIFT |
| `uav_building` | 120 consecutive building train frames | One place | Three arms |
| `uav_mixed` | 120 building + 120 rubble train frames | Two places | Exploratory preset; no complete results table |

The earlier 251-image mixed-UAV note is not the current 240-image preset. I do not equate them.

## Compare the three arms

```bash
python src/run_experiments.py --experiment mixed3 --pipeline learned --seed 0
python src/run_experiments.py --experiment mixed3 --pipeline colmap-shortlist --seed 0
python src/run_experiments.py --experiment mixed3 --pipeline colmap-default --seed 0
```

I run arms as separate processes and archive each run immediately. For a controlled comparison I inspect the saved image and pair manifests, actual keypoint counts, camera models, runtime capabilities, and mapper options. Equal CLI settings alone are insufficient.

The commands above use the corrected shortlist policy. `--shortlist-policy legacy` reproduces the former pair-selection logic, not the entire former environment or experiment. I label new outputs separately and do not combine policies in one comparison.

## Revisit fusion

```bash
python src/run_experiments.py --experiment revisit --pipeline learned --seed 0
python src/run_experiments.py --experiment revisit --pipeline colmap-shortlist --seed 0
```

The runner treats both sessions as one physical scene for clustering/scene agreement, while preserving two reference frames for pose evaluation. A purity of 1 is not enough: the intended output is one model, and all-image coverage must be checked.

## Aerial sequence

```bash
python src/run_experiments.py --check-poses --uav-root ./data/mill19 \
  --uav-start 0 --pose-check-count 120

python src/run_experiments.py --experiment uav_building --pipeline learned \
  --uav-root ./data/mill19 --uav-start 0 --max-num-models 4 --seed 0
python src/run_experiments.py --experiment uav_building --pipeline colmap-shortlist \
  --uav-root ./data/mill19 --uav-start 0 --max-num-models 4 --seed 0
python src/run_experiments.py --experiment uav_building --pipeline colmap-default \
  --uav-root ./data/mill19 --uav-start 0 --max-num-models 4 --seed 0
```

The continuity diagnostic currently checks both building and rubble, so both sites must be prepared for that command. The reconstruction preset itself needs only its listed sites. Continuity and orthonormality are sanity checks; they cannot prove the camera-axis convention.

## Post-verification edge filtering

```bash
python src/run_experiments.py --experiment mixed3 --pipeline learned \
  --min-inliers 100 --seed 0
```

This first runs the unfiltered pipeline, copies its database, removes verified edges below 100 inliers, and reruns mapping. The report retains the original and filtered outputs. I compare both scene separation and lost registration: removing false links can also remove useful weak links. No complete filtered-run benchmark is claimed in the historical results.

## Paths and settings

| Option | Default / meaning |
| :--- | :--- |
| `--data-root` / `SFM_DATA` | ETH3D root, default `data/eth3d` |
| `--uav-root` / `SFM_UAV` | Mill 19 root, default `data/mill19` |
| `--work-root` / `SFM_WORK` | Outputs, default `work` |
| `--device` | `auto`, `cpu`, or `cuda` |
| `--seed` | 0; recorded, not a promise of bitwise reproducibility |
| `--shortlist-policy` | `corrected`; optional `legacy` |
| `--sim-th` | 0.3 descriptor-distance threshold |
| `--min-pairs` | Minimum-neighbour target 58, capped at N−1; threshold-selected neighbours can exceed it |
| `--exhaustive-if-less` | Exhaustive pairing when image count is ≤22 |
| `--num-features`, `--resize-to` | 4600 and 1024 for learned / shortlist SIFT |
| `--min-matches` | 20 putative matches for retaining a learned pair |
| `--min-model-size`, `--max-num-models` | 3 and 25; keep them identical across compared arms |
| `--uav-start` | First frame index, default 0 |

## Retain the evidence

New runs use `work/<experiment>/<pipeline>-<UTC timestamp>/`, including a separate staged `images/` directory. After input preparation succeeds, `report.json` records configuration, runtime, input/source hashes, pose data, metrics, and completion or failure status. Preparation failures can occur before a report exists. The database and reconstruction files remain beside the report. Preserve the whole directory and console log outside an ephemeral runtime.

The report improves traceability but is not a complete environment lock or a guarantee of bitwise replay. Dataset license terms also govern redistribution of images and derived artifacts. I keep private image collections out of Git.

Each run has its own image staging directory, so changing a later UAV frame window does not modify an earlier run's staged inputs. The staged files are links to the source images: keep the source data stable while runs are active.
