# Recorded experimental summaries

I transcribed these tables from [`RESULTS.md` at commit `7e0d60c`](source-results-7e0d60c.md) to make the figures editable, inspectable and reproducible. **These files are historical summary data, not outputs from a new experiment.** The original hosted-runtime logs and reconstructions are not included, so I cannot independently audit the original measurements from these tables alone.

The immutable source snapshot preserves the original wording as well as the numbers. Some of its interpretations are too strong; the current [results discussion](../../RESULTS.md) gives the corrected scope. In particular, I do not interpret these configurations as a matcher-only causal ablation, treat different run contexts as controlled replicates, or equate a large full-rotation error with a reversed optical axis.

## Data files

| File | Contents | Units / missing values |
|---|---|---|
| [single-scenes.csv](single-scenes.csv) | Nine ETH3D scene × configuration rows | Position in millimetres; rotation in degrees; mapping in seconds; normalized AUC |
| [mixed-scenes.csv](mixed-scenes.csv) | Three configurations on 75 mixed images | Counts; fractions; seconds |
| [mixed-scene-poses.csv](mixed-scene-poses.csv) | Nine within-scene summaries from the mixed run | AUC@5° and position in millimetres; blank orientation-outlier counts mean unreported, not zero |
| [revisit.csv](revisit.csv) | Two configurations on 62 images from two sessions of one place | Counts; fractions; seconds; default configuration unreported |
| [uav.csv](uav.csv) | Three configurations on 120 Mill 19 building frames | Position in normalized units `u`, **not metres**; rotation in degrees; seconds |
| [verified-link-ranges.csv](verified-link-ranges.csv) | Within-scene and false cross-scene link summaries from the mixed run | Inlier counts; `p10` / `p90` are distribution percentiles, not uncertainty intervals |
| [revisit-links.csv](revisit-links.csv) | True cross-session and within-session link counts and medians | Additional distribution quantiles are unavailable in this source |
| [retrieval.csv](retrieval.csv) | Reported shortlist sizes | Percentages retained as rounded source values |
| [legacy-figure-data.json](legacy-figure-data.json) | Exact arrays/statistics from the previous figure generator | A separate, unverified snapshot; excluded from current figures |

The source table's `same_scene_verified_pairs / all_verified_pairs` counts are preserved alongside its rounded pair-precision values. For example, `889 / 1039` is displayed as the recorded `0.8556`; I do not replace the source's precision with additional digits. The mixed-run per-scene AUC values correspond to AUC@5° in the surrounding results and evaluation output.

## What the figure generator does

[`tools/make_figures.py`](../../tools/make_figures.py) reads these CSV files without filtering or averaging scenes. It converts registration counts to percentages while keeping the original numerator and denominator as labels. It places position errors and inlier ranges on explicitly labeled base-10 logarithmic axes. Other numerical axes are linear; bars start at zero. It leaves the unreported default revisit result blank with `NR`, does not draw error bars without replicate data, and does not infer histograms from percentile summaries.

I keep the datasets' position units separate: ETH3D is in millimetres and Mill 19 uses Mega-NeRF normalized coordinates. The plots show only CPU mapping time. Historical front-end timings mix CPU and GPU work and are not suitable for a direct speed comparison. All plot inputs and transformations are listed in [`figures/manifest.json`](../../figures/manifest.json), together with source-file SHA-256 digests.

## Why the old histogram is archived

The earlier generator contains within-scene `n=890`, median `302` and maximum `3379`; the results table contains `n=889`, median `308` and maximum `3376`. Its false cross-scene maximum is `83`, while the table reports `85`. I cannot establish that these snapshots describe the same run. I therefore preserve both instead of changing a bin to force agreement.

The legacy histogram and its four-population range plot remain in [`figures/archive/`](../../figures/archive/), with the original generator. The extra revisit quantiles and cross-site UAV population in that script are not supported by the current results table and do not enter the new figures. Its suggested 100-inlier threshold is not validated as a general decision rule. The new [link-range plot](../../figures/link-ranges.svg) uses only the two complete distribution summaries recorded in the results table.

## Evidence still needed

To upgrade these summaries to independently traceable results, I still need fresh runs with configuration and environment manifests, complete logs, per-pair geometric-verification records, camera poses, evaluation outputs and reconstruction assets. A plotting refresh does not supply those missing experiments.
