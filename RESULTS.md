# Recorded experimental results

[Overview](README.md) · [Method and metrics](docs/METHOD.md) · [Result data](results/reported/README.md) · [Evidence audit](docs/AUDIT.md)

I preserve the numerical summaries from repository revision `7e0d60c`. They describe earlier experiments, **not new runs performed during this repository revision**. The original per-run logs, databases, and reconstructed poses are unavailable here. My local notes record that hosted runtimes were reclaimed before those artifacts were retained. The tables are therefore historical summaries pending source recovery or fresh reproduction.

The recorded environment was a Colab Tesla T4 with Python 3.13.15, torch 2.11.0+cu128, pycolmap 4.2.0, kornia 0.8.3, and transformers 5.16.1. I have not independently reconstructed that environment. I do not supply confidence intervals because the available summaries do not support them.

## Configurations and controls

| | pair selection | detector | matcher | budget |
|---|---|---|---|---|
| `learned` | DINOv2 shortlist | ALIKED | LightGlue | cap 4600 (realized 4600), 1024 px |
| `colmap-shortlist` | DINOv2 shortlist | SIFT | nearest neighbour + ratio test | cap 4600 (realized ~7592), 1024 px |
| `colmap-default` | exhaustive | SIFT | nearest neighbour + ratio test | stock cap 8192 (realized ~11988), 3200 px |

The first two arms use the same shortlisting procedure, nominal resolution, and feature-cap setting. They change both the detector and matcher. Realized feature counts, camera initialization, and feature/verification paths are not fully matched. The shared mapper alone does not make this a component-isolating comparison.

The realized counts in the configuration table refer to the recorded aerial run; they are not universal counts across scenes. On `mixed3`, SIFT / shortlist recorded 4247 keypoints per image, fewer than ALIKED's 4600. On the aerial sequence it recorded 7592. I preserve scene-specific counts in the corresponding sections.

The stock-setting baseline changes pairing, resolution, and feature cap together. I use it as a practical reference, not as a resolution-only or matcher-only ablation.

## Timing boundary

In the recorded environment, SIFT extraction and nearest-neighbour matching ran on CPU while ALIKED and LightGlue ran on GPU. Their front-end timings cannot establish algorithmic speed superiority. I retain the values as environment-specific observations.

The mapping calls used the same CPU path. Mapping times can be compared within those recorded runs, but remain descriptive: I have no retained repeat timing distribution and do not extrapolate them to a 12,000-image survey. Current pycolmap builds may differ; the runner records runtime capabilities for new runs.

## Metric definitions and historical limitations

- **AUC@5/10/20°:** normalized area under the empirical relative-pose error CDF. Pair error is the maximum of rotation error and translation-direction error. The implementation uses a sign-invariant translation direction, `acos(abs(dot(t_pred, t_gt)))`, after normalizing both translation vectors to unit length. Missing or separately reconstructed pairs receive 180°. I evaluate within each reference coordinate frame.
- **Absolute position and rotation:** a Sim(3) alignment to reference poses followed by medians, using only each scene/session's dominant reconstructed component. These errors do not include unregistered cameras; registration and component coverage must accompany them.
- **Inlier ratio (legacy):** the old code pooled RANSAC inliers and putative matches only over pairs with nonzero verified inliers. It did **not** include all matched pairs in the denominator. All historical values below retain that definition. The corrected runner reports the all-pair ratio separately from `inlier_ratio_verified_pairs`.
- **Pair precision:** fraction of geometrically verified image pairs whose labels identify the same physical scene. This measures scene agreement, not ground-truth pixel correspondence accuracy. The historical revisit labels represented sessions and were unsuitable for this metric.
- **Purity:** dominant scene-label count summed over reconstructed components, divided by registered images. Purity alone can reward over-fragmentation; I report model count and coverage beside it.
- **Rotation error >170°:** a full SO(3) orientation error. A large roll can satisfy this condition while preserving the optical axis. I do not call these cameras backward-facing without a separate optical-axis test.

## Single scenes

| Scene | Config | Registered | AUC@5 / 10 / 20° | Median position | Median rotation | Mapping |
|---|---|---|---|---|---|---|
| `pipes` (14) | learned | 14 / 14 | 0.890 / 0.947 / 0.973 | 6 mm | 0.255° | 10.4 s |
| | colmap-shortlist | **8 / 14** | 0.241 / 0.271 / 0.289 | 12 mm | 0.624° | 7.8 s |
| | colmap-default | 14 / 14 | **0.928 / 0.964 / 0.982** | **5 mm** | **0.238°** | 11.4 s |
| `terrace` (23) | learned | 23 / 23 | 0.893 / 0.946 / 0.973 | 16 mm | 0.430° | 36.0 s |
| | colmap-shortlist | 23 / 23 | 0.931 / 0.966 / 0.983 | 7 mm | 0.285° | 16.8 s |
| | colmap-default | 23 / 23 | **0.932 / 0.966 / 0.983** | **7 mm** | **0.285°** | 31.9 s |
| `courtyard` (38) | learned | 38 / 38 | 0.643 / 0.698 / 0.726 | **1111 mm** | 5.331° | 154.2 s |
| | colmap-shortlist | 38 / 38 | 0.891 / 0.945 / 0.973 | **29 mm** | 0.168° | 48.4 s |
| | colmap-default | 38 / 38 | **0.899 / 0.949 / 0.975** | 31 mm | 0.180° | 105.6 s |

On `courtyard`, the recorded standalone comparison gives 1111 mm for learned, 29 mm for SIFT / shortlist, and 31 mm for SIFT / stock settings. Other recorded learned values are 516 mm and 849 mm; the 849 mm result is explicitly from `mixed3` below. I cannot treat these values as three controlled repeats on identical input. The available summaries motivate a repeatability study rather than establish one.

On `pipes`, learned registers 14/14, SIFT / shortlist registers 8/14, and SIFT / stock settings registers 14/14. The stock arm changes several factors, so these rows alone cannot attribute the difference exclusively to resolution.

Recorded SIFT keypoint summaries: 1822–5542 per image for the shortlist arm and 11393–12511 for the stock-setting arm in the single-scene summaries. The original summaries do not retain the per-image vectors needed to recompute these ranges.

![Single-scene comparison](figures/results-single-scenes.png)

## Different places: `mixed3`

I combined 75 images from `courtyard`, `terrace`, and `pipes`. The intended output is three scene-specific reconstructions.

| | Registered | Clusters | Purity | Inlier ratio | Pair precision | Mapping |
|---|---|---|---|---|---|---|
| learned | 75 / 75 | **2** | **0.6933** | 0.8396 | **0.8556** (889 / 1039) | 543.8 s |
| colmap-shortlist | 69 / 75 | **3** ✓ | **1.0000** | 0.8532 | **1.0000** (559 / 559) | 71.0 s |
| colmap-default | 75 / 75 | **3** ✓ | **1.0000** | 0.8946 | **1.0000** (614 / 614) | 199.7 s |

Per-scene accuracy in the mixed run follows. Each cell begins with AUC@5°. Absolute errors describe the dominant component for that scene.

| Scene | learned | colmap-shortlist | colmap-default |
|---|---|---|---|
| `courtyard` | 0.678 / 849 mm | 0.906 / 30 mm | 0.898 / 34 mm |
| `terrace` | 0.101 / 2246 mm, **8 cameras >170° off** | 0.928 / 8 mm | 0.933 / 7 mm |
| `pipes` | 0.861 / 8 mm | 0.209 / 32 mm (57% registered) | **0.927 / 5 mm** |

The historical table's phrase “8 cameras >170° off” means full rotation error, not a verified reversal of viewing direction.

The learned configuration merged `courtyard` and `terrace`; the SIFT configurations produced three models. This associates the failure with the tested learned front end. It does not isolate LightGlue from ALIKED, camera initialization, verification behavior, or their interactions.

The stock SIFT run recorded 11928 keypoints per image and zero cross-scene verified pairs. More keypoints alone were therefore not sufficient to produce the same false links in that configuration. This is weaker than identifying the matcher as the sole cause.

The learned summaries report 296 `courtyard`–`terrace`, 19 `courtyard`–`pipes`, and 11 `pipes`–`terrace` cross-scene pairs before geometric verification; 150 cross-scene pairs survived verification. Repeated architectural appearance is a possible explanation, but I do not have retained correspondence images to establish the mechanism.

### Verified-link strength

The entries are inlier counts per verified image pair. These are quantiles, not a recoverable raw distribution.

| Link type | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|
| within-scene | 889 | 15 | 33 | 308 | 1419 | 3376 |
| cross-scene (false) | 150 | 15 | 16 | 22 | 54 | 85 |

Older figure inputs disagree with this table (`within n=890`, median 302, max 3379; false-link max 83). I retain those assets in the [figure archive](figures/README.md) with their discrepancy documented. I do not combine them with this table or reconstruct a histogram from these quantiles.

![Mixed-scene and revisit comparison](figures/results-mixed-scenes.png)

## Same place, two sessions: `revisit`

I combined 31 `relief` and 31 `relief_2` images of the same interior. The intended output is one reconstruction. The two sessions have independent reference coordinate frames, so I evaluate their pose errors separately.

The historical pair-precision implementation treated the session names as scene identities. That would count legitimate cross-session links as errors. I omit that metric here and separate physical-scene labels from reference-frame labels in the corrected runner.

| | Registered | Clusters | Inlier ratio | Mapping |
|---|---|---|---|---|
| learned | 62 / 62 | **2** | 0.7154 | 147.5 s |
| colmap-shortlist | 62 / 62 | **1** ✓ | **0.9053** | 116.0 s |

| Scene | learned | colmap-shortlist |
|---|---|---|
| `relief` | 0.449, 58% in dominant reconstruction | **0.674, 100%**, 132 mm, 1.419° |
| `relief_2` | 0.289, **179.037°**, **20 of 20 cameras >170° off** | **0.899, 8 mm, 0.337°** |

The original `relief_2` cell uses “20 of 20 cameras >170° off.” This covers the 20 cameras evaluated in that session's dominant model, not all 31 session images. The recorded 179.037° value is full rotation error; optical-axis reversal was not measured. The learned position error was described only as approximately one metre, so I do not add a more precise value.

An additional historical summary records 370 cross-session verified links with median 51 inliers, versus 412 within-session links with median 452. Their lower inlier counts suggest a link-strength difference; they do not by themselves explain why the mapper produced two models. The stock-setting revisit arm was not reported.

## Scene separation and fusion together

| Situation | Correct answer | learned | SIFT + NN |
|---|---|---|---|
| three different scenes in one folder | 3 clusters | **2**: merges different places | 3 ✓ |
| one place photographed twice | 1 cluster | **2**: splits the two visits | 1 ✓ |

The two tests probe different errors. Their combined result is specific to these inputs and configurations. It is not a general ranking of learned and classical matchers.

## Aerial sequence: Mill 19 building

The recorded input contains 120 consecutive frames from one `building` flight, at 4608 × 3456 pixels per frame. My notes describe two flight strips and optical-axis changes near frames 55 and 113. The selected poses are not retained in this checkout, so I cannot independently recheck those indices.

**Units.** The available metadata was interpreted in a normalized Mega-NeRF frame, denoted `u`; no metric scale factor was retained. The recorded median consecutive-frame baseline is 0.041 u. Thus 0.001 u is about 1/41 of that baseline and 0.098 u is about 2.4 baselines. I do not compare these values directly with ETH3D millimetres.

**Reference-pose checks.** The notes record a 2.3e-07 orthonormality error and 0.26° median consecutive optical-axis change. Such diagnostics can detect malformed or abrupt poses, but cannot establish the correct axis convention: a consistent orthogonal axis transformation preserves smoothness and orthonormality. Independent projected-point checks remain on the roadmap.

| | Registered | Clusters | AUC@5 / 10 / 20° | Median position | Median rotation | Mapping | Inlier ratio |
|---|---|---|---|---|---|---|---|
| learned | 120 / 120 | 1 ✓ | 0.593 / 0.621 / 0.634 | **0.098 u** (2.4 baselines) | **64.33°**, 27 cameras >170° off | 2211.5 s | 0.7655 |
| colmap-shortlist | 120 / 120 | 1 ✓ | **0.947 / 0.974 / 0.987** | **0.001 u** (1/41 baseline) | **0.94°**, none | **911.6 s** | 0.9311 |
| colmap-default | 120 / 120 | 1 ✓ | **0.953 / 0.977 / 0.988** | **0.001 u** | **0.52°**, none | 1517.7 s | **0.9412** |

Every arm registers all 120 images into one model. The recorded reference-pose errors distinguish the outputs where coverage and cluster count do not. The 27 cameras above 170° have large orientation errors; a quarter would be 30 cameras, so the recorded fraction is 22.5%.

The SIFT / shortlist control runs through the same evaluation and returns 0.94° median rotation error. This reduces concern about some shared evaluation mistakes; it does not establish that all conventions and pipeline-specific data paths are correct.

SIFT / shortlist has 7592 recorded keypoints per image, versus 4600 for ALIKED; the stock arm has 11988. I cannot rule out feature budget or other confounders without a matched-count ablation.

The learned arm supplied 3367 verified pairs and the shortlist SIFT arm 2388. Their recorded mapping times are 2211.5 s and 911.6 s. This is an association within one run per arm, not an established scaling law.

![Aerial results](figures/results-uav.png)

### Aerial front-end stage timings

CPU/GPU placement differs across arms. These values are retained for context, not a speed claim.

| Configuration | Retrieval | Detect | Match | Verify |
| :--- | ---: | ---: | ---: | ---: |
| learned | 70.4 s | 52.1 s | 761.4 s | 106.2 s |
| colmap-shortlist | Not separately retained here | 309.1 s | 736.3 s | Included in matching |
| colmap-default | Exhaustive | 2292.4 s | 1671.7 s | Included in matching |

## Other recorded timings

The historical `Total` column includes mapping, in addition to the front-end stages listed. The learned `mixed3` total differs from the sum of displayed rounded stage times by 0.1 s; I retain the recorded value without manufacturing precision.

| Run | Config | Retrieval | Detect | Match | Verify | Total |
|---|---|---|---|---|---|---|
| mixed3 | learned | 108.5 s | 78.0 s | 213.4 s | 20.2 s | 963.8 s |
| mixed3 | colmap-shortlist | 72.1 s | 183.9 s | 149.9 s | (in match) | 476.9 s |
| mixed3 | colmap-default | — (exhaustive) | 1522.7 s | 692.3 s | (in match) | 2414.7 s |
| revisit | learned | 66.8 s | 44.9 s | 109.3 s | 24.8 s | 393.2 s |
| revisit | colmap-shortlist | 60.6 s | 145.1 s | 64.1 s | (in match) | 385.8 s |

## Retrieval coverage

The historical setting was `min_pairs=58`. The earlier implementation counted the query image among its nearest neighbours and omitted the final image as a query; the corrected implementation removes those mistakes. The preserved `legacy` policy is available for diagnostic comparisons. These historical pair counts must not be presented as outputs of the corrected policy.

| Run | Images | Shortlisted | All possible | Retained |
|---|---|---|---|---|
| revisit | 62 | 1860 | 1891 | **98%** |
| mixed3 | 75 | 2508 | 2775 | **90%** |

At these sizes, the shortlist retained most possible pairs. At 12,000 images, exhaustive pairing contains exactly 71,994,000 undirected pairs. That combinatorial count motivates retrieval, but does not establish measured scalability: the present implementation still constructs a dense descriptor-distance matrix. I need a separate large-scale retrieval and reconstruction benchmark.

## What remains unverified

I need retained raw artifacts, matched camera/feature settings, seed repetitions, broader sites and captures, and independent evaluation checks before making stronger claims. I keep the historical numbers unchanged while fixing definitions and code for subsequent experiments. [Audit](docs/AUDIT.md) · [Next experiments](docs/ROADMAP.md)
