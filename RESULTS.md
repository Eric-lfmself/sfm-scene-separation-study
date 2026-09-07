# Full results

Environment: Colab T4 (15 GB), Python 3.13.15, torch 2.11.0+cu128, pycolmap 4.2.0,
kornia 0.8.3, transformers 5.16.1. One run per configuration.

Pipeline hyper-parameters are the original configuration, unchanged throughout:
ALIKED 4600 keypoints / 1024 px / threshold 0.08 · DINOv2-base MAC descriptors with
`sim_th=0.3`, `min_pairs=58`, `exhaustive_if_less=22` · LightGlue via
`kornia.feature.LightGlueMatcher('aliked')` with `min_matches=20` · mapper with
`min_model_size=3`, `max_num_models=25`.

**Metric.** mAA over relative poses. For each ground-truth image pair,
`err = max(rotation angle error, translation direction angle error)`; mAA is the mean of
the fractions under 1°, 2°, 5° and 10°. Unregistered pairs, and pairs split across
different clusters, score 180°.

---

## 1. ETH3D single scenes

| Scene | Images | Registered | Clusters | Shortlisted | Kept | mAA | 1° / 2° / 5° / 10° |
|---|---|---|---|---|---|---|---|
| pipes | 14 | 14 / 14 | 1 | 91 | 71 | 0.9505 | .824 .989 .989 1.000 |
| terrace | 23 | 23 / 23 | 1 | 253 | 210 | 0.9684 | .889 .984 1.000 1.000 |
| courtyard | 38 | 38 / 38 | 1 | 703 | 670 | 0.7966 | .697 .802 .841 .846 |

Timings (seconds):

| Scene | shortlist | detect | match | ransac | mapping |
|---|---|---|---|---|---|
| pipes | 0.0 | 13.5 | 3.6 | 0.8 | 10.0 |
| terrace | 30.7 | 16.2 | 22.1 | 3.6 | 43.8 |
| courtyard | 36.7 | 27.5 | 77.7 | 9.4 | 129.7 |

`pipes` has 14 images, below `exhaustive_if_less=22`, so retrieval is skipped entirely.

---

## 2. mixed3 — courtyard + terrace + pipes, 75 images

Shortlist 2508 pairs (of 2775 possible) in 92.5 s · detect 56.9 s · match 221.7 s ·
1268 pairs kept · ransac 21.6 s · mapping 448.6 s.

### Baseline

Registered 75 / 75 in **2 clusters** (correct: 3).

| Cluster | Composition | Size |
|---|---|---|
| 0 | courtyard 38 + terrace 23 | 61 |
| 1 | pipes 14 | 14 |

Purity **0.6933**. Per-scene mAA: courtyard 0.6945, pipes 0.9615, terrace 0.9713.

### Where the merge comes from

LightGlue pairs kept (≥20 matches), by scene pair:

| | Pair | Count |
|---|---|---|
| WITHIN | courtyard \| courtyard | 661 |
| **CROSS** | **courtyard \| terrace** | **296** |
| WITHIN | terrace \| terrace | 210 |
| WITHIN | pipes \| pipes | 71 |
| CROSS | courtyard \| pipes | 19 |
| CROSS | pipes \| terrace | 11 |

After COLMAP geometric verification (1040 non-empty two-view geometries):

| | Pair | Count | Share |
|---|---|---|---|
| WITHIN | courtyard \| courtyard | 643 | 61.8% |
| WITHIN | terrace \| terrace | 183 | 17.6% |
| **CROSS** | **courtyard \| terrace** | **147** | **14.1%** |
| WITHIN | pipes \| pipes | 64 | 6.2% |
| CROSS | courtyard \| pipes | 3 | 0.3% |

The cross-scene links concentrate on a handful of images — `courtyard/DSC_0302–0305`
against `terrace/DSC_0259, 0279, 0285` — with 64–83 inliers each.

Inlier distributions:

| | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|
| within-scene | 890 | 15 | 32 | 302 | 1421 | 3379 |
| cross-scene | 150 | 15 | 16 | 22 | 54 | 83 |

Threshold sweep:

| Threshold | Within kept | Cross kept | Contamination |
|---|---|---|---|
| 20 | 860 | 94 | 9.9% |
| 40 | 777 | 31 | 3.8% |
| 60 | 718 | 10 | 1.4% |
| **84** | **676** | **0** | **0.0%** |
| 100 | 652 | 0 | 0.0% |
| 200 | 532 | 0 | 0.0% |

### Ablation: drop verified geometries below 100 inliers

388 of 1040 geometries dropped. Mapping 172.0 s (from 448.6 s).

Registered 75 / 75 in **3 clusters**, one per scene. Purity **1.0000**.

| Scene | baseline mAA | filtered mAA |
|---|---|---|
| courtyard | 0.6945 | **0.8332** |
| terrace | 0.9713 | 0.9783 |
| pipes | 0.9615 | 0.9698 |

Note the filtered `courtyard` (0.833) beats its own standalone run (0.797).

---

## 3. revisit — relief + relief_2, 62 images

`relief` and `relief_2` are the **same physical interior**, photographed in two sessions.
Correct output is **one** cluster. DINOv2 cannot separate the sets: cross-session L2
descriptor distance median 0.305, against 0.308 within `relief` and 0.295 within
`relief_2`.

Shortlist 1860 pairs (of 1891) in 61.3 s · detect 44.3 s · match 115.6 s · 905 pairs kept ·
ransac 25.6 s · mapping 131.9 s.

### Baseline

Registered 62 / 62 in **2 clusters** (correct: 1).

| Cluster | Composition | Size |
|---|---|---|
| 0 | relief 13 + relief_2 11 | 24 |
| 1 | relief 18 + relief_2 20 | 38 |

**The split is not by session.** Both clusters contain both sessions; cross-session linking
worked (427 pairs kept, 370 verified). The reconstruction fragmented spatially instead.

Within-session mAA: relief 0.4581, relief_2 0.3629.

### The link populations

| | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|
| same-session | 410 | — | — | 460 | — | — |
| **cross-session (true)** | **370** | 15 | 18 | **51** | 225 | 860 |

Survival under a threshold:

| Threshold | Cross-session links surviving |
|---|---|
| 20 | 321 / 370 |
| 50 | 190 / 370 |
| **100** | **73 / 370** |
| 150 | 55 / 370 |
| 200 | 46 / 370 |

### Ablation: the same 100-inlier filter

431 of 780 geometries dropped. Mapping 131.8 s.

Registered 62 / 62 in **3 clusters** — worse than the baseline's 2, against a correct
answer of 1.

| Scene | baseline mAA | filtered mAA | |
|---|---|---|---|
| relief | 0.4581 | 0.4747 | better |
| relief_2 | 0.3629 | **0.3097** | **worse** |

### The overlap

| Link type | Should be | n | p10 | median | max |
|---|---|---|---|---|---|
| within-scene | kept | 890 | 32 | 302 | 3379 |
| cross-session (revisit) | **kept** | 370 | 18 | **51** | 860 |
| cross-scene | **dropped** | 150 | 16 | **22** | 83 |

A threshold that removes all 150 false cross-scene links (84) removes 80% of the 370 true
cross-session links.

---

## 4. Mill 19 — real UAV imagery

4608 × 3456 frames from two Pittsburgh industrial sites. Poses are a PixSfM reconstruction,
not independent survey ground truth.

### 4a. Validation split — unusable, and it looks like a result

20 `building` frames sampled roughly every 97th frame of a ~1900-image flight.

Shortlist 190 (exhaustive) · detect 11.4 s · match 24.4 s · **67 pairs kept (35%)** ·
ransac 1.4 s · mapping 62.8 s · registered **13 / 20** in **3 clusters** (5 / 3 / 5).

mAA 0.0842 — but that is fragmentation, not pose error. Within the largest reconstruction,
rotation error median **0.47°** and translation direction **0.74°** against the PixSfM
reference, which validates the coordinate conversion
(`R_cw = diag(1,-1,-1) @ R_wc.T`, `t_cw = -R_cw @ t_wc`).

### 4b. Single site, consecutive frames

120 consecutive `building` frames streamed from the 11 GB archive.

Shortlist 6283 (of 7140) in 68.8 s · detect 51.1 s · match 768.3 s · **3538 pairs kept
(56%)** · ransac 159.5 s · mapping 2263.5 s · registered **120 / 120** in **1 cluster**.

### 4c. Mixed sites — incomplete

251 frames (`building` 120 + `rubble` 131).

Shortlist **16580 of 31375 (53%)** in 143.6 s · detect 108.8 s · match 2031.5 s ·
**7742 pairs kept** · ransac 265.7 s.

The incremental mapper did not return; the Colab session was lost during it. With
`max_num_models=25` on a 251-image two-site set, the mapper explores far more sub-models
than the correct answer (2) requires. Reported as incomplete.

---

## 5. Retrieval retention across all runs

| Run | Images | Shortlisted | All possible | Retained | DINOv2 cost |
|---|---|---|---|---|---|
| revisit | 62 | 1860 | 1891 | 98% | 61 s |
| mixed3 | 75 | 2508 | 2775 | 90% | 93 s |
| uav building | 120 | 6283 | 7140 | 88% | 69 s |
| uav mixed | 251 | 16580 | 31375 | 53% | 144 s |

`min_pairs=58` — not `sim_th=0.3` — is what sets the shortlist at every size below ~250
images.
