# Full results

Environment: Colab T4 (15 GB), Python 3.13.15, torch 2.11.0+cu128, pycolmap 4.2.0,
kornia 0.8.3, transformers 5.16.1. One run per configuration.

Pipeline hyper-parameters are the original configuration, unchanged throughout:
ALIKED 4600 keypoints / 1024 px / threshold 0.08 · DINOv2-base MAC descriptors with
`sim_th=0.3`, `min_pairs=58`, `exhaustive_if_less=22` · LightGlue via
`kornia.feature.LightGlueMatcher('aliked')` with `min_matches=20` · mapper with
`min_model_size=3`, `max_num_models=25`.

## Metrics

**AUC@5/10/20°.** Per image pair, error = `max(rotation angular error, translation
angular error)`; AUC is the normalised area under the cumulative error curve at each
threshold — the convention used across the feature-matching literature, and the same
error definition LightGlue reports against. Unregistered pairs, and pairs split across
two reconstructions, score 180°.

**Sim(3)-aligned absolute error.** An SfM reconstruction is defined only up to a
similarity transform, so camera centres are aligned to ground truth with a Sim(3)
(Umeyama) before comparison. Reported as median camera-centre error and median absolute
rotation error, computed inside the scene's dominant reconstruction. ETH3D ground truth
is metrically scaled from laser scans, so positions are in real metres.

---

## 1. ETH3D single scenes

| Scene | Images | Registered | Clusters | AUC@5 / 10 / 20° | Median position | Median rotation |
|---|---|---|---|---|---|---|
| pipes | 14 | 14 / 14 | 1 | 0.873 / 0.937 / 0.969 | 0.007 m | 0.268° |
| terrace | 23 | 23 / 23 | 1 | 0.890 / 0.945 / 0.972 | 0.016 m | 0.422° |
| courtyard | 38 | 38 / 38 | 1 | 0.744 / 0.795 / 0.821 | 0.516 m | 1.758° |

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

Registered 75 / 75 in **2 clusters** (correct: 3). Purity **0.6933**.

| Cluster | Composition | Size |
|---|---|---|
| 0 | courtyard 38 + terrace 23 | 61 |
| 1 | pipes 14 | 14 |

| Scene | AUC@5 / 10 / 20° | Median position | Median rotation |
|---|---|---|---|
| courtyard | 0.646 / 0.700 / 0.728 | 0.809 m | 4.383° |
| pipes | 0.894 / 0.947 / 0.974 | 0.005 m | 0.252° |
| terrace | 0.897 / 0.948 / 0.974 | 0.012 m | 0.406° |

Note that `terrace` and `pipes` are barely affected — the merge damages the larger,
weaker scene while leaving its partner's numbers intact. Registration counts alone would
show nothing at all.

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

| Scene | AUC@5° | AUC@10° | AUC@20° | Median position | Median rotation |
|---|---|---|---|---|---|
| courtyard | 0.646 → **0.779** | 0.700 → 0.837 | 0.728 → 0.867 | 0.809 → **0.378 m** | 4.383 → **1.775°** |
| terrace | 0.897 → 0.893 | 0.948 → 0.947 | 0.974 → 0.973 | 0.012 → 0.015 m | 0.406 → 0.435° |
| pipes | 0.894 → 0.897 | 0.947 → 0.948 | 0.974 → 0.974 | 0.005 → 0.004 m | 0.252 → 0.319° |

The filtered `courtyard` (0.378 m) beats its own standalone run (0.516 m).

---

## 3. revisit — relief + relief_2, 62 images

`relief` and `relief_2` are the **same physical interior**, photographed in two sessions.
Correct output is **one** cluster. DINOv2 cannot separate the sets: cross-session L2
descriptor distance median 0.305, against 0.308 within `relief` and 0.295 within
`relief_2`.

Shortlist 1860 pairs (of 1891) in 61.3 s · detect 44.3 s · match 115.6 s · 905 pairs kept ·
ransac 25.6 s · mapping 131.9 s.

### Baseline

Registered 62 / 62 in **2 clusters** (correct: 1). Purity **0.5323**.

| Cluster | Composition | Size |
|---|---|---|
| 0 | relief 13 + relief_2 11 | 24 |
| 1 | relief 18 + relief_2 20 | 38 |

**The split is not by session.** Both clusters contain both sessions; cross-session
linking worked (427 pairs kept, 370 verified). The reconstruction fragmented spatially
instead.

| Scene | AUC@5 / 10 / 20° | Median position | Median rotation | In dominant cluster |
|---|---|---|---|---|
| relief | 0.423 / 0.460 / 0.478 | 0.016 m | 0.561° | 58% |
| relief_2 | 0.339 / 0.352 / 0.359 | 0.354 m | 1.248° | 65% |

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

431 of 780 geometries dropped. Mapping 131.8 s. Still 2 clusters, purity 0.5323.

| Scene | AUC@5° | Median position | Median rotation |
|---|---|---|---|
| relief | 0.423 → 0.441 | 0.016 → 0.010 m | 0.561 → 0.499° |
| relief_2 | 0.339 → **0.290** | 0.354 → 0.949 m | 1.248 → **178.537°** |

**The `relief_2` reconstruction collapsed.** After Sim(3) alignment:

| | min | p25 | median | p75 | max | cameras > 170° |
|---|---|---|---|---|---|---|
| baseline | 0.84° | 1.09° | 1.25° | 45.52° | 178.30° | **5 / 20** |
| + filter | 178.42° | 178.52° | **178.54°** | 179.85° | 179.86° | **20 / 20** |

Every camera in the filtered model points roughly backwards. The alignment is a proper
rotation (`det = +1.000`), so this is the reconstruction, not the comparison. Camera
positions still fit to within a metre, which is why the model looks plausible until
orientation is checked.

### The overlap that makes a global threshold impossible

| Link type | Should be | n | p10 | median | max |
|---|---|---|---|---|---|
| within-scene | kept | 890 | 32 | 302 | 3379 |
| cross-session (revisit) | **kept** | 370 | 18 | **51** | 860 |
| cross-scene | **dropped** | 150 | 16 | **22** | 83 |

A threshold removing all 150 false cross-scene links (84) removes 80% of the 370 true
cross-session links.

---

## 4. Mill 19 — real UAV imagery

4608 × 3456 frames from two Pittsburgh industrial sites. No ground-truth poses were
obtainable (the pose metadata sits at the tail of an 11 GB archive the host stopped
serving mid-study), so these runs report registration and clustering only.

### 4a. Validation split — unusable, and it looks like a result

20 `building` frames sampled roughly every 97th frame of a ~1900-image flight.

Shortlist 190 (exhaustive) · detect 11.4 s · match 24.4 s · **67 pairs kept (35%)** ·
ransac 1.4 s · mapping 62.8 s · registered **13 / 20** in **3 clusters** (5 / 3 / 5).

The fragmentation is missing overlap, not scene confusion. An earlier check on this split
did confirm the pose-handling code: within its largest reconstruction, rotation error
against the PixSfM reference had a median of 0.47° and translation direction 0.74°.

### 4b. Single site, consecutive frames

120 consecutive `building` frames streamed from the 11 GB archive.

Shortlist 6283 (of 7140) in 68.8 s · detect 51.1 s · match 768.3 s · **3538 pairs kept
(56%)** · ransac 159.5 s · mapping 2263.5 s · registered **120 / 120** in **1 cluster**.

### 4c. Mixed sites — clean separation, no filter

251 frames (`building` 120 + `rubble` 131).

Shortlist **16580 of 31375 (53%)** in 143.6 s · detect 108.8 s · match 2031.5 s ·
**7742 pairs kept** · ransac 265.7 s.

Registered **234 / 251** in **2 clusters**, purity **1.0000**:

| Cluster | Composition |
|---|---|
| 0 | rubble 131 / 131 |
| 1 | building 103 / 120 |

LightGlue pairs kept, by site pair:

| | Pair | Count | Share |
|---|---|---|---|
| WITHIN | building \| building | 3530 | 45.6% |
| WITHIN | rubble \| rubble | 3305 | 42.7% |
| CROSS | building \| rubble | 907 | 11.7% |

After geometric verification:

| | n | min | p10 | median | p90 | max | ≥100 inliers |
|---|---|---|---|---|---|---|---|
| within-site | 5623 | 15 | 26 | 412 | 2025 | 3706 | many |
| **cross-site** | **195** | 15 | 15 | **16** | 19 | **37** | **0** |

Cross-site links are 3.4% of verified geometries and sit at the floor RANSAC accepts.
Compare `courtyard`↔`terrace`, whose false links reached 83 inliers and merged two
scenes. The merge failure requires two distinct places that genuinely look alike; when
they do not, geometric verification removes the cross-links unaided.

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
