# Two silent failure modes in an SfM matching pipeline

A study of the retrieval-and-matching stage of a Structure-from-Motion pipeline
(**DINOv2 shortlisting → ALIKED → LightGlue → COLMAP**), ported forward to current
library versions and evaluated against ground truth on public data.

The pipeline reconstructs single scenes near-perfectly. It then does two things
silently wrong, and the obvious fix for the first one makes the second one worse.

---

## Results

All runs on one Colab T4. Metric is **mAA over relative poses**, which is invariant to
the arbitrary gauge of an SfM reconstruction: per image pair,
error = `max(rotation angle error, translation direction error)`, averaged over the
1°/2°/5°/10° thresholds. Pairs that are unregistered — or that land in different
clusters — score 180° and count as failures.

### Single scenes (ETH3D, laser-scanned ground truth)

| Scene | Images | Registered | Clusters | mAA | 1° / 2° / 5° / 10° |
|---|---|---|---|---|---|
| `pipes` | 14 | **14 / 14** | 1 | 0.951 | .824 .989 .989 1.000 |
| `terrace` | 23 | **23 / 23** | 1 | 0.968 | .889 .984 1.000 1.000 |
| `courtyard` | 38 | **38 / 38** | 1 | 0.797 | .697 .802 .841 .846 |

The port is correct. Note `courtyard`: mAA stops climbing at 0.846 even by the 10°
threshold, so ~15% of its pairs are not marginally off — they are wrong.

### Failure 1 — two different scenes get welded together

75 images from `courtyard` + `terrace` + `pipes` shuffled into one folder. This is the
setting the study is about: unrelated scenes arrive in one bucket and the pipeline has
to separate them.

| | Registered | Clusters | Purity |
|---|---|---|---|
| baseline | 75 / 75 | **2** (should be 3) | **0.693** |
| + 100-inlier filter | 75 / 75 | 3 | **1.000** |

`courtyard` and `terrace` merged into one reconstruction. Both sit on the ETH Zürich
campus and share facade, railing and paving texture, so ALIKED and LightGlue find
*real*, locally-consistent correspondences between them: **296** cross-scene pairs
cleared the match threshold and **147** survived COLMAP's geometric verification.

Dropping two-view geometries below 100 RANSAC inliers before the mapper fixes it,
and improves everything else too — `courtyard` mAA 0.695 → **0.833** (beating its own
0.797 standalone run), `terrace` 0.971 → 0.978, `pipes` 0.962 → 0.970, mapping time
449 s → **172 s**.

### Failure 2 — the fix, falsified

A weak link has two possible causes and the filter cannot tell them apart: a **false**
link between different places that look alike, or a **true** link between two captures
of the same place. ETH3D's `relief` and `relief_2` are the second case — 31 + 31 images
of the *same* interior, photographed twice. DINOv2 cannot separate them at all
(cross-session descriptor distance 0.305 vs 0.308 / 0.295 within), because there is
nothing to separate. Correct output is **one** cluster.

| | Registered | Clusters | `relief` mAA | `relief_2` mAA |
|---|---|---|---|---|
| baseline | 62 / 62 | **2** (should be 1) | 0.458 | 0.363 |
| + 100-inlier filter | 62 / 62 | **3** | 0.475 | **0.310** |

The filter deleted 431 of 780 verified geometries. It had to — true cross-session
links are weak:

| Link type | Should be | n | p10 | median | max |
|---|---|---|---|---|---|
| within-scene | kept | 890 | 32 | **302** | 3379 |
| **cross-session (revisit)** | **kept** | 370 | 18 | **51** | 860 |
| cross-scene | **dropped** | 150 | 16 | **22** | 83 |

**Inlier count alone cannot distinguish "a different place that looks similar" from
"the same place, seen again."** A threshold high enough to remove all 150 false links
(84) also removes **80% of the 370 true ones**. On a survey where flight strips overlap
and lighting changes between passes, that is the difference between one reconstruction
and several.

### Aerial transfer (Mill 19, real UAV imagery)

| Run | Frames | Pairs matched | Registered | Clusters | Mapping |
|---|---|---|---|---|---|
| val split | 20 | 67 / 190 · 35% | 13 / 20 | 3 | 63 s |
| `building` | 120 | 3538 / 6283 · 56% | **120 / 120** | **1** | 2264 s |
| `building` + `rubble` | 251 | 7742 / 16580 · 47% | *mapping did not finish* | | |

**The val-split row is the cautionary one.** It reads like a clustering failure and is
not one: that split samples roughly every 97th frame of a ~1900-image flight, so
consecutive images barely overlap. The fragmentation is missing overlap. A benchmark
split built for one task (novel-view synthesis) can be actively wrong for another, and
the symptom is indistinguishable from the failure you are hunting.

Streaming *consecutive* frames out of the 11 GB archives fixed it, and the single-site
run is the clean aerial transfer result. Pose handling was validated where overlap does
exist: within one reconstruction, rotation error against the PixSfM reference has a
median of **0.47°** and translation direction **0.74°**.

### The retrieval parameter that only starts working at ~250 images

`min_pairs=58` forces each image to keep its 58 nearest DINOv2 neighbours. Across every
run, that floor — not the `sim_th` similarity test — decides the shortlist:

| Run | Images | Shortlisted | All possible | Retained | DINOv2 cost |
|---|---|---|---|---|---|
| revisit | 62 | 1860 | 1891 | **98%** | 61 s |
| mixed3 | 75 | 2508 | 2775 | **90%** | 93 s |
| uav building | 120 | 6283 | 7140 | **88%** | 69 s |
| uav mixed | 251 | 16580 | 31375 | **53%** | 144 s |

At 62 images the retrieval stage spends a minute of GPU time to discard 31 pairs out of
1891. The default was tuned for collections of roughly 200 images; anything smaller runs
near-exhaustive matching while believing it is running retrieval.

---

## What actually broke in the port

The pipeline was written against Python 3.10 with pinned wheels that no longer resolve on
3.13. Current PyPI versions install cleanly, but four API changes stop the code. Details
and fixes in [`docs/porting-notes.md`](docs/porting-notes.md).

1. **COLMAP grew a rig/frame schema.** pycolmap 4.x requires every image to belong to a
   frame and every frame to a rig, so the classic `h5_to_db.py` / `database.py` SQLite
   writers produce an obsolete schema. Replaced with one trivial rig + frame per image.
2. **`pycolmap.Database()` has no constructor** — use the factory `Database.open(path)`.
3. **`Image.cam_from_world` is now a method**, derived from the image's frame rather than
   stored. Reading it as an attribute yields a bound method and fails downstream.
4. **`kornia.io.load_image` is broken** against current `kornia_rs` (it calls
   `read_image_jpegturbo`, renamed to `read_image_jpeg`). Swapped for a cv2 loader with an
   identical output contract.

---

## Layout

```
src/sfm_pipeline.py      the ported pipeline: retrieval, detection, matching,
                         COLMAP ingestion (pycolmap 4.x), ground truth, metrics
src/run_experiments.py   dataset builders, the experiment driver, link diagnostics,
                         and the inlier-filter ablation
report/index.html        the written study, with the two figures
RESULTS.md               every number, including timings
docs/porting-notes.md    the four API breakages, with before/after
notebooks/               the original working log (see notebooks/README.md)
```

## Running it

```bash
pip install pycolmap kornia kornia_moons h5py transformers opencv-python
pip install git+https://github.com/cvg/LightGlue.git

python src/run_experiments.py --fetch eth3d --scenes courtyard terrace pipes
python src/run_experiments.py --experiment mixed3
python src/run_experiments.py --experiment mixed3 --min-inliers 100
python src/run_experiments.py --experiment revisit
python src/run_experiments.py --experiment revisit --min-inliers 100
```

Needs a CUDA GPU. Verified on a Colab T4 (15 GB) with Python 3.13.15, torch 2.11.0+cu128,
pycolmap 4.2.0, kornia 0.8.3, transformers 5.16.1. Total compute for the study was about
four hours.

Set `--max-num-models` well below the default 25 for mixed-scene runs — on 251 images the
mapper spent over an hour exploring sub-models when the correct answer was 2.

## Limitations

- Scoring is the relative-pose mAA defined above. Other pose benchmarks define mAA with
  different thresholds and different handling of clustering, so numbers here are not
  directly comparable to figures reported elsewhere.
- Five ETH3D scenes and at most 251 images per run. Each failure was observed on one
  specific scene pair; neither is established as general.
- One run per configuration. COLMAP's incremental mapper is seed-dependent; mAA
  differences under ~0.02 are not necessarily real.
- Mill 19 poses are a PixSfM reconstruction, not independent survey ground truth, so
  aerial pose agreement measures consistency with a careful reference rather than
  absolute accuracy.
- The 251-frame mixed UAV run did not finish mapping and is reported as incomplete.

## Data and credits

- **ETH3D** high-resolution multi-view — <https://www.eth3d.net/datasets>
- **Mill 19** (Mega-NeRF) — <https://meganerf.cmusatyalab.org>
- **LightGlue / ALIKED** — <https://github.com/cvg/LightGlue>
- **DINOv2** — `facebook/dinov2-base`
- **COLMAP / pycolmap** — <https://colmap.github.io>
