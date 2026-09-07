# Two silent failure modes in an SfM matching pipeline

A study of the retrieval-and-matching stage of a Structure-from-Motion pipeline
(**DINOv2 shortlisting → ALIKED → LightGlue → COLMAP**), ported forward to current
library versions and evaluated against ground truth on public data.

The pipeline reconstructs single scenes to millimetres. It then does two things
silently wrong, and the obvious fix for the first one makes the second one worse.

---

## Metrics

Two standard views of pose accuracy, both reported throughout:

**AUC@5/10/20°** — the convention across the feature-matching literature. Per image
pair, error = `max(rotation angular error, translation angular error)`; AUC is the
normalised area under the cumulative error curve at each threshold. This is the same
error definition LightGlue reports AUC@5/10/20° against, so the numbers sit on the same
scale as that literature. Pairs that are unregistered — or that land in two different
reconstructions, where relative pose is meaningless — score 180°.

**Sim(3)-aligned absolute error** — the Structure-from-Motion convention. An SfM
reconstruction is defined only up to a similarity transform, so the reconstruction is
first aligned to ground truth with a Sim(3); the table then reports median camera-centre
error and median absolute rotation error. ETH3D ground truth is metrically scaled from
laser scans, so these are **real metres**.

---

## Results

All runs on one Colab T4.

### Single scenes (ETH3D, laser-scanned ground truth)

| Scene | Images | Registered | Clusters | AUC@5 / 10 / 20° | Median position | Median rotation |
|---|---|---|---|---|---|---|
| `pipes` | 14 | **14 / 14** | 1 | 0.873 / 0.937 / 0.969 | **7 mm** | 0.27° |
| `terrace` | 23 | **23 / 23** | 1 | 0.890 / 0.945 / 0.972 | **16 mm** | 0.42° |
| `courtyard` | 38 | **38 / 38** | 1 | 0.744 / 0.795 / 0.821 | **516 mm** | 1.76° |

The port is correct: on two of three scenes the reconstruction lands within centimetres
of a laser scan. `courtyard` is 30× worse than its neighbours on the same rig and
settings — half a metre — which is the first hint that something in this pipeline fails
without announcing itself.

### Failure 1 — two different scenes get welded together

75 images from `courtyard` + `terrace` + `pipes` shuffled into one folder. This is the
setting the study is about: unrelated scenes arrive in one bucket and the pipeline has
to separate them.

| | Registered | Clusters | Purity | `courtyard` AUC@5° | `courtyard` position |
|---|---|---|---|---|---|
| baseline | 75 / 75 | **2** (should be 3) | **0.693** | 0.646 | 809 mm |
| + 100-inlier filter | 75 / 75 | 3 | **1.000** | **0.779** | **378 mm** |

`courtyard` and `terrace` merged into one reconstruction. Both sit on the ETH Zürich
campus and share facade, railing and paving texture, so ALIKED and LightGlue find
*real*, locally-consistent correspondences between them: **296** cross-scene pairs
cleared the match threshold and **147** survived COLMAP's geometric verification.

Dropping two-view geometries below 100 RANSAC inliers before the mapper separates the
scenes perfectly, and improves accuracy on top: `courtyard` goes from 809 mm to 378 mm —
better than its own 516 mm standalone run — while mapping time falls 449 s → **172 s**.

### Failure 2 — the fix, falsified

A weak link has two possible causes and the filter cannot tell them apart: a **false**
link between different places that look alike, or a **true** link between two captures
of the same place. ETH3D's `relief` and `relief_2` are the second case — 31 + 31 images
of the *same* interior, photographed twice. DINOv2 cannot separate them at all
(cross-session descriptor distance 0.305, against 0.308 and 0.295 within), because there
is nothing to separate. Correct output is **one** cluster.

| | Clusters | `relief` AUC@5° | `relief_2` AUC@5° | `relief_2` rotation |
|---|---|---|---|---|
| baseline | **2** (should be 1) | 0.423 | 0.339 | 1.25° |
| + 100-inlier filter | **2** | 0.441 | **0.290** | **178.5°** |

The filter deleted 431 of 780 verified geometries, and the `relief_2` reconstruction
collapsed: after Sim(3) alignment, **all 20 of its cameras are more than 178° from their
true orientation.** The camera *positions* still fit (0.95 m median), so the model looks
plausible until I checked where the cameras were pointing. The baseline was not perfect
either — 5 of 20 cameras flipped — but 15 were within 5°. Starving the mapper of
constraints turned a mostly-correct reconstruction into a uniformly inverted one.

It had to delete them, because true cross-session links are weak:

| Link type | Should be | n | p10 | median | max |
|---|---|---|---|---|---|
| within-scene | kept | 890 | 32 | **302** | 3379 |
| **cross-session (revisit)** | **kept** | 370 | 18 | **51** | 860 |
| cross-scene (`courtyard`↔`terrace`) | **dropped** | 150 | 16 | **22** | 83 |

**Inlier count alone cannot distinguish "a different place that looks similar" from
"the same place, seen again."** A threshold high enough to remove all 150 false links
(84) also removes **80% of the 370 true ones**.

### The failure needs visual similarity, not just mixing

Two real UAV sites — an industrial building and a rubble field, 251 frames — mixed into
one folder, and the pipeline separated them **perfectly with no filter at all**:

| | Registered | Clusters | Purity |
|---|---|---|---|
| `building` + `rubble` | 234 / 251 | **2** | **1.000** |

LightGlue still produced 907 cross-site pairs (11.7% of everything it kept). Geometric
verification then destroyed them:

| | n | p10 | median | max | ≥100 inliers |
|---|---|---|---|---|---|
| within-site | 5623 | 26 | **412** | 3706 | many |
| cross-site | 195 | 15 | **16** | **37** | **0** |

A median of 16 inliers sits at the floor RANSAC will accept — that is noise, not
structure. Compare `courtyard`↔`terrace`, whose false links reached **83**: those were
genuine correspondences on genuinely similar architecture. So the merge failure is not
"mixed input breaks the pipeline"; it needs two distinct places that actually look
alike, and geometric verification handles the rest on its own.

### Aerial runs

| Run | Frames | Pairs matched | Registered | Clusters | Mapping |
|---|---|---|---|---|---|
| val split | 20 | 67 / 190 · 35% | 13 / 20 | 3 | 63 s |
| `building` | 120 | 3538 / 6283 · 56% | **120 / 120** | **1** | 2264 s |
| `building` + `rubble` | 251 | 7742 / 16580 · 47% | **234 / 251** | **2** | — |

**The val-split row is the cautionary one.** It reads like a clustering failure and is
not one: that split samples roughly every 97th frame of a ~1900-image flight, so
consecutive images barely overlap. The fragmentation is missing overlap. A benchmark
split built for one task (novel-view synthesis) can be actively wrong for another, and
the symptom is indistinguishable from the failure I was looking for. Streaming
*consecutive* frames out of the 11 GB archives fixed it.

No ground-truth poses were obtainable for the aerial frames (the pose metadata sits at
the tail of an 11 GB archive the host stopped serving), so the aerial rows report
registration and clustering only — both metric-independent.

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
RESULTS.md               every number, every timing, and the two figures
docs/porting-notes.md    the four API breakages, with before/after
figures/                 the figures as static SVG, light and dark
tools/make_figures.py    regenerates them
```

## Running it

```bash
pip install pycolmap kornia kornia_moons h5py transformers opencv-python
pip install git+https://github.com/cvg/LightGlue.git

python src/run_experiments.py --fetch eth3d --scenes courtyard terrace pipes relief relief_2
python src/run_experiments.py --experiment mixed3
python src/run_experiments.py --experiment mixed3 --min-inliers 100
python src/run_experiments.py --experiment revisit
python src/run_experiments.py --experiment revisit --min-inliers 100
```

Needs a CUDA GPU. Verified on a Colab T4 (15 GB) with Python 3.13.15, torch 2.11.0+cu128,
pycolmap 4.2.0, kornia 0.8.3, transformers 5.16.1.

Set `--max-num-models` well below the default 25 for mixed-scene runs — on 251 images the
mapper spent over an hour exploring sub-models when the correct answer was 2.

## Limitations

- Five ETH3D scenes and two UAV sites; at most 251 images per run. Each failure was
  observed on one specific scene pair, and neither is established as general.
- One run per configuration. COLMAP's incremental mapper is seed-dependent, so small
  AUC differences are not necessarily real. The large ones (a 178° rotation collapse, a
  purity change from 0.693 to 1.000) are well outside that noise.
- The aerial runs have no ground-truth poses, so they contribute clustering evidence
  only.
- `relief` and `relief_2` are registered in independent coordinate frames, so only
  within-session pose error is meaningful there; the cross-session question is answered
  by cluster count, not by pose error.

## Data and credits

- **ETH3D** high-resolution multi-view — <https://www.eth3d.net/datasets>
- **Mill 19** (Mega-NeRF) — <https://meganerf.cmusatyalab.org>
- **LightGlue / ALIKED** — <https://github.com/cvg/LightGlue>
- **DINOv2** — `facebook/dinov2-base`
- **COLMAP / pycolmap** — <https://colmap.github.io>
