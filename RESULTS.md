# Results

Every number below comes from one Colab session on a single Tesla T4, with
Python 3.13.15, torch 2.11.0+cu128, pycolmap 4.2.0, kornia 0.8.3, transformers 5.16.1.
Raw logs are one per run; each table cell traces to a line in one of them.

## The three configurations

The point of the study is a controlled comparison, so everything downstream of the
feature database is the *same code path* for all three: the same
`pycolmap.incremental_mapping` call, the same options, the same metrics.

| | pair selection | detector | matcher | budget |
|---|---|---|---|---|
| `learned` | DINOv2 shortlist | ALIKED | LightGlue | 4600 features, 1024 px |
| `colmap-shortlist` | DINOv2 shortlist | SIFT | nearest neighbour + ratio test | 4600 features, 1024 px |
| `colmap-default` | exhaustive | SIFT | nearest neighbour + ratio test | COLMAP stock: 8192 features, 3200 px |

`learned` and `colmap-shortlist` see **the identical set of image pairs** at **the
identical feature budget and input resolution**. The only difference between them is the
detector and the matcher. `colmap-default` is COLMAP as shipped, and is included because
that is what a pipeline like this is usually being compared against.

## A timing caveat that has to come first

`pycolmap.has_cuda` is `False` on every published wheel. SIFT detection and
nearest-neighbour matching therefore run on the CPU, while ALIKED and LightGlue go
through torch and run on the GPU. **Front-end timings across configurations compare a
GPU against a CPU and are not meaningful.** They are listed for completeness and marked.

`incremental_mapping` is the exception: COLMAP maps on the CPU in every configuration,
through the identical call. Mapping time is comparable and is the only timing used for
any claim here.

## Metrics

**AUC@5/10/20°** — per image pair, error = `max(rotation angular error, translation
angular error)`; AUC is the normalised area under the cumulative error curve. Pairs that
are unregistered, or that land in two different reconstructions, score 180°.

**Sim(3)-aligned absolute error** — the reconstruction is aligned to ground truth with a
similarity transform, then median camera-centre error and median absolute rotation error
are reported. ETH3D ground truth is metrically scaled from laser scans, so the position
column is real millimetres.

**Inlier ratio** — pooled over every matched pair: RANSAC inliers / putative matches.
This is what "matching precision" means in the feature-matching literature.

**Pair precision** — of the image pairs that passed geometric verification, the fraction
that are genuinely the same scene. Note the sign flips for the revisit experiment, where
cross-session pairs are *true* links; that table is annotated accordingly.

---

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

Two things in this table are worth stating plainly.

**`courtyard` is where the learned front end falls apart, and it is not a fluke of one
run.** Across three independent runs of the same `learned` configuration I measured
516 mm, 849 mm and 1111 mm. The two SIFT configurations landed on 29 mm and 31 mm, and
`colmap-shortlist` reproduced 29-30 mm in both the standalone and the mixed run. The
learned front end is not merely less accurate on this scene, it is unstable run to run.

**`pipes` is where SIFT falls apart, and that one *is* an artefact — of resolution, not
of SIFT.** At 1024 px SIFT registers 8 of 14 images. At full resolution it registers all
14 with the best pose accuracy of any configuration in the table. Any conclusion of the
form "the learned matcher wins on low-texture scenes" that is drawn from the
`colmap-shortlist` row alone would be wrong; the `colmap-default` row is the control that
catches it.

Feature counts actually extracted, for reference: `colmap-shortlist` 1822-5542 keypoints
per image, `colmap-default` 11393-12511.

---

## Mixed scenes: `courtyard` + `terrace` + `pipes`

75 images shuffled into one folder. Correct output is **3** reconstructions.

| | Registered | Clusters | Purity | Inlier ratio | Pair precision | Mapping |
|---|---|---|---|---|---|---|
| learned | 75 / 75 | **2** | **0.6933** | 0.8396 | **0.8556** (889 / 1039) | 543.8 s |
| colmap-shortlist | 69 / 75 | **3** ✓ | **1.0000** | 0.8532 | **1.0000** (559 / 559) | 71.0 s |
| colmap-default | 75 / 75 | **3** ✓ | **1.0000** | 0.8946 | **1.0000** (614 / 614) | 199.7 s |

Per-scene accuracy inside the mixed run:

| Scene | learned | colmap-shortlist | colmap-default |
|---|---|---|---|
| `courtyard` | 0.678 / 849 mm | 0.906 / 30 mm | 0.898 / 34 mm |
| `terrace` | 0.101 / 2246 mm, **8 cameras >170° off** | 0.928 / 8 mm | 0.933 / 7 mm |
| `pipes` | 0.861 / 8 mm | 0.209 / 32 mm (57% registered) | **0.927 / 5 mm** |

`courtyard` and `terrace` merged into a single reconstruction under `learned`, and
`terrace` was wrecked in the process: 8 of its 23 cameras ended up pointing more than
170° from their true orientation, while the camera positions still looked plausible.

**The merge is a property of the learned matcher, not of feature density.** This is the
claim `colmap-default` exists to test. It extracts 11928 keypoints per image -- 2.6x the
budget the learned configuration gets, at three times the resolution -- and still
produced **zero** cross-scene verified pairs. More features did not manufacture false
links. LightGlue did: 296 cross-scene pairs cleared its match threshold and 150 survived
COLMAP's geometric verification.

Where the learned front end's false links sit, against the true ones:

| Link type | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|
| within-scene | 889 | 15 | 33 | 308 | 1419 | 3376 |
| cross-scene (false) | 150 | 15 | 16 | 22 | 54 | 85 |

Cross-scene pairs kept by LightGlue, by scene pair: `courtyard`|`terrace` **296**,
`courtyard`|`pipes` 19, `pipes`|`terrace` 11. The confusion is overwhelmingly between the
two scenes that actually look alike -- both sit on the ETH Zürich campus and share
facade, railing and paving texture.

---

## Revisit: `relief` + `relief_2`

31 + 31 images of the *same* interior, photographed in two sessions. Correct output is
**1** reconstruction. Ground truth for the two sessions lives in two independent
coordinate frames, so only within-session pose error is meaningful.

**The sign of `pair precision` is inverted here.** A cross-session pair is a *true* link,
so a lower "precision" in this table is not a defect -- it is the metric counting correct
behaviour as error. The column that matters is cluster count.

| | Registered | Clusters | Inlier ratio | Mapping |
|---|---|---|---|---|
| learned | 62 / 62 | **2** | 0.7154 | 147.5 s |
| colmap-shortlist | 62 / 62 | **1** ✓ | **0.9053** | 116.0 s |

| Scene | learned | colmap-shortlist |
|---|---|---|
| `relief` | 0.449, 58% in dominant reconstruction | **0.674, 100%**, 132 mm, 1.419° |
| `relief_2` | 0.289, **179.037°**, **20 of 20 cameras >170° off** | **0.899, 8 mm, 0.337°** |

The learned front end split one place into two reconstructions and then inverted one of
them: after Sim(3) alignment every camera in the `relief_2` model points backwards, while
the positions still fit to about a metre. SIFT fused the two sessions into one model, as
it should, and reconstructed `relief_2` to 8 mm.

Its cross-session links were weak but numerous under `learned` -- n=370, median 51,
against within-session n=412, median 452 -- and evidently not strong enough for the
mapper to fuse the sessions.

---

## What the two mixed experiments say together

| Situation | Correct answer | learned | SIFT + NN |
|---|---|---|---|
| three different scenes in one folder | 3 clusters | **2** — welds two different places together | 3 ✓ |
| one place photographed twice | 1 cluster | **2** — splits one place in half | 1 ✓ |

The learned front end is wrong in both directions on this data: it merges what should be
separate and separates what should be merged. SIFT with nearest-neighbour matching gets
both right, at 1/2.6 of the feature budget and 1/6 of the input resolution.

---

## Front-end timings (not comparable — CPU vs GPU)

Listed only so the logs can be traced. See the caveat above.

| Run | Config | Retrieval | Detect | Match | Verify | Total |
|---|---|---|---|---|---|---|
| mixed3 | learned | 108.5 s | 78.0 s | 213.4 s | 20.2 s | 963.8 s |
| mixed3 | colmap-shortlist | 72.1 s | 183.9 s | 149.9 s | (in match) | 476.9 s |
| mixed3 | colmap-default | — (exhaustive) | 1522.7 s | 692.3 s | (in match) | 2414.7 s |
| revisit | learned | 66.8 s | 44.9 s | 109.3 s | 24.8 s | 393.2 s |
| revisit | colmap-shortlist | 60.6 s | 145.1 s | 64.1 s | (in match) | 385.8 s |

## Retrieval: the parameter that only starts working at scale

`min_pairs=58` forces every image to keep its 58 nearest DINOv2 neighbours. On these
collection sizes that floor, not the `sim_th` similarity test, decides the shortlist:

| Run | Images | Shortlisted | All possible | Retained |
|---|---|---|---|---|
| revisit | 62 | 1860 | 1891 | **98%** |
| mixed3 | 75 | 2508 | 2775 | **90%** |

At 62 images the retrieval stage spends a minute of GPU time to discard 31 pairs out of
1891. Retrieval earns its cost from collection size, not from this parameter: at 12000
images an exhaustive pairing is 72 million pairs, and no threshold choice changes the
fact that shortlisting is what makes the problem finite.
