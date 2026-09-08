# Does the learned matching front end actually help?

A Structure-from-Motion pipeline built on **DINOv2 retrieval → ALIKED → LightGlue →
COLMAP**, ported forward to current library versions, and then measured against the front
end COLMAP already ships — **SIFT with nearest-neighbour matching** — on identical inputs,
identical image pairs, an identical feature cap, and an identical mapper.

I expected to be documenting how the learned front end fails in one specific way. What I
measured is that on this data it does not beat the classical one at all — on ground-level
scenes or on real UAV survey frames — and that the failure I set out to document is caused
by the learned matcher rather than being a property of the pipeline.

All numbers, all timings and the raw per-run breakdown are in [`RESULTS.md`](RESULTS.md).

---

## What is held identical

| | pair selection | detector | matcher | budget |
|---|---|---|---|---|
| `learned` | DINOv2 shortlist | ALIKED | LightGlue | cap 4600 (realized 4600), 1024 px |
| `colmap-shortlist` | DINOv2 shortlist | SIFT | nearest neighbour | cap 4600 (realized ~7592), 1024 px |
| `colmap-default` | exhaustive | SIFT | nearest neighbour | stock cap 8192 (realized ~11988), 3200 px |

`learned` and `colmap-shortlist` see **the same image pairs at the same input resolution**,
configured to the same feature cap; the only difference is the detector and the matcher.
`colmap-default` is COLMAP as shipped. Everything downstream of the feature database —
the mapper call, its options, every metric — is one shared code path.

One caveat on that word "budget", because it cuts against my own conclusion: a cap of 4600
means exactly 4600 keypoints for ALIKED, but COLMAP writes more than it is asked for —
7592 per image here. The SIFT arm therefore ran with about 65% more keypoints than the
learned arm, an advantage to the side that wins below. Measured behaviour and the reason
are in [`docs/porting-notes.md`](docs/porting-notes.md) §7.

Two views of pose accuracy are reported throughout: **AUC@5/10/20°** of the relative-pose
error, the convention in the feature-matching literature; and **Sim(3)-aligned absolute
error**, the Structure-from-Motion convention. ETH3D ground truth is metrically scaled
from laser scans, so the position figures are real millimetres.

---

## Result 1 — on single scenes the classical front end is better, and steadier

| Scene | learned | SIFT @ matched res | SIFT @ stock |
|---|---|---|---|
| `pipes` (14) | 14/14 · 0.890 · 6 mm | 8/14 · 0.241 | **14/14 · 0.928 · 5 mm** |
| `terrace` (23) | 23/23 · 0.893 · 16 mm | 23/23 · 0.931 · 7 mm | **23/23 · 0.932 · 7 mm** |
| `courtyard` (38) | 38/38 · 0.643 · **1111 mm** | 38/38 · 0.891 · **29 mm** | **38/38 · 0.899 · 31 mm** |

*(registered · AUC@5° · median camera-centre error)*

`courtyard` is a 36× accuracy gap, and it is not one unlucky run: across three
independent runs of the same learned configuration I measured **516 mm, 849 mm and
1111 mm**, while the two SIFT configurations landed on 29 mm and 31 mm. The learned front
end is not just less accurate on that scene — it is unstable between runs.

The `pipes` row is the one that nearly fooled me. At 1024 px SIFT registers 8 of 14
images, which reads as "the learned matcher wins on low-texture scenes". At full
resolution SIFT registers all 14 with the best accuracy in the table. That gap was
resolution, not SIFT, and the stock column is the control that caught it.

## Result 2 — the learned matcher welds two different places together

75 images from three scenes shuffled into one folder. Correct output is 3 reconstructions.

| | Registered | Clusters | Purity | Pair precision |
|---|---|---|---|---|
| learned | 75/75 | **2** | **0.693** | 0.856 (889/1039) |
| SIFT @ matched res | 69/75 | 3 ✓ | 1.000 | **1.000** (559/559) |
| SIFT @ stock | 75/75 | 3 ✓ | 1.000 | **1.000** (614/614) |

`courtyard` and `terrace` merged, and `terrace` was wrecked in the process — 8 of its 23
cameras ended up more than 170° from their true orientation while the positions still
looked plausible. Both scenes sit on the ETH Zürich campus and share facade, railing and
paving texture, so LightGlue found *real*, locally consistent correspondences between
them: 296 cross-scene pairs cleared its threshold and 150 survived geometric verification.

**This is not about feature density.** The stock configuration writes 11928 keypoints per
image — 2.6× what the learned arm gets, at three times the resolution — and produced
**zero** cross-scene verified pairs. More features did not manufacture false links; the learned
matcher did.

## Result 3 — and it splits one place in half

`relief` and `relief_2` are 31 + 31 images of the *same* interior photographed twice.
Correct output is 1 reconstruction.

| | Clusters | `relief` | `relief_2` |
|---|---|---|---|
| learned | **2** | 0.449, 58% registered | 0.289, **179.0°**, 20/20 cameras inverted |
| SIFT @ matched res | **1** ✓ | **0.674**, 100% | **0.899 · 8 mm · 0.337°** |

The learned run split the two sessions and then inverted one of them: after Sim(3)
alignment every camera in the `relief_2` model points backwards, while the positions still
fit to about a metre. The model looks plausible until you check where the cameras are
aimed.

## The two together

| Situation | Correct | learned | SIFT + NN |
|---|---|---|---|
| three different places in one folder | 3 clusters | **2** — merges what should be separate | 3 ✓ |
| one place photographed twice | 1 cluster | **2** — separates what should be merged | 1 ✓ |

Wrong in both directions, on the same data where SIFT with nearest-neighbour matching is
right in both, at 1/6 of the input resolution the stock configuration uses.

## Result 4 — on real aerial survey data, the gap is two orders of magnitude

Everything above is handheld ground-level capture. 120 consecutive frames from a Mill 19
UAV flight — constant altitude, two flight strips, ground truth poses — test whether it
transfers to the geometry the pipeline is meant for.

| | Registered | Clusters | AUC@5° | Median position | Median rotation |
|---|---|---|---|---|---|
| learned | 120/120 | 1 ✓ | 0.593 | **2.4 baselines** | **64.3°**, 27 cameras inverted |
| SIFT @ matched res | 120/120 | 1 ✓ | **0.947** | **1/41 baseline** | **0.94°**, none |
| SIFT @ stock | 120/120 | 1 ✓ | **0.953** | **1/41 baseline** | **0.52°**, none |

*(Mill 19 ships no scale factor, so positions are normalised units expressed as multiples
of the 0.041 median inter-frame baseline — not metres, and not comparable with the ETH3D
figures above.)*

**All three register every frame into one correct reconstruction.** On completeness and
cluster count — the two things you can check without ground truth — they tie. The ground
truth is what separates them: the learned reconstruction has a quarter of its cameras
facing backwards while its positions still look plausible. It is `relief_2` again, on real
survey data.

**The control matters here.** A 64° median could just as easily mean my ground truth is
wrong. It is not: the SIFT arm ran the same poses through the same evaluation, alignment
and mapper and returned 0.94°.

Mapping took 2211.5 s for the learned front end against 911.6 s — it handed the mapper
3367 verified pairs to SIFT's 2388, on the same CPU through the same call.

---

---

## A timing caveat, stated before any timing is quoted

`pycolmap.has_cuda` is `False` on every published wheel. SIFT detection and
nearest-neighbour matching run on the CPU; ALIKED and LightGlue go through torch and run
on the GPU. **Front-end timings across configurations compare a GPU against a CPU and are
meaningless.** `incremental_mapping` is the exception — COLMAP maps on the CPU either way,
through the same call — so mapping time is the only timing behind any claim here. On the
mixed set it was 543.8 s for the learned front end against 71.0 s and 199.7 s for the two
SIFT configurations, which is a consequence of how many two-view geometries each fed the
mapper.

## Retrieval is a separate question from matching

`min_pairs=58` keeps each image's 58 nearest DINOv2 neighbours, and at these collection
sizes that floor — not the similarity threshold — decides the shortlist: 98% of all
possible pairs retained at 62 images, 90% at 75. Retrieval earns its cost from collection
size rather than from parameter tuning: an exhaustive pairing over 12000 images is 72
million pairs, and shortlisting is what makes that finite. That argument is independent of
which matcher runs afterwards, and nothing measured here weakens it.

## What broke in the port

Written against Python 3.10 with pinned wheels that no longer resolve on 3.13. Current
PyPI versions install cleanly, but four API changes stop the code, and two properties of
the current wheel quietly change what a benchmark means. Details in
[`docs/porting-notes.md`](docs/porting-notes.md).

## Layout

```
src/sfm_pipeline.py      the ported learned front end, COLMAP ingestion, metrics
src/baseline_colmap.py   COLMAP's own front end, and the matching-precision measures
src/uav_dataset.py       Mill 19 aerial frames and the Mega-NeRF pose convention
src/run_experiments.py   dataset builders, the experiment driver, link diagnostics
RESULTS.md               every number and every timing
docs/porting-notes.md    the API breakages, with before/after
figures/                 static SVG, light and dark
tools/make_figures.py    regenerates them
```

## Running it

```bash
pip install pycolmap kornia kornia_moons h5py transformers opencv-python
pip install git+https://github.com/cvg/LightGlue.git

python src/run_experiments.py --fetch eth3d --scenes courtyard terrace pipes relief relief_2
python src/run_experiments.py --experiment mixed3 --pipeline learned
python src/run_experiments.py --experiment mixed3 --pipeline colmap-shortlist
python src/run_experiments.py --experiment mixed3 --pipeline colmap-default
python src/run_experiments.py --experiment revisit --pipeline learned

# aerial: expects Mill 19 under $SFM_UAV (building-pixsfm / rubble-pixsfm)
python src/run_experiments.py --check-poses
python src/run_experiments.py --experiment uav_building --pipeline learned --max-num-models 4
python src/run_experiments.py --experiment uav_building --pipeline colmap-shortlist --max-num-models 4
```

Needs a CUDA GPU for the learned front end. Verified on a Colab T4 with Python 3.13.15,
torch 2.11.0+cu128, pycolmap 4.2.0, kornia 0.8.3, transformers 5.16.1.

## Limitations

- Five ETH3D scenes, at most 75 images per run, one run per configuration except
  `courtyard` under `learned`, which I ran three times precisely because it was unstable.
  These are small numbers.
- ETH3D is ground-level handheld DSLR capture: high overlap, controlled, well textured.
  LightGlue's published advantages are largest under wide baselines, low overlap and
  illumination change, which this data does not test. **The result here is "on this data",
  not "learned matchers are worse".**
- I ran the learned front end at 1024 px. The matched SIFT arm ran at the same resolution,
  so that variable is controlled, but both are below what a production pipeline would use —
  and the `pipes` row shows how much resolution can matter. The realized keypoint counts
  are *not* matched: see the caveat under "What is held identical".
- The aerial section is one site, one 120-frame window, one run per configuration. It is
  a single flight over an industrial building, not a village, and nothing here establishes
  how either front end behaves across a 12000-image survey.
- Mill 19 has no metric scale factor, so its position errors are ratios, not distances.

## License

MIT — see [`LICENSE`](LICENSE). The datasets are not mine and carry their own terms; see
below.

## Data and credits

- **ETH3D** high-resolution multi-view — <https://www.eth3d.net/datasets>
- **Mill 19** (Mega-NeRF) — <https://meganerf.cmusatyalab.org>
- **LightGlue / ALIKED** — <https://github.com/cvg/LightGlue>
- **DINOv2** — `facebook/dinov2-base`
- **COLMAP / pycolmap** — <https://colmap.github.io>
