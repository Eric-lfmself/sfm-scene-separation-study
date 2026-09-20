# Figures and visual evidence

I organize the visuals around the question I am investigating: can a reconstruction register many images while connecting the wrong places or estimating the wrong camera poses? The overview explains the three configurations; the schematic separates failure types; the result plots show the historical measurements with their controls and limitations.

Every current figure has an editable SVG with native text, a PNG with an explicit white background, and a description below. These are original diagrams and charts. The illustrated plates use synthetic geometry to explain the method; they contain no dataset photographs or measured reconstruction outputs. The summary plots have **not** been independently reproduced by rerunning the experiments.

## Figure guide and text alternatives

### Illustrated method: from image collections to 3D scene structure

[PNG preview](method-visual.png) · [Editable SVG](method-visual.svg) · [Geometry manifest](method-visual.manifest.json)

**Alt text:** An illustrated SfM workflow shows perspective views of a warm stone courtyard, descriptors and a similarity matrix, a verified image graph, and an architectural point cloud with camera frusta. Lower panels enlarge feature correspondences and reconstruction geometry.

**Long description:** I built an original L-shaped courtyard in one synthetic 3D coordinate system, including repeated windows, masonry bands, roof panels, paving, planters and a railing. Three pinhole-camera projections depict this same scene from different viewpoints. In the correspondence close-up, colored lines join the projections of identical 3D landmarks; dashed orange lines illustrate rejected candidate matches. The sparse cloud samples surfaces of the same mesh and is shown with camera frusta and a trajectory. Three colored rays meet at one shared facade landmark, illustrating how corresponding observations constrain a 3D point. The descriptor bars, six-view similarity matrix, candidate pair labels and mixed-scene graph are separate explanatory examples; none are extracted model outputs or experimental measurements. The three comparison paths use open labels and small matching symbols; their thin connector lines join before COLMAP geometry and the shared incremental mapper. Scene grouping, camera position and camera orientation remain distinct output checks.

The visible label **“Illustrative geometry · not an experimental reconstruction”** applies to all image-like content in this figure. No photographs, paper images or learned-model outputs are used. I use concrete visual transformations to explain the method while keeping the recorded result plots separate. The layout is original; it follows the general scientific-figure principle of an overview plus enlarged intermediate stages.

The independent builder is [`tools/make_method_visual.py`](../tools/make_method_visual.py). It records seed `20260920`, the perspective-projection method, the script hash and rendering versions in its sidecar manifest. SVG elements remain editable, with the same explicit sans-serif fallback stack used by the other figures.

### Illustrated failure examples

[PNG preview](failure-visual.png) · [Editable SVG](failure-visual.svg) · [Geometry manifest](failure-visual.manifest.json)

**Alt text:** Three synthetic 3D scenes illustrate a false merge between different sites, a false split between visits to one courtyard, and a camera with a correct centre but an incorrect viewing direction.

**Long description:** I draw a tower-and-courtyard site and a different shed-and-tank site joined by dashed red false links. The second panel repeats the same recognizable courtyard with disconnected blue and teal session groups. The third panel shows a registered trajectory around one building, with a reference frustum and an erroneous red frustum sharing a centre. Smaller view cards and an enlarged orientation inset explain the local mechanisms. All buildings, point clouds, views and poses are deterministic explanatory geometry, not measured results. The reversed viewing direction is one example of a rotation error; it is not a claim that every recorded rotation error above 170 degrees represents a reversed optical axis.

The builder is [`tools/make_failure_visual.py`](../tools/make_failure_visual.py). Its sidecar records the geometry seed and generator hash. The simpler [failure-mode reference](#failure-modes) remains available below.

### Framework overview

[SVG](framework-overview.svg) · [PNG](framework-overview.png)

**Alt text:** Three front ends feed COLMAP reconstruction and evaluation: DINOv2 with ALIKED/LightGlue, the same shortlist procedure with SIFT/nearest-neighbour matching, and exhaustive SIFT matching. Evaluation covers separation, pose geometry and CPU mapping cost.

**Long description:** An unordered image collection enters either DINOv2 pair shortlisting or exhaustive pairing. The two shortlist arms use the same retrieval procedure, a 1024-pixel input limit and a configured cap of 4600 features. They change both the detector and matcher. The historical realized pair lists are unavailable, so I need saved lists from fresh runs to verify their identity. Camera initialization and import paths also differ. The default SIFT arm uses exhaustive pairs, a 3200-pixel input limit and a cap of 8192 features. Equal configured caps do not guarantee equal realized keypoint counts. COLMAP performs geometric verification through the arm-specific database paths; incremental mapping and the evaluation code are shared. I inspect registered images, scene clusters, purity, verified-pair precision, relative-pose AUC, aligned positions and rotations, and CPU mapping time. The diagram represents program structure, not an experimental image graph.

### Failure modes

[SVG](failure-modes.svg) · [PNG](failure-modes.png)

**Alt text:** Drawn graph examples contrast false merging of distinct places with false splitting of one revisited place; a camera diagram shows how positions can agree while orientations differ.

**Long description:** The first panel labels images from different places A and B and adds a spurious connection between their components. The second panel labels all images A but leaves two session components disconnected. The third panel places expected and estimated camera frusta at shared centres with opposite view directions as one illustrative orientation error. All node positions and frusta are manually drawn. A recorded full 3D rotation error above 170 degrees does **not** by itself prove the optical axis is reversed; roll can also produce a large rotation error. I inspect orientation as well as position, registration and cluster count.

### Single-scene results

[SVG](results-single-scenes.svg) · [PNG](results-single-scenes.png) · [Source CSV](../results/reported/single-scenes.csv)

**Alt text:** Four plots compare recorded ETH3D AUC@5°, position error, registration and mapping time. The learned configuration reports 1111 mm courtyard error; the shortlist SIFT configuration registers 8/14 pipes images.

**Long description:** Each panel uses the same configuration colors and marker identities. Pipes, terrace and courtyard contain 14, 23 and 38 images. Median camera-centre errors use a base-10 logarithmic axis in millimetres; the other axes are linear. Registration bars start at zero and show the original image counts. The default configuration reports full pipes registration and 5 mm position error, while the 1024-pixel SIFT shortlist arm reports 8/14 and 12 mm. Because default pairing, resolution and feature budget change together, the comparison does not isolate a single cause. AUC@10°, AUC@20° and rotation errors remain in the CSV and results tables. No replicate uncertainty is available.

### Mixed-scene and revisit results

[SVG](results-mixed-scenes.svg) · [PNG](results-mixed-scenes.png) · [Mixed CSV](../results/reported/mixed-scenes.csv) · [Pose CSV](../results/reported/mixed-scene-poses.csv) · [Revisit CSV](../results/reported/revisit.csv)

**Alt text:** The recorded learned configuration returns two reconstructions for both three distinct scenes and one revisited place. SIFT shortlist returns the expected three and one. Other panels show mixed-scene precision/purity, pose errors and mapping times.

**Long description:** Dashed cluster-count references show the expected three components for 75 mixed images and one component for 62 revisit images. The learned configuration reports two in each case. SIFT shortlist reports three and one; the default arm reports three for mixed scenes and has no reported revisit result, marked `NR`. Mixed-scene pair precision and purity are 0.8556 and 0.6933 for learned and 1.0000 for the two SIFT arms. Position-error points in millimetres use a base-10 log scale. The learned terrace component reports 2246 mm and eight cameras with full-rotation errors greater than 170 degrees. Registration is 75/75 for learned and default, versus 69/75 for shortlist. CPU mapping times are displayed separately. True revisit links cross sessions, so a same-session precision convention cannot be reused unchanged.

### UAV results

[SVG](results-uav.svg) · [PNG](results-uav.png) · [Source CSV](../results/reported/uav.csv)

**Alt text:** All configurations register 120 Mill 19 frames into one reconstruction. Dataset-reference evaluation reports learned median position error 0.098 normalized units and rotation error 64.33°, versus 0.001 units and under 1° for both SIFT configurations.

**Long description:** AUC is plotted at 5-, 10- and 20-degree thresholds, with connected points identifying each configuration. Position uses a base-10 logarithmic axis in Mega-NeRF normalized units, `u`, which are not metres and cannot be compared with the ETH3D millimetres. Rotation uses a linear degree axis; learned has 27/120 cameras with full-rotation errors above 170 degrees, while each SIFT arm reports zero. CPU mapping takes 2211.5, 911.6 and 1517.7 seconds for learned, shortlist and default. These are configuration comparisons with unequal feature counts; heterogeneous CPU/GPU front-end timings are omitted.

### Verified-link ranges

[SVG](link-ranges.svg) · [PNG](link-ranges.png) · [Source CSV](../results/reported/verified-link-ranges.csv)

**Alt text:** Summary ranges of inlier counts for recorded mixed-scene links: true within-scene links have n=889, median 308 and maximum 3376; false cross-scene links have n=150, median 22 and maximum 85.

**Long description:** The horizontal base-10 log axis shows RANSAC inlier counts. Thin lines span minimum to maximum, thick lines span the 10th to 90th percentiles, and differently shaped markers show medians. The two populations overlap at low inlier counts. These are distribution summaries, not confidence intervals. The separate revisit experiment reports 370 true cross-session links with median 51, but its full ranges are unavailable in the current results table. I do not reconstruct a histogram from quantiles or infer a universal inlier threshold. The conflicting old histogram remains in the [archive](archive/README.md).

## Rebuild and provenance

From the repository root:

```sh
python -m pip install -r requirements-figures.txt
python tools/make_figures.py
```

The generator reads the CSVs in [`results/reported/`](../results/reported/README.md); it performs no downloads and runs no reconstruction. Existing generated outputs are overwritten. SVG text stays editable with the explicit fallback stack `DejaVu Sans, Arial, Helvetica, sans-serif`; fonts are not embedded, so exact glyph widths can still vary by system. PNGs use Matplotlib’s rendered DejaVu Sans and preserve the inspected typography for embedded previews. PNGs use 180 pixels per inch at the declared figure dimensions, have an opaque white background, and provide a portable preview. These choices target repository reading, not a claim of compliance with a particular journal.

[`manifest.json`](manifest.json) records package versions, figure dimensions, source hashes, transformations and evidence status. I inspected all eight PNGs after export and adjusted labels, footnotes and diagram connections. The figures use direct value labels plus marker shapes or explicit categories, so their meaning does not depend only on color. I do not claim formal accessibility certification.

The original conflicting SVGs and generator are preserved byte-for-byte in [`archive/`](archive/README.md). Their numbers are not silently reconciled with the current result summary.
