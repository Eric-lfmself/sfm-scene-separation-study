# Porting notes

I originally used Python 3.10 with pinned wheels (`pycolmap==0.6.1`,
`kornia==0.7.2`, `kornia_rs==0.1.2`, `lightglue==0.0`) installed offline from a local
wheel directory.

My later Colab notes record Python 3.13.15, where those cp310 wheels were incompatible.
I recorded the following replacement environment:

```
pycolmap     4.2.0
kornia       0.8.3
transformers 5.16.1
torch        2.11.0+cu128
lightglue    from git+https://github.com/cvg/LightGlue.git
```

I retain these versions, error messages and measurements as historical observations.
The original environment export and raw diagnostic logs are unavailable in this checkout;
I have not rerun these diagnostics during the repository cleanup. They do not describe
every current package version or build. See the [audit](AUDIT.md) for evidence limits and
the [method guide](METHOD.md) for the comparison's remaining confounds.

Sections 1–4 record compatibility failures and the changes I made. Sections 5–7 record
execution and feature-budget behavior that affected how I interpreted the comparison.

---

## 1. COLMAP grew a rig/frame schema

**Symptom.** My older `database.py` and `h5_to_db.py` helpers wrote the pre-3.12 SQLite
schema directly. During the port to pycolmap 4.x, I recorded database rejection or a
mapper that found no usable images.

**Compatibility change.** My notes identify the rig/frame schema introduced in COLMAP
3.12. The revised import path associates each image with a frame and each frame with a
rig. My older SQLite writer did not create these records.

**Change I made.** I replaced those database helpers with the `Database` API, creating
one trivial rig and frame per image:

```python
db  = pycolmap.Database.open(database_path)

cam = pycolmap.Camera.create_from_model_id(
        pycolmap.INVALID_CAMERA_ID,
        pycolmap.CameraModelId.SIMPLE_PINHOLE,
        1.2 * max(width, height),      # focal prior from the original config
        width, height)
cam.camera_id = db.write_camera(cam)

rig = pycolmap.Rig()
rig.add_ref_sensor(cam.sensor_id)
rig.rig_id = db.write_rig(rig)

image = pycolmap.Image(name=key, camera_id=cam.camera_id)
image.image_id = db.write_image(image)

frame = pycolmap.Frame()
frame.rig_id = rig.rig_id
frame.add_data_id(image.data_id)
frame.frame_id = db.write_frame(frame)

db.write_keypoints(image.image_id, keypoints.astype(np.float32))
```

Matches are unchanged: `db.write_matches(id1, id2, matches.astype(np.uint32))`.

---

## 2. `pycolmap.Database()` rejected direct construction

```
TypeError: pycolmap._core.Database: No constructor defined!
```

I used the factory method in the recorded pycolmap 4.2.0 environment:

```python
db = pycolmap.Database.open(database_path)   # creates the file if absent
```

---

## 3. `Image.cam_from_world` became a method

```
AttributeError: 'builtin_function_or_method' object has no attribute 'rotation'
```

In the binding I used, the pose was accessed through a method associated with the image's
frame. Reading it as an attribute returned a bound method. I retained an accessor that
handles either interface:

```python
def cam_from_world(im):
    c = im.cam_from_world
    c = c() if callable(c) else c
    return c.rotation.matrix(), np.asarray(c.translation)
```

---

## 4. An image-loader mismatch between Kornia and `kornia_rs`

```
AttributeError: module 'kornia_rs' has no attribute 'read_image_jpegturbo'.
Did you mean: 'read_image_jpeg'?
```

With Kornia 0.8.3 and the `kornia_rs` build installed in that session, I recorded this
missing-function error. I replaced that loader with OpenCV while preserving its output
contract: `(1, 3, H, W)` float32 RGB in `[0, 1]`. This is the historical replacement
snippet; the repository implementation also checks whether the image was read:

```python
def load_torch_image(fname, device=torch.device('cpu')):
    img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
    return t[None, ...].to(device)
```

I also recorded an `AttributeError` from `K.utils.get_cuda_device_if_available(0)` after
`import kornia` in that environment. I used torch directly for device selection.

---

## 5. The recorded pycolmap build did not expose CUDA

```python
>>> pycolmap.has_cuda
False
```

I recorded `pycolmap.has_cuda == False` in the installed build. Setting
`FeatureExtractionOptions.use_gpu = True` did not establish GPU execution: the extraction
log still reported a CPU extractor:

```
sift.cc:765] Creating SIFT CPU feature extractor
```

I interpreted the timings in that environment as follows:

- I recorded CPU execution for SIFT extraction and nearest-neighbour matching. On the
  Colab T4 instance (2 vCPU), extraction took roughly 20 s per 6200x4100 image, or about
  25 minutes for 75 images. These timings are historical observations, not estimates for
  other machines or builds.
- ALIKED and LightGlue used torch's CUDA support. **The reported front-end and end-to-end
  timings therefore mix CPU and GPU execution and do not isolate algorithmic speed.** I
  compare mapping-stage timings only within this recorded environment, where the arms
  used the same CPU and `incremental_mapping` call. Different input graphs and
  reconstructions still affect that cost; it is not a matcher-throughput comparison.

My notes also record `use_gpu=False` in the Python bindings and `True` in the COLMAP
command-line interface I compared. I now check the installed build, effective options
and execution logs explicitly rather than assuming that a CLI setting transfers to
Python. I have not verified these defaults for every release.

---

## 6. I found native feature-type enums in pycolmap 4.2

```python
>>> [t for t in dir(pycolmap.FeatureExtractorType) if not t.startswith('_')]
['ALIKED_N16ROT', 'ALIKED_N32', 'LOMA_B', 'LOMA_B128', 'SIFT', 'UNDEFINED', ...]
>>> [t for t in dir(pycolmap.FeatureMatcherType) if not t.startswith('_')]
['ALIKED_BRUTEFORCE', 'ALIKED_LIGHTGLUE', ..., 'SIFT_BRUTEFORCE', 'SIFT_LIGHTGLUE', ...]
```

I recorded these enum values but did not test their complete execution paths, required
assets or build support. Their presence alone does not establish that a native ALIKED or
LightGlue run works in that environment.

A working native path could replace my HDF5-to-database import and handle rig/frame
creation through COLMAP. I kept the external path for this study and used a common mapper
call afterward. That choice does not make camera initialization or all upstream settings
identical. I plan to validate the native path as a separate implementation before
comparing it with the historical results.

---

## 7. `max_num_features` is a knob, not a keypoint count

For the recorded Mill 19 aerial comparison, I noted **7592 keypoints per image** after
setting `extraction_options.sift.max_num_features = 4600`. I initially suspected that the
nested assignment had failed. I also recorded a separate diagnostic on three Mill 19
frames at `max_image_size = 1024`:

| configured cap | `max_num_orientations` | keypoints / image | ratio |
|---|---|---|---|
| 500 | 1 | 580 | 1.16 |
| 4600 | 1 | 6646 | 1.44 |
| 4600 | 2 (default) | 7964 | 1.73 |
| 8192 (default) | 2 | 12644 | 1.54 |

The diagnostic records 580 rows at a configured cap of 500, and more rows at the larger
caps. For the 4600 setting, changing `max_num_orientations` from 2 to 1 reduces the
reported ratio from 1.73x to 1.44x. Multiple orientations are therefore consistent with
part of the difference. The remaining excess does not, by itself, identify how the
extractor applies its cap across scales. I have not retained the raw diagnostic output
needed to investigate that mechanism further.

My notes report byte-identical results for three assignment styles: direct nested
assignment, read-modify-write, and a newly constructed `SiftExtractionOptions`. That
observation argues against an assignment problem in the tested configuration; it is not
a statement about all nested pybind options.

**Why it matters for my comparison.** A configured budget of 4600 does not establish
equal realized counts. The aerial notes report 4600 for ALIKED and 7592 for SIFT, roughly
65% more for SIFT. Those counts are specific to that dataset and run; the three-frame
diagnostic above reports a different SIFT count. Neither is a general multiplier or a
guarantee that ALIKED always reaches its cap.

For a future comparison, I will record `Database.num_keypoints()` and per-image counts
for each arm, then adjust and remeasure the settings on the chosen data. The observed
1.44x ratio is not a universal correction factor, and setting `max_num_orientations = 1`
also changes the extracted features.

---

## Not an API break, but worth knowing

**The historical shortlist retained most pairs in small collections.** My older notes
describe a regime below approximately 250 images and give 98% / 90% / 88% retention at
62 / 75 / 120 images, with 53% at 251 images. I preserve the 88% entry as an unresolved
historical value: the later research note gives 6378 of 7140 pairs for the 120-image run,
which is approximately 89%. No retained pair manifests resolve the discrepancy, and the
251-image exploratory run is outside the main results.

The legacy `min_pairs=58` implementation included the query image in its nearest-neighbor
selection and omitted the final image as a query. It therefore did not guarantee 58
other neighbors per image. This is a code issue separate from the effect of collection
size; see [METHOD.md](METHOD.md) and [AUDIT.md](AUDIT.md) for the correction and its
implications for reproduction. I will measure shortlist coverage and runtime again
instead of carrying those retention rates into new runs.

**I recorded a long mapping run with `max_num_models=25`.** On the exploratory
251-image two-site set, my notes report over an hour spent exploring sub-models for an
expected 2 scenes. I subsequently reduced the limit. This is an operational observation,
not a measured effect of that option alone; using an expected scene count also introduces
prior information that a future blind scene-separation evaluation must disclose.

**A detached Colab browser did not always mean computation had stopped.** During long
`pycolmap.incremental_mapping` calls, I recorded "Connecting" / "Resuming execution"
and execution counts resetting to `[ ]` while the session remained active. I did not
instrument the communication loop, so I do not assign a specific cause. I check
Runtime → Manage sessions and persistent logs before restarting a run.

**An error interrupted my queued notebook run.** A diagnostic exception prevented later
queued cells from running in the notebook session I used. I now prefer separate processes
with persistent logs and explicit exit-status checks for each arm. Exception handling
should record a failed stage and stop its dependent work, rather than allow an incomplete
run to appear successful.
