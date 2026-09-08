# Porting notes

The pipeline was written against Python 3.10 with pinned wheels — `pycolmap==0.6.1`,
`kornia==0.7.2`, `kornia_rs==0.1.2`, `lightglue==0.0` — installed offline from a local
wheel directory.

Three years later on Colab (Python 3.13.15), none of those cp310 wheels resolve. Current
PyPI versions install cleanly:

```
pycolmap     4.2.0
kornia       0.8.3
transformers 5.16.1
torch        2.11.0+cu128
lightglue    from git+https://github.com/cvg/LightGlue.git
```

Four API changes then stop the code (sections 1-4). Each one below is what actually
failed, and what it was replaced with. Sections 5 to 7 do not stop anything -- they are
three properties of the current wheel that quietly change what a benchmark means.

---

## 1. COLMAP grew a rig/frame schema

**Symptom.** The classic helpers `database.py` and `h5_to_db.py` write the pre-3.12 SQLite
schema directly. Against a pycolmap 4.x database the mapper either rejects the file or
finds no usable images.

**Cause.** COLMAP 3.12 introduced rigs and frames. Every image must belong to a frame;
every frame must belong to a rig. The feature importer creates these automatically, but a
hand-rolled SQLite writer does not.

**Fix.** Write one trivial rig + frame per image through the modern `Database` API. This
replaces `h5_to_db.py` and `database.py` entirely:

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

## 2. `pycolmap.Database()` has no constructor

```
TypeError: pycolmap._core.Database: No constructor defined!
```

Construction moved to a factory:

```python
db = pycolmap.Database.open(database_path)   # creates the file if absent
```

---

## 3. `Image.cam_from_world` became a method

```
AttributeError: 'builtin_function_or_method' object has no attribute 'rotation'
```

It is now derived from the image's frame rather than stored on the image, so reading it as
an attribute silently yields a bound method. Defensive accessor:

```python
def cam_from_world(im):
    c = im.cam_from_world
    c = c() if callable(c) else c
    return c.rotation.matrix(), np.asarray(c.translation)
```

---

## 4. `kornia.io.load_image` is broken against current `kornia_rs`

```
AttributeError: module 'kornia_rs' has no attribute 'read_image_jpegturbo'.
Did you mean: 'read_image_jpeg'?
```

kornia 0.8.3 still calls the old name. Rather than pin around it, drop the dependency —
the loader's whole contract is `(1, 3, H, W)` float32 RGB in `[0, 1]`:

```python
def load_torch_image(fname, device=torch.device('cpu')):
    img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
    return t[None, ...].to(device)
```

Separately, `import kornia` no longer pulls in `kornia.utils`, so
`K.utils.get_cuda_device_if_available(0)` raises `AttributeError`. Use torch directly.

---

## 5. The PyPI `pycolmap` wheel has no CUDA

```python
>>> pycolmap.has_cuda
False
```

Every published wheel I could install is built without CUDA, and nothing warns about it.
`FeatureExtractionOptions.use_gpu = True` is accepted, stored, and then silently ignored;
the log line that gives it away is buried in COLMAP's glog output:

```
sift.cc:765] Creating SIFT CPU feature extractor
```

Two consequences worth stating before anyone quotes a timing from this repository:

- SIFT detection and nearest-neighbour matching run on the CPU. On a Colab T4 instance
  (2 vCPU) that is roughly 20 s per 6200x4100 image, so extraction alone on 75 images is
  about 25 minutes.
- The learned front end is unaffected -- ALIKED and LightGlue go through torch, which has
  its own CUDA. So **any end-to-end timing that compares the two front ends is comparing a
  GPU against a CPU and means nothing.** Only `incremental_mapping` is comparable: COLMAP
  maps on the CPU either way, and both configurations reach it through the identical call.

Also note `use_gpu` defaults to `False` in the Python bindings and `True` in the COLMAP
command line, so a pycolmap script silently runs a different configuration from the CLI
command it was transcribed from.

---

## 6. pycolmap 4.2 ships ALIKED and LightGlue itself

```python
>>> [t for t in dir(pycolmap.FeatureExtractorType) if not t.startswith('_')]
['ALIKED_N16ROT', 'ALIKED_N32', 'LOMA_B', 'LOMA_B128', 'SIFT', 'UNDEFINED', ...]
>>> [t for t in dir(pycolmap.FeatureMatcherType) if not t.startswith('_')]
['ALIKED_BRUTEFORCE', 'ALIKED_LIGHTGLUE', ..., 'SIFT_BRUTEFORCE', 'SIFT_LIGHTGLUE', ...]
```

The enums accept these values. If they work end to end, the HDF5 detour this pipeline
takes -- write keypoints and matches to HDF5, then hand-build the COLMAP database -- is
no longer necessary, and the rig/frame problem in section 1 disappears with it, because
COLMAP's own importer creates rigs and frames. I did not test the native path here: this
study needed the hand-built database anyway, since the whole point was to measure the two
front ends through one identical downstream. Anyone porting this pipeline forward again
should check the native path first.

---

## 7. `max_num_features` is a knob, not a keypoint count

Setting `extraction_options.sift.max_num_features = 4600` and then reading the database
back gives **7592 keypoints per image**, not 4600. My first reading of that was that the
nested assignment silently failed. It does not -- I measured it on three Mill 19 frames at
`max_image_size = 1024`, varying one thing at a time:

| configured cap | `max_num_orientations` | keypoints / image | ratio |
|---|---|---|---|
| 500 | 1 | 580 | 1.16 |
| 4600 | 1 | 6646 | 1.44 |
| 4600 | 2 (default) | 7964 | 1.73 |
| 8192 (default) | 2 | 12644 | 1.54 |

The cap is honoured and monotonic -- 500 in, 580 out -- but what lands in the database
systematically exceeds it, by a factor that grows with the cap. Part of that is
`max_num_orientations = 2`: a feature with two dominant orientations is written as two
keypoint rows sharing one location. Turning it off takes 1.73x down to 1.44x, so the rest
comes from how the cap is applied across the scale-space pyramid, not from orientations.

Setting the value three different ways -- direct nested assignment, read-modify-write, and
assigning a freshly constructed `SiftExtractionOptions` -- gives byte-identical results, so
there is no pybind copy-semantics trap here.

**Why it matters for a comparison.** "Both front ends at 4600 features" is a statement
about configuration, not about what the matcher actually sees. ALIKED's 4600 is exact;
SIFT's 4600 became 7592. Any study that matches the two by the config value is handing the
SIFT arm roughly 65% more keypoints. To match realized counts, extract once, read
`Database.num_keypoints()`, and solve for the cap -- or set `max_num_orientations = 1` and
scale the cap by the measured 1.44x.

---

## Not an API break, but worth knowing

**`min_pairs=58` stops doing anything below ~250 images.** It forces each image to keep its
58 nearest DINOv2 neighbours, which on a 62-image set means keeping essentially all of
them. See the retrieval table in the README: 98% / 90% / 88% retention at 62 / 75 / 120
images, dropping to 53% only at 251. Scale it with the collection size, or the retrieval
stage is pure cost.

**`max_num_models=25` is expensive on mixed-scene input.** On a 251-image two-site set the
incremental mapper spent over an hour exploring sub-models when the correct answer was 2.
I set it near the number of scenes actually expected in the input.

**Colab keeps executing after the browser detaches.** A long `pycolmap.incremental_mapping`
call blocks the kernel's comm loop, so the frontend shows "Connecting" / "Resuming
execution" and execution counts reset to `[ ]`. The kernel is alive; check
Runtime → Manage sessions rather than trusting the cell indicator.

**A Jupyter error cancels the whole queued run.** A trailing diagnostic that raises will
discard every cell queued behind it, however expensive. Wrap each stage in its own
`try/except` instead of chaining cells.
