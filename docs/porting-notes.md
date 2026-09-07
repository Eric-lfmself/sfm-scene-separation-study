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

Four API changes then stop the code. Each one below is what actually failed, and what it
was replaced with.

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
