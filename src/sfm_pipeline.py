"""
Structure-from-Motion matching pipeline: DINOv2 shortlist -> ALIKED -> LightGlue -> COLMAP.

Default feature budgets retain the historical configuration; shortlist bookkeeping
is corrected, with an explicit legacy policy for historical comparisons. Ported to
Python 3.13 / pycolmap 4.2 (rig+frame schema) / kornia 0.8.3 / transformers 5.x;
weights are pulled from the HF hub and torch hub.
"""

import os
import gc
import numpy as np

DINO_ID = 'facebook/dinov2-base'

# ---------------------------------------------------------------- image io

def load_torch_image(fname, device='cpu'):
    """(1, 3, H, W) float32 in [0, 1], RGB -- same contract as
    K.io.load_image(..., ImageLoadType.RGB32).

    NOTE: the original used kornia's loader, but kornia 0.8.3 against a newer
    kornia_rs raises `module 'kornia_rs' has no attribute 'read_image_jpegturbo'`
    (the function was renamed to read_image_jpeg). cv2 avoids the version coupling.
    """
    import cv2
    import torch

    img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'cannot read {fname}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
    return t[None, ...].to(device)


# ------------------------------------------------------- global descriptors

_DINO_CACHE = {}


def _get_dino(device):
    from transformers import AutoImageProcessor, AutoModel

    key = str(device)
    if key not in _DINO_CACHE:
        processor = AutoImageProcessor.from_pretrained(DINO_ID)
        model = AutoModel.from_pretrained(DINO_ID).eval().to(device)
        _DINO_CACHE[key] = (processor, model)
    return _DINO_CACHE[key]


def get_global_desc(fnames, device='cpu'):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    if not fnames:
        raise ValueError('at least one image is required for descriptors')
    processor, model = _get_dino(device)
    descs = []
    for fname in tqdm(fnames, desc='dinov2', leave=False):
        timg = load_torch_image(fname)
        with torch.inference_mode():
            inputs = processor(images=timg, return_tensors='pt', do_rescale=False).to(device)
            outputs = model(**inputs)
            # MAC pooling over patch tokens (drop CLS), L2-normalised
            dino_mac = F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=1, p=2)
        descs.append(dino_mac.detach().cpu())
    return torch.cat(descs, dim=0)


def get_img_pairs_exhaustive(img_fnames):
    return [(i, j) for i in range(len(img_fnames)) for j in range(i + 1, len(img_fnames))]


def pairs_from_distances(distances, distance_threshold=0.3, min_pairs=58,
                         policy='corrected'):
    """Build an undirected shortlist from descriptor distances.

    Corrected mode guarantees up to ``min_pairs`` OTHER neighbours for every image,
    and includes all neighbours within the threshold. Legacy mode preserves the
    historical self-counting and last-query omission for comparison with old runs.
    """
    dm = np.asarray(distances, dtype=float)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError('distances must be a square matrix')
    if not np.isfinite(dm).all() or (dm < 0).any():
        raise ValueError('distances must be finite and nonnegative')
    if not np.isfinite(distance_threshold) or distance_threshold < 0 or min_pairs < 0:
        raise ValueError('distance threshold and min_pairs must be nonnegative')
    if policy not in ('corrected', 'legacy'):
        raise ValueError('policy must be corrected or legacy')
    n = len(dm)
    pairs = set()
    for i in range(n - 1 if policy == 'legacy' else n):
        candidates = np.flatnonzero(dm[i] <= distance_threshold)
        if policy == 'corrected':
            candidates = candidates[candidates != i]
            if len(candidates) < min(min_pairs, max(n - 1, 0)):
                order = np.argsort(dm[i], kind='stable')
                candidates = order[order != i][:min_pairs]
        elif len(candidates) < min_pairs:
            candidates = np.argsort(dm[i])[:min_pairs]
        for j in candidates:
            if i != j and (policy != 'legacy' or dm[i, j] < 1000):
                pairs.add(tuple(sorted((i, int(j)))))
    return sorted(pairs)


def get_image_pairs_shortlist(fnames, sim_th=0.3, min_pairs=58,
                              exhaustive_if_less=22, device='cpu',
                              policy='corrected'):
    """Shortlist with L2 distance on normalized DINOv2 descriptors.

    ``sim_th`` is retained as a CLI-compatible name; it is a distance threshold.
    Use policy='legacy' to retain the original shortlist bookkeeping.
    """
    if exhaustive_if_less < 0 or min_pairs < 0 or not np.isfinite(sim_th) or sim_th < 0:
        raise ValueError('shortlist thresholds must be finite and nonnegative')
    if policy not in ('corrected', 'legacy'):
        raise ValueError('policy must be corrected or legacy')
    if len(fnames) <= exhaustive_if_less or len(fnames) < 2:
        return get_img_pairs_exhaustive(fnames)
    import torch

    descs = get_global_desc(fnames, device=device)
    distances = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    return pairs_from_distances(distances, sim_th, min_pairs, policy)


# --------------------------------------------------------------- detection

def detect_aliked(img_fnames,
                  feature_dir='.featureout',
                  num_features=4600,
                  resize_to=1024,
                  detection_threshold=0.08,
                  device='cpu'):
    import h5py
    import torch
    from lightglue import ALIKED
    from tqdm import tqdm

    dtype = torch.float32  # ALIKED has issues with float16
    extractor = ALIKED(max_num_keypoints=num_features,
                       detection_threshold=detection_threshold).eval().to(device, dtype)
    os.makedirs(feature_dir, exist_ok=True)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in tqdm(img_fnames, desc='aliked', leave=False):
            key = os.path.basename(img_path)
            with torch.inference_mode():
                image0 = load_torch_image(img_path, device=device).to(dtype)
                # LightGlue's Extractor reads resize from extraction kwargs, not
                # ALIKED's constructor configuration. Keypoints return in input pixels.
                feats0 = extractor.extract(image0, resize=resize_to)
                kpts = feats0['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                descs = feats0['descriptors'][0].detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    del extractor
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------- matching

def match_with_lightglue(img_fnames,
                         index_pairs,
                         feature_dir='.featureout',
                         device='cpu',
                         min_matches=20,
                         verbose=False):
    import h5py
    import torch
    import kornia.feature as KF
    from tqdm import tqdm

    lg_matcher = KF.LightGlueMatcher('aliked', {
        'width_confidence': -1,
        'depth_confidence': -1,
        'mp': True if 'cuda' in str(device) else False,
    }).eval().to(device)

    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
         h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for idx1, idx2 in tqdm(index_pairs, desc='lightglue', leave=False):
            key1 = os.path.basename(img_fnames[idx1])
            key2 = os.path.basename(img_fnames[idx2])
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            if len(kp1) == 0 or len(kp2) == 0:
                continue
            with torch.inference_mode():
                _, idxs = lg_matcher(desc1, desc2,
                                         KF.laf_from_center_scale_ori(kp1[None]),
                                         KF.laf_from_center_scale_ori(kp2[None]))
            if len(idxs) == 0:
                continue
            n_matches = len(idxs)
            if verbose:
                print(f'{key1}-{key2}: {n_matches} matches')
            if n_matches >= min_matches:
                group = f_match.require_group(key1)
                group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))

    del lg_matcher
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------- colmap db (v4 API)

def import_into_colmap(img_dir, feature_dir='.featureout', database_path='colmap.db'):
    """Replaces the classic `h5_to_db.py` SQLite writer. pycolmap >= 4 requires every
    image to belong to a frame, and every frame to a rig, so this writes one trivial
    rig+frame per image (equivalent to COLMAP's own feature importer)."""
    import h5py
    import pycolmap

    if os.path.exists(database_path):
        os.remove(database_path)
    db = pycolmap.Database.open(database_path)

    fname_to_id = {}
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp:
        for key in list(f_kp.keys()):
            path = os.path.join(img_dir, key)
            probe = pycolmap.infer_camera_from_image(path)
            # SIMPLE_PINHOLE with the original focal prior, 1.2 * max(w, h)
            cam = pycolmap.Camera.create_from_model_id(
                pycolmap.INVALID_CAMERA_ID,
                pycolmap.CameraModelId.SIMPLE_PINHOLE,
                1.2 * max(probe.width, probe.height),
                probe.width, probe.height)
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

            db.write_keypoints(image.image_id, f_kp[key][...].astype(np.float32))
            fname_to_id[key] = image.image_id

    n_pairs = 0
    with h5py.File(f'{feature_dir}/matches.h5', mode='r') as f_match:
        for key1 in f_match.keys():
            for key2 in f_match[key1].keys():
                m = f_match[key1][key2][...].astype(np.uint32)
                db.write_matches(fname_to_id[key1], fname_to_id[key2], m)
                n_pairs += 1
    db.close()
    return fname_to_id, n_pairs


# ----------------------------------------------------------- ground truth

def read_colmap_images_txt(path):
    """Read COLMAP's two-line image records, including empty point lists.

    Ground-truth basenames must be unique because the flat dataset uses basenames.
    """
    poses = {}
    with open(path, encoding='utf-8') as handle:
        lines = iter(handle)
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=9)
            if len(parts) != 10:
                raise ValueError(f'{path}: malformed image record: {line}')
            q = np.asarray(list(map(float, parts[1:5])))
            t = np.asarray(list(map(float, parts[5:8])))
            name = os.path.basename(parts[9])
            if name in poses:
                raise ValueError(f'{path}: duplicate image basename {name!r}')
            if not np.isfinite(t).all():
                raise ValueError(f'{path}: non-finite translation for {name}')
            poses[name] = (quat_to_rotmat(q), t)
            # Empty lines are valid point lists and MUST consume this second record.
            point_line = next(lines, None)
            while point_line is not None and point_line.lstrip().startswith('#'):
                point_line = next(lines, None)
            if point_line is None:
                raise ValueError(f'{path}: missing point-list line for {name}')
    return poses


def quat_to_rotmat(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('quaternion must have four finite values and nonzero norm')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def cam_from_world(image):
    """pycolmap 4.x exposes Image.cam_from_world as a METHOD (it is derived from
    the image's frame), where 0.6.x exposed it as an attribute."""
    c = image.cam_from_world
    c = c() if callable(c) else c
    return c.rotation.matrix(), np.asarray(c.translation)


# ---------------------------------------------------------------- metrics

def relative_pose(Ri, ti, Rj, tj):
    R = Rj @ Ri.T
    t = tj - R @ ti
    return R, t


def rot_err_deg(R1, R2):
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def trans_err_deg(t1, t2):
    n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = float(np.dot(t1, t2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(abs(cos), -1.0, 1.0))))


def pose_auc(errors, thresholds=(5, 10, 20)):
    """AUC of the pose-error cumulative curve, the convention used across the local
    feature matching literature (SuperGlue, LoFTR, LightGlue).

    `errors` are per-pair angular errors in degrees, where the error of a pair is
    max(rotation angular error, translation angular error) -- the same definition
    LightGlue reports AUC@5/10/20 against.
    """
    thresholds = tuple(thresholds)
    if any(not np.isfinite(t) or t <= 0 for t in thresholds):
        raise ValueError('AUC thresholds must be finite and positive')
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or np.isnan(errors).any() or (errors < 0).any():
        raise ValueError('pose errors must be a vector of nonnegative values without NaN')
    if len(errors) == 0:
        return [float('nan')] * len(thresholds)
    errors = np.sort(errors)
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    out = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1] if last > 0 else 0.0]
        e = np.r_[errors[:last], t]
        area = np.sum((r[1:] + r[:-1]) * np.diff(e) / 2)
        out.append(float(area / t))
    return out


def relative_pose_errors(pred_poses, gt_poses, names=None, clusters=None):
    """Per-pair angular error, max(rotation, translation direction), in degrees.

    A pair whose images are unregistered -- or which lands in two different
    reconstructions, where the relative pose is meaningless -- scores 180.
    """
    names = sorted(names if names is not None else gt_poses.keys())
    errs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            Rg, tg = relative_pose(*gt_poses[na], *gt_poses[nb])
            split = clusters is not None and clusters.get(na) != clusters.get(nb)
            if na not in pred_poses or nb not in pred_poses or split:
                errs.append(180.0)
                continue
            Rp, tp = relative_pose(*pred_poses[na], *pred_poses[nb])
            errs.append(max(rot_err_deg(Rp, Rg), trans_err_deg(tp, tg)))
    return np.array(errs)


def camera_center(pose):
    R, t = pose
    return -R.T @ t


def umeyama(src, dst):
    """Least-squares similarity transform (scale, rotation, translation) taking
    `src` onto `dst`. An SfM reconstruction is only defined up to a Sim(3), so this
    is the alignment step before any absolute pose comparison."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError('alignment requires matching (N, 3) arrays with N >= 3')
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError('alignment coordinates must be finite')
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    if np.linalg.matrix_rank(S) < 2 or np.linalg.matrix_rank(D) < 2:
        raise ValueError('Sim(3) alignment is underdetermined for coincident or collinear centres')
    U, Sig, Vt = np.linalg.svd(D.T @ S / len(src))
    F = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        F[2, 2] = -1                      # keep it a rotation, never a reflection
    R = U @ F @ Vt
    var = (S ** 2).sum() / len(src)
    s = float((Sig * np.diag(F)).sum() / var) if var > 1e-12 else 1.0
    return s, R, mu_d - s * R @ mu_s


def absolute_pose_errors(pred_poses, gt_poses, names):
    """Camera-centre error (in the ground truth's units) and absolute rotation error,
    after aligning the reconstruction to ground truth with a Sim(3). This is the
    Structure-from-Motion convention, and on metric ground truth it reports metres.

    Returns (position_errors, rotation_errors_deg, scale).
    """
    names = [n for n in names if n in pred_poses and n in gt_poses]
    if len(names) < 3:
        return np.array([]), np.array([]), float('nan')
    P = np.array([camera_center(pred_poses[n]) for n in names])
    G = np.array([camera_center(gt_poses[n]) for n in names])
    try:
        s, Ra, ta = umeyama(P, G)
    except ValueError:
        return np.array([]), np.array([]), float('nan')
    pos = np.linalg.norm((s * (Ra @ P.T).T + ta) - G, axis=1)
    rot = np.array([rot_err_deg(pred_poses[n][0] @ Ra.T, gt_poses[n][0]) for n in names])
    return pos, rot, s
