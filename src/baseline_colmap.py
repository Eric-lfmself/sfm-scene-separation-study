"""
COLMAP's own default pipeline, as the reference point for the learned one.

Out of the box COLMAP is SIFT detection, nearest-neighbour descriptor matching with
Lowe's ratio test and a cross-check, and exhaustive pairing. That is the thing a
DINOv2 + ALIKED + LightGlue pipeline is usually being compared against, so it is the
thing this study measures against.

Two configurations are provided, because "the baseline" conflates two independent
changes and I want to see which one each number comes from:

`default`    SIFT, COLMAP's stock budget (max 8192 features, images capped at 3200 px),
             exhaustive pairing. This is COLMAP as shipped.
`shortlist`  SIFT at the same configured feature cap and nominal input resolution as
             the learned pipeline (cap 4600, 1024 px), using the same DINOv2 shortlist
             procedure. The detector and matcher change together; realized feature
             counts and camera initialization can also differ.

`default` -> `shortlist` changes pairing, image resolution and feature budget together.
`shortlist` -> learned changes the detector and matcher; camera initialization and
realized feature counts can also differ. Neither is a matcher-only ablation.

Both write into the same COLMAP database format the learned pipeline writes, so mapping
and every metric downstream are byte-identical code paths.
"""

import os
from time import time

import numpy as np

KMAX = 2147483647  # COLMAP's kMaxNumImages, used to decode pair ids

# COLMAP's shipped defaults, restated here so the baseline is explicit rather than
# whatever a future pycolmap decides to change them to.
COLMAP_MAX_IMAGE_SIZE = 3200
COLMAP_MAX_NUM_FEATURES = 8192


def _extraction_options(max_image_size, max_num_features, gpu):
    import pycolmap

    o = pycolmap.FeatureExtractionOptions()
    o.type = pycolmap.FeatureExtractorType.SIFT
    o.use_gpu = gpu
    o.max_image_size = max_image_size
    o.sift.max_num_features = max_num_features
    return o


def _matching_options(gpu):
    """Nearest neighbour with Lowe's ratio test and cross-check -- COLMAP's defaults.

    `use_gpu` defaults to False in the Python bindings but True in the COLMAP CLI.
    GPU use depends on the installed COLMAP build and the selected device policy.
    Cross-arm timings must be interpreted with the actual device placement.
    """
    import pycolmap

    o = pycolmap.FeatureMatchingOptions()
    o.type = pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    o.use_gpu = gpu
    return o


def write_pair_list(images, pairs, path):
    """COLMAP's imported-pairing matcher reads image *names*, one pair per line."""
    if images is None:
        raise ValueError('images are required when writing a shortlist')
    names = [os.path.basename(p) for p in images]
    if len(set(names)) != len(names) or any(any(c.isspace() for c in n) for n in names):
        raise ValueError('COLMAP pair lists require unique image names without whitespace')
    with open(path, 'w') as f:
        for i, j in pairs:
            f.write(f'{names[i]} {names[j]}\n')
    return path


def run_colmap_baseline(images_dir, out_dir, images=None, pairs=None,
                        max_image_size=COLMAP_MAX_IMAGE_SIZE,
                        max_num_features=COLMAP_MAX_NUM_FEATURES,
                        gpu=True, seed=0):
    """Extract SIFT and match. Returns (db_path, timings).

    `pairs` is a list of index pairs into `images`; passing it restricts matching to
    that shortlist instead of pairing exhaustively. Mapping is deliberately left to the
    caller so that both pipelines go through the identical mapper call.
    """
    import pycolmap

    if pairs is not None and images is None:
        raise ValueError('images must accompany pairs')
    capability = getattr(pycolmap, 'has_cuda', False)
    has_cuda = bool(capability() if callable(capability) else capability)
    if gpu and not has_cuda:
        print('  SIFT uses CPU: this COLMAP build has no CUDA support.', flush=True)
        gpu = False
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, 'colmap.db')
    if os.path.exists(db_path):
        os.remove(db_path)

    T = {}
    t = time()
    pycolmap.extract_features(
        database_path=db_path, image_path=images_dir,
        extraction_options=_extraction_options(max_image_size, max_num_features, gpu))
    T['detect'] = time() - t

    db = pycolmap.Database.open(db_path)
    n_img, n_kp = db.num_images(), db.num_keypoints()
    db.close()
    print(f'  sift:      {n_img} images, {n_kp} keypoints '
          f'({n_kp/max(n_img,1):.0f}/image) in {T["detect"]:.1f}s', flush=True)

    t = time()
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.random_seed = seed
    if pairs is None:
        pycolmap.match_exhaustive(database_path=db_path,
                                  matching_options=_matching_options(gpu),
                                  verification_options=verification)
    elif pairs:
        pairing = pycolmap.ImportedPairingOptions()
        pairing.match_list_path = write_pair_list(
            images, pairs, os.path.join(out_dir, 'pairs.txt'))
        pycolmap.match_image_pairs(database_path=db_path,
                                   matching_options=_matching_options(gpu),
                                   pairing_options=pairing,
                                   verification_options=verification)
    # COLMAP's matcher runs geometric verification in the same pass, so unlike the
    # learned pipeline there is no separate RANSAC stage to time.
    T['match'] = time() - t
    print(f'  nn match:  {T["match"]:.1f}s', flush=True)
    return db_path, T


# ------------------------------------------------------------ matching precision

def summarize_matches(raw_counts, verified_counts, pair_scenes=None):
    """Summarize geometric consistency over ALL putative correspondences.

    Failed verification contributes zero inliers and retains its denominator. This
    is a RANSAC consistency proxy, not ground-truth correspondence precision.
    ``pair_scenes`` maps pair ids to physical scene labels (not capture sessions).
    """
    if any(n < 0 for n in raw_counts.values()) or any(n < 0 for n in verified_counts.values()):
        raise ValueError('match counts must be nonnegative')
    for pid, n in verified_counts.items():
        if n > raw_counts.get(pid, 0):
            raise ValueError(f'pair {pid}: inliers exceed the stored putative matches')
    n_putative = sum(raw_counts.values())
    n_inliers = sum(verified_counts.values())
    verified = [pid for pid, n in verified_counts.items() if n > 0]
    per_pair = [verified_counts.get(pid, 0) / n for pid, n in raw_counts.items() if n]
    verified_denominator = sum(raw_counts[pid] for pid in verified)
    n_same = sum(pair_scenes[pid][0] == pair_scenes[pid][1] for pid in verified) if pair_scenes is not None else None
    return dict(
        n_putative=n_putative, n_inliers=n_inliers,
        n_verified_pairs=len(verified), n_matched_pairs=len(raw_counts),
        n_same_scene_pairs=n_same,
        n_putative_verified_pairs=verified_denominator,
        inlier_ratio_verified_pairs=n_inliers / verified_denominator if verified_denominator else float('nan'),
        inlier_ratio=n_inliers / n_putative if n_putative else float('nan'),
        median_pair_inlier_ratio=float(np.median(per_pair)) if per_pair else float('nan'),
        pair_precision=n_same / len(verified) if pair_scenes is not None and verified else float('nan'),
    )


def matching_precision(db_path, labels=None):
    """Read pooled geometric consistency and physical-scene pair precision.

    The caller must use the same physical label for sessions from one place.
    """
    import pycolmap

    db = pycolmap.Database.open(db_path)
    try:
        raw = {pid: len(m) for pid, m in zip(*db.read_all_matches())}
        verified = {pid: len(tvg.inlier_matches) for pid, tvg in zip(*db.read_two_view_geometries())}
        id2name = {im.image_id: im.name for im in db.read_all_images()}
    finally:
        db.close()
    pair_scenes = None if labels is None else {
        pid: (labels[id2name[pid // KMAX]], labels[id2name[pid % KMAX]])
        for pid, n in verified.items() if n
    }
    out = summarize_matches(raw, verified, pair_scenes)
    print(f'  consistency: inlier ratio {out["inlier_ratio"]:.4f} '
          f'({out["n_inliers"]} / {out["n_putative"]} correspondences)'
          + (f'   | physical-scene pair precision {out["pair_precision"]:.4f} '
             f'({out["n_same_scene_pairs"]} / {out["n_verified_pairs"]} verified pairs)'
             if labels is not None else ''), flush=True)
    return out


def database_summary(db_path):
    """Record realized features and actual camera initialization for fair comparisons."""
    import pycolmap

    db = pycolmap.Database.open(db_path)
    try:
        cameras = {
            camera.camera_id: dict(model=camera.model.name, width=camera.width,
                                   height=camera.height, params=camera.params.tolist(),
                                   has_prior_focal_length=camera.has_prior_focal_length)
            for camera in db.read_all_cameras()
        }
        images = {
            image.name: dict(camera_id=image.camera_id,
                             n_keypoints=len(db.read_keypoints(image.image_id)))
            for image in db.read_all_images()
        }
    finally:
        db.close()
    return dict(cameras=cameras, images=images)
