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
`shortlist`  SIFT at the same feature budget and the same input resolution as the
             learned pipeline (4600 features, 1024 px), restricted to the same DINOv2
             shortlist. Only the detector and the matcher differ from the learned run,
             so the difference here is the matcher alone.

`default` -> `shortlist` isolates the effect of pair selection; `shortlist` -> learned
isolates the effect of the detector and matcher.

Both write into the same COLMAP database format the learned pipeline writes, so mapping
and every metric downstream are byte-identical code paths.
"""

import os
from time import time

import numpy as np
import pycolmap

KMAX = 2147483647  # COLMAP's kMaxNumImages, used to decode pair ids

# COLMAP's shipped defaults, restated here so the baseline is explicit rather than
# whatever a future pycolmap decides to change them to.
COLMAP_MAX_IMAGE_SIZE = 3200
COLMAP_MAX_NUM_FEATURES = 8192


def _extraction_options(max_image_size, max_num_features, gpu):
    o = pycolmap.FeatureExtractionOptions()
    o.type = pycolmap.FeatureExtractorType.SIFT
    o.use_gpu = gpu
    o.max_image_size = max_image_size
    o.sift.max_num_features = max_num_features
    return o


def _matching_options(gpu):
    """Nearest neighbour with Lowe's ratio test and cross-check -- COLMAP's defaults.

    `use_gpu` defaults to False in the Python bindings but True in the COLMAP CLI.
    The baseline gets the GPU, or a timing comparison against it means nothing.
    """
    o = pycolmap.FeatureMatchingOptions()
    o.type = pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    o.use_gpu = gpu
    return o


def write_pair_list(images, pairs, path):
    """COLMAP's imported-pairing matcher reads image *names*, one pair per line."""
    names = [os.path.basename(p) for p in images]
    with open(path, 'w') as f:
        for i, j in pairs:
            f.write(f'{names[i]} {names[j]}\n')
    return path


def run_colmap_baseline(images_dir, out_dir, images=None, pairs=None,
                        max_image_size=COLMAP_MAX_IMAGE_SIZE,
                        max_num_features=COLMAP_MAX_NUM_FEATURES,
                        gpu=True):
    """Extract SIFT and match. Returns (db_path, timings).

    `pairs` is a list of index pairs into `images`; passing it restricts matching to
    that shortlist instead of pairing exhaustively. Mapping is deliberately left to the
    caller so that both pipelines go through the identical mapper call.
    """
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
    if pairs is None:
        pycolmap.match_exhaustive(database_path=db_path,
                                  matching_options=_matching_options(gpu))
    else:
        pairing = pycolmap.ImportedPairingOptions()
        pairing.match_list_path = write_pair_list(
            images, pairs, os.path.join(out_dir, 'pairs.txt'))
        pycolmap.match_image_pairs(database_path=db_path,
                                   matching_options=_matching_options(gpu),
                                   pairing_options=pairing)
    # COLMAP's matcher runs geometric verification in the same pass, so unlike the
    # learned pipeline there is no separate RANSAC stage to time.
    T['match'] = time() - t
    print(f'  nn match:  {T["match"]:.1f}s', flush=True)
    return db_path, T


# ------------------------------------------------------------ matching precision

def matching_precision(db_path, labels=None):
    """Two precisions, both read straight out of the COLMAP database.

    `inlier_ratio`  pooled over every matched pair: RANSAC inliers / putative matches.
                    This is what "matching precision" means in the feature-matching
                    literature -- the fraction of proposed correspondences that survive
                    a geometric model.

    `pair_precision` only defined when scene labels are available: of the image pairs
                    that passed geometric verification, the fraction that really are
                    the same scene. A pair linking two different scenes is a false
                    positive no matter how self-consistent its correspondences look.
    """
    db = pycolmap.Database.open(db_path)
    raw = {pid: len(m) for pid, m in zip(*db.read_all_matches())}
    pair_ids, tvgs = db.read_two_view_geometries()
    id2name = {im.image_id: im.name for im in db.read_all_images()}
    db.close()

    inl_sum = put_sum = 0
    n_verified = n_same = 0
    per_pair = []
    for pid, tvg in zip(pair_ids, tvgs):
        n_inl = len(tvg.inlier_matches)
        if n_inl == 0:
            continue
        n_put = raw.get(pid, 0)
        if n_put:
            inl_sum += n_inl
            put_sum += n_put
            per_pair.append(n_inl / n_put)
        n_verified += 1
        if labels is not None:
            a, b = id2name[pid // KMAX], id2name[pid % KMAX]
            n_same += labels[a] == labels[b]

    out = dict(
        n_putative=put_sum, n_inliers=inl_sum, n_verified_pairs=n_verified,
        n_matched_pairs=len(raw),
        inlier_ratio=(inl_sum / put_sum) if put_sum else float('nan'),
        median_pair_inlier_ratio=float(np.median(per_pair)) if per_pair else float('nan'),
        pair_precision=(n_same / n_verified) if (labels is not None and n_verified)
        else float('nan'),
    )
    print(f'  precision: inlier ratio {out["inlier_ratio"]:.4f} '
          f'({inl_sum} / {put_sum} correspondences)'
          + (f'   | pair precision {out["pair_precision"]:.4f} '
             f'({n_same} / {n_verified} verified pairs same-scene)'
             if labels is not None else ''), flush=True)
    return out
