"""
Mill 19 (Mega-NeRF) aerial frames as a dataset for the same experiment driver.

Two real UAV sites -- an industrial building and a rubble field -- flown as continuous
sequences of 1920 and 1657 frames. Every frame ships a pose, so unlike the earlier pass
over this data the aerial runs carry ground truth and not just cluster counts.

Two things about this dataset are worth stating before using it:

**Use the train split, not val.** The published val split samples roughly every 97th
frame of the flight, so consecutive val images barely overlap. Reconstructing it
fragments -- and the fragmentation looks exactly like the clustering failure this study
is about, while actually being missing overlap. A split built for novel-view synthesis
is not automatically a split for Structure-from-Motion.

**The poses are camera-to-world in the NeRF convention** (x right, y up, z backward),
stored as a 3x4 `c2w` next to `intrinsics = (fx, fy, cx, cy)`. COLMAP wants
world-to-camera with y down and z forward, so every pose goes through `FLIP` below.
`--check-poses` reports trajectory continuity and rotation orthonormality. These
checks cannot by themselves establish the physical camera-axis convention.
"""

import glob
import os

import numpy as np
from dataset_utils import sync_image_links

# NeRF/OpenGL camera axes -> COLMAP/OpenCV camera axes.
FLIP = np.diag([1.0, -1.0, -1.0])

SITES = {
    'building': 'building-pixsfm',
    'rubble': 'rubble-pixsfm',
}


def site_dir(data_root, site, split='train'):
    return os.path.join(data_root, SITES[site], split)


def frame_ids(data_root, site, split='train'):
    d = site_dir(data_root, site, split)
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(d, 'rgbs', '*.jpg')))


def read_pose(meta_path):
    """One Mega-NeRF metadata file -> (R_cw, t_cw) in COLMAP's world-to-camera form."""
    import torch

    # Tensor metadata only: never fall back to executing arbitrary pickle objects.
    m = torch.load(meta_path, map_location='cpu', weights_only=True)
    return convert_c2w(m['c2w'])


def convert_c2w(c2w):
    """Convert a finite rigid 3x4 camera-to-world matrix to COLMAP axes."""
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.shape != (3, 4) or not np.isfinite(c2w).all():
        raise ValueError('c2w must be a finite 3x4 matrix')
    R_wc, t_wc = c2w[:, :3], c2w[:, 3]
    if not np.allclose(R_wc.T @ R_wc, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R_wc), 1, atol=1e-4):
        raise ValueError('c2w rotation must be orthonormal with determinant +1')
    R_cw = FLIP @ R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw


def build_uav_dataset(work, data_root, name, sites, n_per_site, start=0, stride=1, output_dir=None):
    """Symlink `n_per_site` CONSECUTIVE frames per site into one folder.

    Consecutive is the whole point: it is what gives the frames enough overlap to
    reconstruct at all. Returns (images_dir, gt, labels) exactly like the ETH3D builder,
    so everything downstream is shared.
    """
    if not sites or len(set(sites)) != len(sites):
        raise ValueError('provide at least one distinct UAV site')
    if n_per_site <= 0 or start < 0 or stride <= 0:
        raise ValueError('n_per_site and stride must be positive; start must be nonnegative')
    d = output_dir or os.path.join(work, name, 'images')
    gt, labels, sources = {}, {}, {}
    multi = len(sites) > 1
    for s in sites:
        base = site_dir(data_root, s)
        ids = frame_ids(data_root, s)[start::stride][:n_per_site]
        if len(ids) < n_per_site:
            raise ValueError(f'{s}: wanted {n_per_site} frames from {start}, got {len(ids)}')
        for fid in ids:
            key = f'{s}__{fid}.jpg' if multi else f'{fid}.jpg'
            sources[key] = os.path.join(base, 'rgbs', f'{fid}.jpg')
            labels[key] = s
            gt[key] = read_pose(os.path.join(base, 'metadata', f'{fid}.pt'))
    sync_image_links(d, sources)
    return d, gt, labels


def check_poses(data_root, site='building', n=120, start=0):
    """Report continuity over a selected train window; this is not an axis proof.

    A fixed camera-axis flip preserves both orthonormality and consecutive optical-
    axis angles. Validate physical convention separately using documented metadata
    and image/pose correspondences. Printed indices identify the checked window.
    """
    if n < 2 or start < 0:
        raise ValueError('pose check needs at least two frames and a nonnegative start')
    ids = frame_ids(data_root, site)[start:start + n]
    if len(ids) < 2:
        raise ValueError(f'{site}: found fewer than two frames in the requested window')
    base = site_dir(data_root, site)
    poses = [read_pose(os.path.join(base, 'metadata', f'{fid}.pt')) for fid in ids]
    C = np.asarray([-R.T @ t for R, t in poses])
    Z = np.asarray([R[2] for R, _ in poses])
    steps = np.linalg.norm(np.diff(C, axis=0), axis=1)
    ang = np.degrees(np.arccos(np.clip(np.sum(Z[:-1] * Z[1:], axis=1), -1, 1)))
    residual = max(np.abs(R @ R.T - np.eye(3)).max() for R, _ in poses)
    print(f'  {site}: {len(ids)} consecutive train frames, indices {start}–{start + len(ids) - 1}')
    print(f'    baseline: median {np.median(steps):.3f}, '
          f'p90 {np.percentile(steps, 90):.3f}, max {steps.max():.3f}')
    print(f'    consecutive optical-axis angle: median {np.median(ang):.2f} deg, '
          f'max {ang.max():.2f} deg')
    print(f'    R orthonormal: max |R R^T - I| = {residual:.2e}')
    print('    Scope: continuity only; a fixed axis flip cannot be diagnosed by this check.')
    return steps, ang
