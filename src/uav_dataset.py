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
`--check-poses` verifies the conversion rather than trusting it.
"""

import glob
import os

import numpy as np
import torch

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
    m = torch.load(meta_path, map_location='cpu', weights_only=False)
    c2w = np.asarray(m['c2w'], dtype=np.float64)
    R_wc, t_wc = c2w[:, :3], c2w[:, 3]
    R_cw = FLIP @ R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw


def build_uav_dataset(work, data_root, name, sites, n_per_site, start=0, stride=1):
    """Symlink `n_per_site` CONSECUTIVE frames per site into one folder.

    Consecutive is the whole point: it is what gives the frames enough overlap to
    reconstruct at all. Returns (images_dir, gt, labels) exactly like the ETH3D builder,
    so everything downstream is shared.
    """
    d = os.path.join(work, name, 'images')
    os.makedirs(d, exist_ok=True)
    gt, labels = {}, {}
    multi = len(sites) > 1
    for s in sites:
        base = site_dir(data_root, s)
        ids = frame_ids(data_root, s)[start::stride][:n_per_site]
        if len(ids) < n_per_site:
            raise SystemExit(f'{s}: wanted {n_per_site} frames from {start}, got {len(ids)}')
        for fid in ids:
            key = f'{s}__{fid}.jpg' if multi else f'{fid}.jpg'
            dst = os.path.join(d, key)
            if not os.path.exists(dst):
                os.symlink(os.path.join(base, 'rgbs', f'{fid}.jpg'), dst)
            labels[key] = s
            gt[key] = read_pose(os.path.join(base, 'metadata', f'{fid}.pt'))
    return d, gt, labels


def check_poses(data_root, site='building', n=60):
    """Sanity-check the convention without running the pipeline.

    Over a continuous flight the camera moves smoothly, so consecutive baselines should
    be small and comparable, and consecutive optical axes should be nearly parallel. A
    wrong axis convention shows up here as optical axes that disagree by ~90 or ~180
    degrees while the positions still look plausible -- the same silent failure the
    Sim(3) rotation column catches later.
    """
    ids = frame_ids(data_root, site)[:n]
    base = site_dir(data_root, site)
    C, Z = [], []
    for fid in ids:
        R, t = read_pose(os.path.join(base, 'metadata', f'{fid}.pt'))
        C.append(-R.T @ t)          # camera centre in world coordinates
        Z.append(R[2])              # optical axis in world coordinates
    C, Z = np.asarray(C), np.asarray(Z)
    steps = np.linalg.norm(np.diff(C, axis=0), axis=1)
    dots = np.clip(np.sum(Z[:-1] * Z[1:], axis=1), -1, 1)
    ang = np.degrees(np.arccos(dots))
    print(f'  {site}: {len(ids)} consecutive frames')
    print(f'    baseline between consecutive frames: median {np.median(steps):.3f}, '
          f'p90 {np.percentile(steps, 90):.3f}, max {steps.max():.3f}')
    print(f'    turn between consecutive optical axes: median {np.median(ang):.2f} deg, '
          f'max {ang.max():.2f} deg')
    print(f'    R orthonormal: max |R R^T - I| = '
          f'{max(np.abs(np.asarray(read_pose(os.path.join(base, "metadata", f"{f}.pt"))[0]) @ np.asarray(read_pose(os.path.join(base, "metadata", f"{f}.pt"))[0]).T - np.eye(3)).max() for f in ids[:5]):.2e}')
    return steps, ang
