"""
Experiment driver for the SfM matching pipeline study.

Fetches ETH3D scenes, assembles single-scene / mixed-scene / revisit datasets,
runs the pipeline, and reports registration, cluster purity, pose AUC, Sim(3)-aligned
absolute pose error, and the cross-scene link diagnostics.

    python run_experiments.py --fetch eth3d --scenes courtyard terrace pipes
    python run_experiments.py --experiment mixed3
    python run_experiments.py --experiment mixed3  --min-inliers 100
    python run_experiments.py --experiment revisit --min-inliers 100

Needs a CUDA GPU. See ../README.md for the environment.
"""

import argparse
import os
import glob
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from time import time

import numpy as np
import h5py
import torch
import pycolmap

from baseline_colmap import (
    COLMAP_MAX_IMAGE_SIZE,
    COLMAP_MAX_NUM_FEATURES,
    matching_precision,
    run_colmap_baseline,
)
from uav_dataset import build_uav_dataset, check_poses
from sfm_pipeline import (
    absolute_pose_errors,
    detect_aliked,
    get_image_pairs_shortlist,
    import_into_colmap,
    match_with_lightglue,
    pose_auc,
    read_colmap_images_txt,
    relative_pose_errors,
)

KMAX = 2147483647  # COLMAP's kMaxNumImages, used to decode pair ids

DATA = os.environ.get('SFM_DATA', '/content/eth3d')
WORK = os.environ.get('SFM_WORK', '/content/work')
UAV = os.environ.get('SFM_UAV', '/content/data/mill19')

# Scenes used in the study. relief and relief_2 are the SAME physical interior
# photographed in two sessions -- the revisit case, where one cluster is correct.
EXPERIMENTS = {
    'pipes':     dict(scenes=['pipes']),
    'terrace':   dict(scenes=['terrace']),
    'courtyard': dict(scenes=['courtyard']),
    'mixed3':    dict(scenes=['courtyard', 'terrace', 'pipes'], expect_clusters=3),
    'revisit':   dict(scenes=['relief', 'relief_2'], expect_clusters=1, same_place=True),
    # Mill 19 aerial. Consecutive frames from the TRAIN split -- see uav_dataset.py for
    # why the published val split cannot be used for Structure-from-Motion.
    'uav_building': dict(uav=['building'], n_per_site=120, expect_clusters=1, units='u'),
    'uav_mixed':    dict(uav=['building', 'rubble'], n_per_site=120, expect_clusters=2, units='u'),
}


# ----------------------------------------------------------------- data

def fetch_eth3d(scenes):
    """Download + extract ETH3D high-res multi-view scenes (undistorted DSLR)."""
    os.makedirs(DATA, exist_ok=True)
    subprocess.run('apt-get -qq install -y p7zip-full', shell=True, capture_output=True)
    for s in scenes:
        if os.path.isdir(f'{DATA}/{s}/images'):
            print(f'{s}: already present')
            continue
        url = f'https://www.eth3d.net/data/{s}_dslr_undistorted.7z'
        print(f'{s}: downloading {url}', flush=True)
        subprocess.run(f'wget -q -O {DATA}/{s}.7z {url}', shell=True, check=True)
        subprocess.run(f'7z x -y -o{DATA} {DATA}/{s}.7z > /dev/null', shell=True, check=True)
        os.remove(f'{DATA}/{s}.7z')
        n = len(glob.glob(f'{DATA}/{s}/images/dslr_images_undistorted/*.JPG'))
        print(f'{s}: {n} images')


def scene_images(scene):
    return sorted(glob.glob(f'{DATA}/{scene}/images/dslr_images_undistorted/*.JPG'))


def build_dataset(name, scenes):
    """Symlink scenes into one folder. Returns (images_dir, gt, labels).

    For a multi-scene set, filenames are prefixed with the scene so they stay unique
    and so every image carries its true scene label for the purity metric.
    """
    d = f'{WORK}/{name}/images'
    os.makedirs(d, exist_ok=True)
    gt, labels = {}, {}
    multi = len(scenes) > 1
    for s in scenes:
        poses = read_colmap_images_txt(f'{DATA}/{s}/dslr_calibration_undistorted/images.txt')
        for p in scene_images(s):
            b = os.path.basename(p)
            key = f'{s}__{b}' if multi else b
            dst = os.path.join(d, key)
            if not os.path.exists(dst):
                os.symlink(p, dst)
            labels[key] = s
            if b in poses:
                gt[key] = poses[b]
    return d, gt, labels


# ------------------------------------------------------------- pipeline

def cam_from_world(im):
    """pycolmap 4.x turned this into a method derived from the image's frame."""
    c = im.cam_from_world
    c = c() if callable(c) else c
    return c.rotation.matrix(), np.asarray(c.translation)


def run_dataset(name, images_dir, device, args, labels=None):
    """Run one pipeline end to end. Every stage after the database is shared.

    `args.pipeline` selects the front end:
      learned           DINOv2 shortlist -> ALIKED -> LightGlue
      colmap-shortlist  DINOv2 shortlist -> SIFT -> nearest neighbour
      colmap-default    exhaustive       -> SIFT -> nearest neighbour   (COLMAP as shipped)
    """
    images = sorted(glob.glob(images_dir + '/*'))
    suffix = '' if args.pipeline == 'learned' else '_' + args.pipeline
    feature_dir = f'{WORK}/{name}/featureout{suffix}'
    os.makedirs(feature_dir, exist_ok=True)
    T = {}
    n_possible = len(images) * (len(images) - 1) // 2

    pairs = None
    if args.pipeline != 'colmap-default':
        t = time()
        pairs = get_image_pairs_shortlist(images, args.sim_th, args.min_pairs,
                                          args.exhaustive_if_less, device)
        T['shortlist'] = time() - t
        print(f'  shortlist: {len(pairs)} of {n_possible} possible pairs '
              f'({100*len(pairs)/max(n_possible,1):.0f}%) in {T["shortlist"]:.1f}s', flush=True)
    else:
        print(f'  pairing:   exhaustive, {n_possible} pairs', flush=True)

    if args.pipeline == 'learned':
        t = time()
        detect_aliked(images, feature_dir, args.num_features, args.resize_to, device=device)
        T['detect'] = time() - t
        print(f'  aliked:    {T["detect"]:.1f}s', flush=True)

        t = time()
        match_with_lightglue(images, pairs, feature_dir, device, args.min_matches)
        T['match'] = time() - t
        print(f'  lightglue: {T["match"]:.1f}s', flush=True)

        db_path = f'{feature_dir}/colmap.db'
        _, n_kept = import_into_colmap(images_dir, feature_dir, db_path)
        print(f'  kept {n_kept} pairs with >= {args.min_matches} matches', flush=True)

        t = time()
        pycolmap.match_exhaustive(db_path)
        T['ransac'] = time() - t
        print(f'  ransac:    {T["ransac"]:.1f}s', flush=True)
    else:
        stock = args.pipeline == 'colmap-default'
        db_path, Tb = run_colmap_baseline(
            images_dir, feature_dir, images=images, pairs=pairs,
            max_image_size=COLMAP_MAX_IMAGE_SIZE if stock else args.resize_to,
            max_num_features=COLMAP_MAX_NUM_FEATURES if stock else args.num_features)
        T.update(Tb)
        db = pycolmap.Database.open(db_path)
        n_kept = db.num_matched_image_pairs()
        db.close()

    prec = matching_precision(db_path, labels)

    preds, clusters, n_models = map_and_collect(name, db_path, images_dir, args, T)
    T['total'] = sum(v for k, v in T.items() if k != 'total')
    print(f'  TOTAL:     {T["total"]:.1f}s', flush=True)
    return dict(name=name, images=images, n_pairs=len(pairs) if pairs else n_possible,
                n_kept=n_kept, preds=preds, clusters=clusters, n_clusters=n_models,
                timings=T, precision=prec, feature_dir=feature_dir, db_path=db_path)


def map_and_collect(name, db_path, images_dir, args, T, tag=''):
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = args.min_model_size
    opts.max_num_models = args.max_num_models
    out = f'{WORK}/{name}/rec{tag}'
    os.makedirs(out, exist_ok=True)
    t = time()
    maps = pycolmap.incremental_mapping(database_path=db_path, image_path=images_dir,
                                        output_path=out, options=opts)
    T['mapping'] = time() - t
    print(f'  mapping:   {T["mapping"]:.1f}s', flush=True)
    preds, clusters = {}, {}
    for mi, rec in maps.items():
        for _, im in rec.images.items():
            preds[im.name] = cam_from_world(im)
            clusters[im.name] = mi
    return preds, clusters, len(maps)


def refilter_and_map(name, db_path, images_dir, min_inliers, args):
    """Drop verified two-view geometries below `min_inliers`, then re-run the mapper.

    Matching is untouched -- this is a post-verification edge filter only.
    """
    dst = db_path.replace('.db', f'_inl{min_inliers}.db')
    shutil.copy(db_path, dst)
    db = pycolmap.Database.open(dst)
    pair_ids, tvgs = db.read_two_view_geometries()
    dropped = 0
    for pid, tvg in zip(pair_ids, tvgs):
        if len(tvg.inlier_matches) < min_inliers:
            db.delete_two_view_geometry(pid // KMAX, pid % KMAX)
            dropped += 1
    db.close()
    print(f'  dropped {dropped} / {len(pair_ids)} geometries below {min_inliers} inliers',
          flush=True)
    T = {}
    return map_and_collect(name, dst, images_dir, args, T, tag=f'_inl{min_inliers}') + (T,)


# ------------------------------------------------------------ reporting

def show_clusters(clusters, labels, expect=None):
    comp = defaultdict(Counter)
    for n, c in clusters.items():
        comp[c][labels[n]] += 1
    print('  cluster composition:')
    for c in sorted(comp):
        print(f'    cluster {c:<3d} {dict(comp[c])}  size={sum(comp[c].values())}')
    total = sum(sum(v.values()) for v in comp.values())
    purity = sum(max(v.values()) for v in comp.values()) / max(total, 1)
    note = ''
    if expect is not None:
        note = '  <-- CORRECT' if len(comp) == expect else f'  <-- expected {expect}'
    print(f'    purity = {purity:.4f}  ({len(comp)} clusters over {total} registered){note}')
    return purity


def eval_scene(preds, clusters, gt, labels, scene):
    """Two standard views of pose accuracy for one scene.

    AUC@5/10/20 of the relative-pose error, as reported across the feature-matching
    literature; and, after a Sim(3) alignment to ground truth, the absolute camera
    centre and rotation error, as reported in the SfM literature. The absolute figures
    are computed inside the scene's dominant reconstruction -- across two
    reconstructions there is no common gauge to align in.
    """
    names = sorted(n for n in gt if labels[n] == scene)
    errs = relative_pose_errors(preds, gt, names, clusters)
    auc = pose_auc(errs)

    reg = [n for n in names if n in preds]
    out = dict(auc=auc, n_pairs=len(errs), abs_pos=float('nan'),
               abs_rot=float('nan'), frac=0.0, flipped=0)
    if reg:
        dominant = Counter(clusters[n] for n in reg).most_common(1)[0][0]
        sel = [n for n in reg if clusters[n] == dominant]
        out['frac'] = len(sel) / len(names)
        pos, rot, _ = absolute_pose_errors(preds, gt, sel)
        if len(pos):
            out['abs_pos'] = float(np.median(pos))
            out['abs_rot'] = float(np.median(rot))
            out['flipped'] = int((rot > 170).sum())   # cameras pointing the wrong way
    return out


def link_diagnostics(feature_dir, db_path, labels):
    """Where do the cross-scene links come from, and how strong are they?"""
    # Only the learned front end writes matches.h5; COLMAP's matcher goes straight to
    # the database. The database half below works for either, and is the important half.
    h5 = f'{feature_dir}/matches.h5'
    if os.path.exists(h5):
        kept = Counter()
        with h5py.File(h5, 'r') as f:
            for k1 in f.keys():
                for k2 in f[k1].keys():
                    kept.update([tuple(sorted((labels[k1], labels[k2])))])
        print('  matcher pairs kept, by scene pair:')
        for k, v in sorted(kept.items(), key=lambda x: -x[1]):
            tag = 'WITHIN' if k[0] == k[1] else 'CROSS '
            print(f'    {tag}  {k[0]} | {k[1]:<24s} {v:6d}')

    db = pycolmap.Database.open(db_path)
    id2name = {im.image_id: im.name for im in db.read_all_images()}
    pair_ids, tvgs = db.read_two_view_geometries()
    win, cro = [], []
    for pid, tvg in zip(pair_ids, tvgs):
        n = len(tvg.inlier_matches)
        if n == 0:
            continue
        a, b = id2name[pid // KMAX], id2name[pid % KMAX]
        (win if labels[a] == labels[b] else cro).append(n)
    db.close()
    win, cro = np.array(win), np.array(cro)

    def stat(x, tag):
        if len(x) == 0:
            print(f'    {tag:<14s} n=0')
            return
        print(f'    {tag:<14s} n={len(x):5d}  min {x.min():4d}  p10 {np.percentile(x,10):5.0f}  '
              f'median {np.median(x):6.0f}  p90 {np.percentile(x,90):7.0f}  max {x.max():6d}')

    print('  verified-geometry inliers:')
    stat(win, 'within-scene')
    stat(cro, 'CROSS-scene')
    if len(cro):
        print('  threshold sweep:')
        for t in [20, 40, 60, 84, 100, 150, 200, 300]:
            w, c = int((win >= t).sum()), int((cro >= t).sum())
            print(f'    >={t:<5d} within {w:6d}   cross {c:5d}   '
                  f'contamination {100*c/max(w+c,1):5.1f}%')
    return win, cro


# ----------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fetch', choices=['eth3d'])
    p.add_argument('--scenes', nargs='+', default=['courtyard', 'terrace', 'pipes'])
    p.add_argument('--experiment', choices=sorted(EXPERIMENTS))
    p.add_argument('--pipeline', default='learned',
                   choices=['learned', 'colmap-default', 'colmap-shortlist'],
                   help='learned = DINOv2 + ALIKED + LightGlue; the colmap-* modes are '
                        'the SIFT + nearest-neighbour baseline, exhaustive or on the '
                        'same shortlist')
    p.add_argument('--check-poses', action='store_true',
                   help='verify the Mega-NeRF pose convention and exit')
    p.add_argument('--uav-start', type=int, default=0,
                   help='first frame index of the consecutive aerial window')
    p.add_argument('--min-inliers', type=int, default=None,
                   help='post-verification inlier filter; re-runs mapping only')
    # the pipeline's original hyper-parameters, unchanged by default
    p.add_argument('--sim-th', type=float, default=0.3)
    p.add_argument('--min-pairs', type=int, default=58)
    p.add_argument('--exhaustive-if-less', type=int, default=22)
    p.add_argument('--num-features', type=int, default=4600)
    p.add_argument('--resize-to', type=int, default=1024)
    p.add_argument('--min-matches', type=int, default=20)
    p.add_argument('--min-model-size', type=int, default=3)
    p.add_argument('--max-num-models', type=int, default=25,
                   help='default is 25; lower it for mixed-scene runs')
    args = p.parse_args()

    if args.check_poses:
        print('=== Mega-NeRF pose convention check ===')
        for site in ('building', 'rubble'):
            check_poses(UAV, site)
        return

    if args.fetch == 'eth3d':
        fetch_eth3d(args.scenes)
        if not args.experiment:
            return

    if not args.experiment:
        p.error('nothing to do: pass --experiment (or --fetch)')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('WARNING: no CUDA device; this will be extremely slow', file=sys.stderr)

    cfg = EXPERIMENTS[args.experiment]
    parts = cfg.get('scenes') or cfg['uav']
    print(f'=== {args.experiment}: {" + ".join(parts)} [{args.pipeline}] ===', flush=True)
    if 'uav' in cfg:
        images_dir, gt, labels = build_uav_dataset(
            WORK, UAV, args.experiment, cfg['uav'], cfg['n_per_site'], args.uav_start)
    else:
        images_dir, gt, labels = build_dataset(args.experiment, cfg['scenes'])
    print(f'  {len(labels)} images, {len(gt)} with ground truth', flush=True)

    r = run_dataset(args.experiment, images_dir, device, args, labels)
    print(f'  -> registered {len(r["preds"])}/{len(r["images"])} '
          f'in {r["n_clusters"]} clusters', flush=True)

    if args.min_inliers:
        print(f'\n--- ablation: drop geometries below {args.min_inliers} inliers ---')
        preds, clusters, n_models, T = refilter_and_map(
            args.experiment, r['db_path'], images_dir, args.min_inliers, args)
        print(f'  -> registered {len(preds)}/{len(r["images"])} in {n_models} clusters')
        r = dict(r, preds=preds, clusters=clusters, n_clusters=n_models)

    print()
    show_clusters(r['clusters'], labels, cfg.get('expect_clusters'))

    print()
    if cfg.get('same_place'):
        print('  NOTE: these two sets are the same physical scene photographed twice,')
        print('        so ground truth lives in two independent coordinate frames and')
        print('        only within-session pose error is meaningful.')
    for s in sorted(set(labels.values())):
        m = eval_scene(r['preds'], r['clusters'], gt, labels, s)
        a = m['auc']
        # ETH3D ground truth is metrically scaled from laser scans, so 'm' is real
        # metres there. Mill 19 ships no scale factor, so 'u' marks Mega-NeRF's
        # normalised units -- comparable within a run, not across datasets.
        unit = cfg.get('units', 'm')
        line = (f'  {s:<12s} AUC@5/10/20 = {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}'
                f'   | abs {m["abs_pos"]:.3f} {unit}, {m["abs_rot"]:.3f} deg'
                f'  ({m["frac"]*100:.0f}% in dominant reconstruction)')
        if m['flipped']:
            line += f'  [{m["flipped"]} cameras >170 deg off -- degenerate]'
        print(line)

    if len(set(labels.values())) > 1:
        print()
        link_diagnostics(r['feature_dir'], r['db_path'], labels)

    print('\n  timings (s):', {k: round(v, 1) for k, v in r['timings'].items()})


if __name__ == '__main__':
    main()
