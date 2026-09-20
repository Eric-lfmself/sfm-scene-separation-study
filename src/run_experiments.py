"""
Experiment driver for the SfM matching pipeline study.

Fetches ETH3D scenes, assembles single-scene / mixed-scene / revisit datasets,
runs the pipeline, and reports registration, cluster purity, pose AUC, Sim(3)-aligned
absolute pose error, and the cross-scene link diagnostics.

    python run_experiments.py --fetch eth3d --scenes courtyard terrace pipes
    python run_experiments.py --experiment mixed3
    python run_experiments.py --experiment mixed3  --min-inliers 100
    python run_experiments.py --experiment revisit --min-inliers 100

CUDA is recommended for experiments; --help and --fetch do not need ML packages.
See ../docs/GETTING_STARTED.md for the environment and current metric changes.
"""

import argparse
import os
import glob
import shutil
import subprocess
import sys
import math
import random
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen
from collections import Counter, defaultdict
from time import time

from dataset_utils import sync_image_links

KMAX = 2147483647  # COLMAP's kMaxNumImages, used to decode pair ids

DATA = os.environ.get('SFM_DATA', 'data/eth3d')
WORK = os.environ.get('SFM_WORK', 'work')
UAV = os.environ.get('SFM_UAV', 'data/mill19')

# Scenes used in the study. relief and relief_2 are the SAME physical interior
# photographed in two sessions -- the revisit case, where one cluster is correct.
EXPERIMENTS = {
    'pipes':     dict(scenes=['pipes']),
    'terrace':   dict(scenes=['terrace']),
    'courtyard': dict(scenes=['courtyard']),
    'mixed3':    dict(scenes=['courtyard', 'terrace', 'pipes'], expect_clusters=3),
    'revisit':   dict(scenes=['relief', 'relief_2'], expect_clusters=1, same_place=True),
    # Mill 19 aerial. Consecutive frames from the TRAIN split -- see uav_dataset.py for
    # why sparse val frames are a poor overlap control for this SfM study.
    'uav_building': dict(uav=['building'], n_per_site=120, expect_clusters=1, units='u'),
    'uav_mixed':    dict(uav=['building', 'rubble'], n_per_site=120, expect_clusters=2, units='u'),
}


# ----------------------------------------------------------------- data

def fetch_eth3d(scenes):
    """Download public ETH3D archives; requires an existing 7z installation."""
    if not scenes or any(not re.fullmatch(r'[a-z0-9_]+', s) for s in scenes):
        raise ValueError('scene names must contain only lowercase letters, digits and underscores')
    extractor = shutil.which('7z') or shutil.which('7zz')
    if extractor is None:
        raise RuntimeError('install 7-Zip (7z or 7zz) before using --fetch')
    os.makedirs(DATA, exist_ok=True)
    for scene in scenes:
        if scene_images(scene):
            print(f'{scene}: already present')
            continue
        url = f'https://www.eth3d.net/data/{scene}_dslr_undistorted.7z'
        archive = Path(DATA) / f'{scene}.7z'
        partial = archive.with_suffix('.7z.part')
        print(f'{scene}: downloading {url}', flush=True)
        try:
            with urlopen(url, timeout=60) as source, open(partial, 'wb') as dest:
                shutil.copyfileobj(source, dest)
            partial.replace(archive)
            subprocess.run([extractor, 'x', '-y', f'-o{Path(DATA).resolve()}', str(archive)],
                           check=True, stdout=subprocess.DEVNULL)
            if not scene_images(scene):
                raise RuntimeError(f'{scene}: archive extracted but no expected images were found')
            archive.unlink()
        finally:
            partial.unlink(missing_ok=True)
        print(f'{scene}: {len(scene_images(scene))} images')


def scene_images(scene):
    return sorted(glob.glob(f'{DATA}/{scene}/images/dslr_images_undistorted/*.JPG'))


def build_dataset(name, scenes, output_dir=None):
    """Symlink scenes into one folder. Returns (images_dir, gt, labels).

    For a multi-scene set, filenames are prefixed with the scene so they stay unique
    and so every image carries its true scene label for the purity metric.
    """
    from sfm_pipeline import read_colmap_images_txt

    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError('provide at least one distinct ETH3D scene')
    d = output_dir or f'{WORK}/{name}/images'
    gt, labels, sources = {}, {}, {}
    multi = len(scenes) > 1
    for s in scenes:
        poses = read_colmap_images_txt(f'{DATA}/{s}/dslr_calibration_undistorted/images.txt')
        images = scene_images(s)
        if not images:
            raise FileNotFoundError(f'{s}: no images under {DATA}; fetch the scene first')
        for p in images:
            b = os.path.basename(p)
            key = f'{s}__{b}' if multi else b
            sources[key] = p
            labels[key] = s
            if b in poses:
                gt[key] = poses[b]
    sync_image_links(d, sources)
    return d, gt, labels


# ------------------------------------------------------------- pipeline

def cam_from_world(im):
    """pycolmap 4.x turned this into a method derived from the image's frame."""
    import numpy as np

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
    import pycolmap
    from baseline_colmap import (COLMAP_MAX_IMAGE_SIZE, COLMAP_MAX_NUM_FEATURES,
                                 matching_precision, run_colmap_baseline, write_pair_list, database_summary)
    from sfm_pipeline import (detect_aliked, get_image_pairs_shortlist,
                              import_into_colmap, match_with_lightglue)

    images = sorted(glob.glob(images_dir + '/*'))
    if len(images) < 2:
        raise ValueError('an SfM run needs at least two images')
    feature_dir = os.path.join(args.run_dir, 'features')
    os.makedirs(feature_dir, exist_ok=False)
    T = {}
    n_possible = len(images) * (len(images) - 1) // 2

    pairs = None
    if args.pipeline != 'colmap-default':
        t = time()
        pairs = get_image_pairs_shortlist(images, args.sim_th, args.min_pairs,
                                          args.exhaustive_if_less, device, args.shortlist_policy)
        T['shortlist'] = time() - t
        write_pair_list(images, pairs, os.path.join(args.run_dir, 'shortlist.txt'))
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
        # Verify the imported matches directly: no SIFT descriptor matching here.
        verification = pycolmap.TwoViewGeometryOptions()
        verification.ransac.random_seed = args.seed
        pycolmap.geometric_verification(db_path, two_view_geometry_options=verification)
        T['ransac'] = time() - t
        print(f'  ransac:    {T["ransac"]:.1f}s', flush=True)
    else:
        stock = args.pipeline == 'colmap-default'
        db_path, Tb = run_colmap_baseline(
            images_dir, feature_dir, images=images, pairs=pairs,
            max_image_size=COLMAP_MAX_IMAGE_SIZE if stock else args.resize_to,
            max_num_features=COLMAP_MAX_NUM_FEATURES if stock else args.num_features,
            gpu=device.type == 'cuda', seed=args.seed)
        T.update(Tb)
        db = pycolmap.Database.open(db_path)
        n_kept = db.num_matched_image_pairs()
        db.close()

    prec = matching_precision(db_path, labels)

    preds, clusters, n_models = map_and_collect(name, db_path, images_dir, args, T)
    T['total'] = sum(v for k, v in T.items() if k != 'total')
    print(f'  TOTAL:     {T["total"]:.1f}s', flush=True)
    return dict(name=name, images=images, n_pairs=len(pairs) if pairs is not None else n_possible,
                n_kept=n_kept, preds=preds, clusters=clusters, n_clusters=n_models,
                timings=T, precision=prec, feature_dir=feature_dir, db_path=db_path,
                database_summary=database_summary(db_path))


def map_and_collect(name, db_path, images_dir, args, T, tag=''):
    import pycolmap

    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = args.min_model_size
    opts.max_num_models = args.max_num_models
    opts.mapper.random_seed = args.seed
    opts.triangulation.random_seed = args.seed
    out = os.path.join(args.run_dir, f'reconstruction{tag}')
    os.makedirs(out, exist_ok=False)
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
    import pycolmap

    dst = str(Path(db_path).with_name(f'{Path(db_path).stem}_inl{min_inliers}.db'))
    # SQLite backup includes any pending WAL pages; copying only the main file does not.
    with sqlite3.connect(db_path) as source, sqlite3.connect(dst) as target:
        source.backup(target)
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
    return map_and_collect(name, dst, images_dir, args, T, tag=f'_inl{min_inliers}') + (T, dst)


# ------------------------------------------------------------ reporting

def show_clusters(clusters, labels, expect=None):
    comp = defaultdict(Counter)
    for n, c in clusters.items():
        comp[c][labels[n]] += 1
    print('  cluster composition:')
    for c in sorted(comp):
        print(f'    cluster {c:<3d} {dict(comp[c])}  size={sum(comp[c].values())}')
    total = sum(sum(v.values()) for v in comp.values())
    purity = sum(max(v.values()) for v in comp.values()) / total if total else float('nan')
    note = ''
    if expect is not None:
        note = '  (expected cluster count)' if len(comp) == expect else f'  (expected {expect})'
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
    import numpy as np
    from sfm_pipeline import relative_pose_errors, pose_auc, absolute_pose_errors

    names = sorted(n for n in gt if labels[n] == scene)
    errs = relative_pose_errors(preds, gt, names, clusters)
    auc = pose_auc(errs)

    reg = [n for n in names if n in preds]
    out = dict(auc=auc, n_pairs=len(errs), n_gt=len(names), n_registered=len(reg), abs_pos=float('nan'),
               abs_rot=float('nan'), frac=0.0, flipped=None, n_abs_evaluated=0)
    if reg:
        dominant = Counter(clusters[n] for n in reg).most_common(1)[0][0]
        sel = [n for n in reg if clusters[n] == dominant]
        out['frac'] = len(sel) / len(names)
        pos, rot, _ = absolute_pose_errors(preds, gt, sel)
        if len(pos) and len(rot) and np.isfinite(rot).all():
            out['n_abs_evaluated'] = len(rot)
            out['abs_pos'] = float(np.median(pos))
            out['abs_rot'] = float(np.median(rot))
            out['flipped'] = int((rot > 170).sum())   # full rotation error, including roll
    return out


def link_diagnostics(feature_dir, db_path, labels):
    """Where do the cross-scene links come from, and how strong are they?"""
    # Only the learned front end writes matches.h5; COLMAP's matcher goes straight to
    # the database. The database half below works for either, and is the important half.
    import numpy as np
    import pycolmap

    h5 = f'{feature_dir}/matches.h5'
    if os.path.exists(h5):
        import h5py

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
    global DATA, WORK, UAV

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default=DATA, help='ETH3D root; also SFM_DATA')
    p.add_argument('--work-root', default=WORK, help='output root; also SFM_WORK')
    p.add_argument('--uav-root', default=UAV, help='Mill 19 root; also SFM_UAV')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--seed', type=int, default=0, help='recorded seed; does not guarantee bitwise reproducibility')
    p.add_argument('--shortlist-policy', choices=['corrected', 'legacy'], default='corrected',
                   help='legacy retains historical self-counting and omitted-last-query behavior')
    p.add_argument('--pose-check-count', type=int, default=120)
    p.add_argument('--fetch', choices=['eth3d'])
    p.add_argument('--scenes', nargs='+', default=['courtyard', 'terrace', 'pipes'])
    p.add_argument('--experiment', choices=sorted(EXPERIMENTS))
    p.add_argument('--pipeline', default='learned',
                   choices=['learned', 'colmap-default', 'colmap-shortlist'],
                   help='learned = DINOv2 + ALIKED + LightGlue; the colmap-* modes are '
                        'the SIFT + nearest-neighbour baseline, exhaustive or on the '
                        'same shortlist')
    p.add_argument('--check-poses', action='store_true',
                   help='report UAV pose continuity (not a proof of axis convention) and exit')
    p.add_argument('--uav-start', type=int, default=0,
                   help='first frame index of the consecutive aerial window')
    p.add_argument('--min-inliers', type=int, default=None,
                   help='post-verification inlier filter; re-runs mapping only')
    # Historical feature budgets; shortlist bookkeeping is selected separately.
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
    for key in ('num_features', 'resize_to', 'min_matches', 'min_model_size', 'max_num_models'):
        if getattr(args, key) <= 0:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if args.min_pairs < 0 or args.exhaustive_if_less < 0 or args.uav_start < 0 or args.seed < 0:
        p.error('pair counts, frame start and seed must be nonnegative')
    if args.seed > 2**32 - 1:
        p.error('--seed must fit in an unsigned 32-bit integer')
    if not math.isfinite(args.sim_th) or args.sim_th < 0:
        p.error('--sim-th must be finite and nonnegative')
    if args.min_inliers is not None and args.min_inliers <= 0:
        p.error('--min-inliers must be positive')
    if args.pose_check_count < 2:
        p.error('--pose-check-count must be at least 2')
    DATA, WORK, UAV = (str(Path(value).expanduser().resolve()) for value in
                       (args.data_root, args.work_root, args.uav_root))
    args.data_root, args.work_root, args.uav_root = DATA, WORK, UAV

    if args.check_poses:
        from uav_dataset import check_poses

        print('=== Mega-NeRF pose continuity check ===')
        for site in ('building', 'rubble'):
            check_poses(UAV, site, args.pose_check_count, args.uav_start)
        return

    if args.fetch == 'eth3d':
        fetch_eth3d(args.scenes)
        if not args.experiment:
            return

    if not args.experiment:
        p.error('nothing to do: pass --experiment (or --fetch)')

    import numpy as np
    import torch
    import pycolmap

    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('--device cuda requested but CUDA is unavailable')
    capability = getattr(pycolmap, 'has_cuda', False)
    colmap_cuda = bool(capability() if callable(capability) else capability)
    if args.device == 'cuda' and args.pipeline.startswith('colmap-') and not colmap_cuda:
        p.error('--device cuda requires a CUDA-enabled COLMAP build for SIFT; use --device cpu or install one')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pycolmap.set_random_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    if device.type != 'cuda':
        print('WARNING: no CUDA device; this will be extremely slow', file=sys.stderr)

    cfg = EXPERIMENTS[args.experiment]
    parts = cfg.get('scenes') or cfg['uav']
    print(f'=== {args.experiment}: {" + ".join(parts)} [{args.pipeline}] ===', flush=True)
    args.run_dir = str(Path(WORK) / args.experiment /
                       f'{args.pipeline}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')
    os.makedirs(args.run_dir, exist_ok=False)
    args.images_dir = os.path.join(args.run_dir, 'images')
    if 'uav' in cfg:
        from uav_dataset import build_uav_dataset

        images_dir, gt, labels = build_uav_dataset(
            WORK, UAV, args.experiment, cfg['uav'], cfg['n_per_site'], args.uav_start,
            output_dir=args.images_dir)
    else:
        images_dir, gt, labels = build_dataset(args.experiment, cfg['scenes'], output_dir=args.images_dir)
    print(f'  {len(labels)} images, {len(gt)} with ground truth', flush=True)

    physical_labels = {name: 'relief-interior' for name in labels} if cfg.get('same_place') else labels
    from run_report import write_run_report
    write_run_report(args, device, labels, physical_labels, gt, status='started')
    try:
        r = run_dataset(args.experiment, images_dir, device, args, physical_labels)
        base_result = dict(r)
        print(f'  -> registered {len(r["preds"])}/{len(r["images"])} '
              f'in {r["n_clusters"]} clusters', flush=True)

        if args.min_inliers is not None:
            print(f'\n--- ablation: drop geometries below {args.min_inliers} inliers ---')
            preds, clusters, n_models, T, filtered_db = refilter_and_map(
                args.experiment, r['db_path'], images_dir, args.min_inliers, args)
            print(f'  -> registered {len(preds)}/{len(r["images"])} in {n_models} clusters')
            from baseline_colmap import matching_precision

            timings = dict(r['timings'], ablation_mapping=T['mapping'])
            timings['total'] += T['mapping']
            r = dict(r, preds=preds, clusters=clusters, n_clusters=n_models, db_path=filtered_db,
                     timings=timings, precision=matching_precision(filtered_db, physical_labels))

        print()
        purity = show_clusters(r['clusters'], physical_labels, cfg.get('expect_clusters'))
        if cfg.get('same_place'):
            print('  capture-session composition (not distinct scene labels):')
            show_clusters(r['clusters'], labels)

        print()
        if cfg.get('same_place'):
            print('  NOTE: these two sets are the same physical scene photographed twice,')
            print('        so ground truth lives in two independent coordinate frames and')
            print('        only within-session pose error is meaningful.')
        scene_metrics = {}
        for s in sorted(set(labels.values())):
            m = eval_scene(r['preds'], r['clusters'], gt, labels, s)
            scene_metrics[s] = m
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

        diagnostics = None
        if len(set(physical_labels.values())) > 1:
            print()
            within, cross = link_diagnostics(r['feature_dir'], r['db_path'], physical_labels)
            diagnostics = dict(within_scene_inliers=within, cross_scene_inliers=cross)

        print('\n  timings (s):', {k: round(v, 1) for k, v in r['timings'].items()})
        report = write_run_report(args, device, labels, physical_labels, gt, status='completed',
                                  result=r, base_result=base_result if args.min_inliers else None,
                                  scene_metrics=scene_metrics, purity=purity, diagnostics=diagnostics)
        print(f'  saved report: {report}')
    except BaseException as exc:
        write_run_report(args, device, labels, physical_labels, gt, status='failed',
                         error=dict(type=type(exc).__name__, message=str(exc)))
        raise



if __name__ == '__main__':
    main()
