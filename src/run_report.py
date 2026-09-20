"""Persist evidence for new runs without altering the historical study tables."""

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = ('numpy', 'torch', 'torchvision', 'pycolmap', 'pycolmap-cuda12', 'kornia', 'kornia_rs',
            'transformers', 'lightglue', 'h5py', 'opencv-python', 'tqdm')


def json_ready(value):
    """Convert NumPy objects and unavailable numbers to strict JSON (null)."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if hasattr(value, 'tolist'):
        return json_ready(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_metadata(device):
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata = dict(python=sys.version, platform=platform.platform(),
                    packages=packages, device=str(device))
    torch = sys.modules.get('torch')
    if torch is not None:
        metadata['torch_cuda'] = torch.version.cuda
        if getattr(device, 'type', str(device)) == 'cuda':
            metadata['gpu'] = torch.cuda.get_device_name(device)
    pycolmap = sys.modules.get('pycolmap')
    if pycolmap is not None:
        has_cuda = getattr(pycolmap, 'has_cuda', None)
        metadata['pycolmap_has_cuda'] = bool(has_cuda() if callable(has_cuda) else has_cuda) if has_cuda is not None else None
    try:
        direct = importlib.metadata.distribution('lightglue').read_text('direct_url.json')
        metadata['lightglue_vcs'] = json.loads(direct).get('vcs_info') if direct else None
    except (importlib.metadata.PackageNotFoundError, ValueError):
        metadata['lightglue_vcs'] = None
    sfm = sys.modules.get('sfm_pipeline')
    if sfm is not None:
        metadata['dinov2_model'] = sfm.DINO_ID
        metadata['dinov2_resolved_revisions'] = {
            key: getattr(model.config, '_commit_hash', None)
            for key, (_, model) in sfm._DINO_CACHE.items()
        }
    metadata['determinism_note'] = 'Seeds are recorded; multithreaded and GPU execution may still vary.'
    return metadata


def source_metadata():
    root = Path(__file__).resolve().parent.parent
    out = {'sha256': {str(p.relative_to(root)): file_sha256(p)
                      for p in sorted((root / 'src').glob('*.py'))}}
    try:
        out['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        out['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        out['git_commit'] = None
        out['git_dirty'] = None
    return out


def write_run_report(args, device, session_labels, physical_labels, gt, status, **results):
    """Atomically save config, input hashes, poses, runtime and computed metrics."""
    path = Path(args.run_dir) / 'report.json'
    if path.exists():
        report = json.loads(path.read_text(encoding='utf-8'))
    else:
        image_dir = Path(args.images_dir)
        report = dict(
            schema_version=1,
            created_utc=datetime.now(timezone.utc).isoformat(),
            source=source_metadata(),
            inputs={name: dict(source=str((image_dir / name).resolve()),
                               sha256=file_sha256(image_dir / name),
                               session=session_labels[name], physical_scene=physical_labels[name])
                    for name in sorted(session_labels)},
            ground_truth_world_to_camera=gt,
            evaluation_notes=[
                'Current metrics include corrections; historical published rows have not been recomputed.',
                'Pooled inlier ratio includes all putative matches and is geometric consistency, not correspondence truth.',
                'Pose AUC includes missing registrations and split-reconstruction pairs as 180-degree failures.',
                'Absolute errors use each session\'s dominant model; degenerate Sim(3) alignment is unavailable.',
                'Learned uses independent SIMPLE_PINHOLE cameras with focal=1.2*max(width,height); SIFT uses COLMAP reader defaults.',
                'A matching cluster count alone does not establish correct scene separation.',
            ],
        )
    report.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
                  configuration=vars(args), runtime=runtime_metadata(device), **results)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(json_ready(report), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
    return str(path)
