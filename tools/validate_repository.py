#!/usr/bin/env python3
"""Validate local documentation, result provenance and figure files without ML packages.

This checks repository integrity, not the truth of the historical experiments.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ERRORS = []


def require(condition, message):
    if not condition:
        ERRORS.append(message)


def repository_files():
    # Include pending additions during a local review, excluding ignored run outputs.
    try:
        raw = subprocess.check_output(
            ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
            cwd=ROOT,
        )
        return sorted({ROOT / n.decode() for n in raw.split(b'\0') if n and (ROOT / n.decode()).is_file()})
    except (OSError, subprocess.CalledProcessError):
        excluded = {'.git', '.venv', 'venv', 'data', 'work', 'preview', '__pycache__'}
        return sorted(p for p in ROOT.rglob('*') if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts))


def check_links(path):
    # The immutable historical snapshot deliberately retains its original relative paths.
    if path.name == 'source-results-7e0d60c.md':
        return
    body = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
    targets = re.findall(r'\[[^\]]*\]\(([^)]+)\)', body)
    targets += re.findall(r'(?:src|href)=["\']([^"\']+)["\']', body)
    for raw in targets:
        target = raw.strip().strip('<>')
        parsed = urlsplit(target)
        if parsed.scheme or not parsed.path:
            continue
        dest = (path.parent / unquote(parsed.path)).resolve()
        require(dest.is_relative_to(ROOT), f'{path.relative_to(ROOT)}: link escapes repository: {target}')
        require(dest.exists(), f'{path.relative_to(ROOT)}: missing link: {target}')


def check_figures():
    manifest = json.loads((ROOT / 'figures/manifest.json').read_text())
    require(manifest['experiments_rerun'] is False, 'Historical figures must not claim a new experiment.')
    for name, expected in manifest['source_sha256'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        require(actual == expected, f'Figure data changed without a rebuild: {name}')
    for record in manifest['figures']:
        if 'generator_sha256' in record:
            actual = hashlib.sha256((ROOT / record['generator']).read_bytes()).hexdigest()
            require(actual == record['generator_sha256'],
                    f"Figure generator changed without a rebuild: {record['name']}")
            sidecar = ROOT / 'figures' / (record['name'] + '.manifest.json')
            require(json.loads(sidecar.read_text()) == record,
                    f"Figure manifest differs from sidecar: {record['name']}")
        for name in record['files']:
            p = ROOT / 'figures' / name
            require(p.is_file(), f'Missing figure {name}')
            if p.suffix == '.svg':
                node = ET.parse(p).getroot()
                ns = {'s': 'http://www.w3.org/2000/svg'}
                require(node.find('s:title', ns) is not None and node.find('s:desc', ns) is not None,
                        f'{name}: missing title/description')
                require(bool(node.findall('.//s:text', ns)), f'{name}: no editable SVG text')
                require(not node.findall('.//s:image', ns), f'{name}: unexpected embedded raster')
            elif p.suffix == '.png':
                data = p.read_bytes()
                require(data[:8] == b'\x89PNG\r\n\x1a\n', f'{name}: invalid PNG signature')
                width, height = struct.unpack('>II', data[16:24])
                require(width >= 1000 and height > 0, f'{name}: invalid preview dimensions')


def check_results():
    for path in (ROOT / 'results/reported').glob('*.csv'):
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        require(bool(rows), f'{path.name}: no data')
        for i, row in enumerate(rows, 2):
            require(None not in row, f'{path.name}:{i}: malformed CSV row')
            numeric = {}
            for key, val in row.items():
                if not val:
                    continue  # Missing is retained, never converted to zero.
                try:
                    numeric[key] = float(val)
                except ValueError:
                    continue
                require(math.isfinite(numeric[key]), f'{path.name}:{i}: nonfinite {key}')
                if key.startswith('auc') or key in {'purity', 'pair_precision', 'inlier_ratio'}:
                    require(0 <= numeric[key] <= 1, f'{path.name}:{i}: invalid fraction {key}')
            if {'registered', 'total'} <= numeric.keys():
                require(0 <= numeric['registered'] <= numeric['total'], f'{path.name}:{i}: invalid coverage')
            if {'auc5', 'auc10', 'auc20'} <= numeric.keys():
                require(numeric['auc5'] <= numeric['auc10'] <= numeric['auc20'], f'{path.name}:{i}: nonmonotone AUC')
            if {'min', 'p10', 'median', 'p90', 'max'} <= numeric.keys():
                vals = [numeric[k] for k in ['min', 'p10', 'median', 'p90', 'max']]
                require(vals == sorted(vals), f'{path.name}:{i}: unordered percentiles')
            if {'same_scene_verified_pairs', 'all_verified_pairs', 'pair_precision'} <= numeric.keys():
                ratio = numeric['same_scene_verified_pairs'] / numeric['all_verified_pairs']
                require(abs(ratio - numeric['pair_precision']) < 0.0001, f'{path.name}:{i}: pair-precision/count mismatch')

    # The eleven historical tables are retained; only explanatory labels may change.
    old = (ROOT / 'results/reported/source-results-7e0d60c.md').read_text()
    new = (ROOT / 'RESULTS.md').read_text()
    tables = lambda text: re.findall(r'(?m)^\|.*(?:\n\|.*)*', text)
    digits = lambda text: re.findall(r'\d+(?:\.\d+)?', text)
    available = [digits(t) for t in tables(new)]
    for i, table in enumerate(tables(old), 1):
        require(digits(table) in available, f'Historical table {i} numerical values changed or disappeared')


def main():
    files = repository_files()
    for p in files:
        rel = p.relative_to(ROOT)
        require(p.suffix.lower() not in {'.pdf', '.tex'} and 'paper' not in rel.parts,
                f'Manuscript excluded from this update: {rel}')
        if p.suffix == '.md':
            check_links(p)
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(rel))
    check_figures()
    check_results()
    if ERRORS:
        print('\n'.join('FAIL: ' + e for e in ERRORS))
        return 1
    print(f'PASS: {len(files)} files checked; local links, Python syntax, historical tables, CSVs and figure provenance.')
    print('Scope: offline repository checks only; no GPU experiments or external-link availability checks.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
