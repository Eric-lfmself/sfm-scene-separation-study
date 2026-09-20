"""CPU-only regression tests; no weights, downloads or GPU dependencies."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from baseline_colmap import summarize_matches, write_pair_list
from dataset_utils import sync_image_links
from run_experiments import eval_scene, show_clusters
from run_report import json_ready, write_run_report
from sfm_pipeline import (absolute_pose_errors, camera_center, pairs_from_distances,
                          pose_auc, quat_to_rotmat, read_colmap_images_txt,
                          relative_pose_errors, umeyama)
from uav_dataset import FLIP, build_uav_dataset, check_poses, convert_c2w


class EvaluationTests(unittest.TestCase):
    def test_failed_pairs_stay_in_pooled_denominator(self):
        result = summarize_matches({1: 100, 2: 100}, {1: 80})
        self.assertEqual(result['n_putative'], 200)
        self.assertEqual(result['n_inliers'], 80)
        self.assertEqual(result['inlier_ratio'], .4)
        self.assertEqual(result['inlier_ratio_verified_pairs'], .8)
        self.assertEqual(result['n_putative_verified_pairs'], 100)
        self.assertEqual(result['median_pair_inlier_ratio'], .4)

    def test_scene_precision_uses_physical_place(self):
        result = summarize_matches({1: 20, 2: 20}, {1: 10, 2: 10},
                                   {1: ('room', 'room'), 2: ('room', 'other')})
        self.assertEqual(result['pair_precision'], .5)

    def test_inliers_cannot_exceed_matches(self):
        with self.assertRaises(ValueError):
            summarize_matches({1: 10}, {1: 20})

    def test_auc_perfect_failed_and_empty(self):
        self.assertEqual(pose_auc([0, 0]), [1., 1., 1.])
        self.assertEqual(pose_auc([180, 180]), [0., 0., 0.])
        self.assertTrue(all(math.isnan(v) for v in pose_auc([])))
        self.assertEqual(pose_auc([0, 180], (5,)), [.5])
        with self.assertRaises(ValueError):
            pose_auc([float('nan')])
        with self.assertRaises(ValueError):
            pose_auc([0], (0,))

    def test_missing_and_split_poses_are_failures(self):
        gt = {str(i): (np.eye(3), np.array([i, 0., 0.])) for i in range(3)}
        np.testing.assert_array_equal(relative_pose_errors({'0': gt['0']}, gt), [180] * 3)
        errs = relative_pose_errors(gt, gt, clusters={'0': 0, '1': 0, '2': 1})
        np.testing.assert_allclose(errs, [0, 180, 180])

    def test_sim3_alignment_recovers_pose_gauge(self):
        centres = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = np.array([4., -2., 3.])
        target = 2.5 * (rotation @ centres.T).T + translation
        scale, R, t = umeyama(centres, target)
        self.assertAlmostEqual(scale, 2.5)
        np.testing.assert_allclose(R, rotation, atol=1e-12)
        np.testing.assert_allclose(t, translation, atol=1e-12)
        pred = {str(i): (np.eye(3), -c) for i, c in enumerate(centres)}
        gt = {str(i): (rotation.T, -rotation.T @ c) for i, c in enumerate(target)}
        pos, rot, scale = absolute_pose_errors(pred, gt, list(pred))
        np.testing.assert_allclose(pos, 0, atol=1e-12)
        np.testing.assert_allclose(rot, 0, atol=2e-6)

    def test_collinear_alignment_is_unavailable(self):
        centres = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        with self.assertRaises(ValueError):
            umeyama(centres, centres)
        poses = {str(i): (np.eye(3), -c) for i, c in enumerate(centres)}
        pos, rot, scale = absolute_pose_errors(poses, poses, list(poses))
        self.assertEqual(len(pos), 0)
        self.assertTrue(math.isnan(scale))

    def test_degenerate_scene_does_not_report_measured_zero_large_rotations(self):
        centres = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        poses = {str(i): (np.eye(3), -c) for i, c in enumerate(centres)}
        labels = {name: 'scene' for name in poses}
        clusters = {name: 0 for name in poses}
        metrics = eval_scene(poses, clusters, poses, labels, 'scene')
        self.assertEqual(metrics['frac'], 1.0)
        self.assertIsNone(metrics['flipped'])
        self.assertEqual(metrics['n_abs_evaluated'], 0)
        self.assertIsNone(json_ready(metrics)['abs_rot'])
        centres[2] = [0., 1., 0.]
        poses = {str(i): (np.eye(3), -c) for i, c in enumerate(centres)}
        metrics = eval_scene(poses, clusters, poses, labels, 'scene')
        self.assertEqual(metrics['flipped'], 0)
        self.assertEqual(metrics['n_abs_evaluated'], 3)

    def test_empty_cluster_purity_unavailable(self):
        self.assertTrue(math.isnan(show_clusters({}, {})))

    def test_session_evaluation_denominator_includes_unregistered(self):
        gt = {str(i): (np.eye(3), np.array([i, 0., 0.])) for i in range(3)}
        m = eval_scene({'0': gt['0']}, {'0': 0}, gt, {str(i): 'one' for i in range(3)}, 'one')
        self.assertEqual(m['n_gt'], 3)
        self.assertEqual(m['n_registered'], 1)
        self.assertEqual(m['n_pairs'], 3)
        self.assertAlmostEqual(m['frac'], 1/3)


class ParsingAndShortlistTests(unittest.TestCase):
    def test_colmap_blank_point_line_does_not_skip_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'images.txt'
            path.write_text('# image records\n1 1 0 0 0 1 2 3 1 a.jpg\n\n'
                            '2 1 0 0 0 4 5 6 2 folder/b name.jpg\n0 0 -1\n')
            poses = read_colmap_images_txt(path)
            self.assertEqual(set(poses), {'a.jpg', 'b name.jpg'})
            np.testing.assert_allclose(poses['b name.jpg'][1], [4, 5, 6])

    def test_colmap_duplicate_basename_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'images.txt'
            path.write_text('1 1 0 0 0 0 0 0 1 x/a.jpg\n\n2 1 0 0 0 0 0 0 2 y/a.jpg\n\n')
            with self.assertRaises(ValueError):
                read_colmap_images_txt(path)

    def test_invalid_quaternion_rejected(self):
        with self.assertRaises(ValueError):
            quat_to_rotmat([0, 0, 0, 0])

    def test_corrected_shortlist_excludes_self_from_minimum(self):
        dm = np.array([[0., .1, 1.], [.1, 0., 2.], [1., 2., 0.]])
        self.assertEqual(pairs_from_distances(dm, 0, 1), [(0, 1), (0, 2)])
        self.assertEqual(pairs_from_distances(dm, 0, 1, 'legacy'), [])
        self.assertEqual(pairs_from_distances(dm, 0, 10), [(0, 1), (0, 2), (1, 2)])

    def test_legacy_matches_original_algorithm(self):
        rng = np.random.default_rng(123)
        values = rng.random((8, 4))
        dm = np.linalg.norm(values[:, None] - values[None, :], axis=2)
        pairs = []
        for i in range(len(dm) - 1):
            neighbours = np.arange(len(dm))[dm[i] <= .5]
            if len(neighbours) < 4:
                neighbours = np.argsort(dm[i])[:4]
            for j in neighbours:
                if i != j and dm[i, j] < 1000:
                    pairs.append(tuple(sorted((i, int(j)))))
        self.assertEqual(pairs_from_distances(dm, .5, 4, 'legacy'), sorted(set(pairs)))

    def test_invalid_distances_rejected(self):
        with self.assertRaises(ValueError):
            pairs_from_distances(np.array([[float('nan')]]))

    def test_pair_file_rejects_ambiguous_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_pair_list(['a b.jpg', 'c.jpg'], [(0, 1)], Path(tmp) / 'pairs.txt')


class DatasetTests(unittest.TestCase):
    def test_sync_removes_only_stale_links_and_fixes_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            a, b = base / 'a.jpg', base / 'b.jpg'
            a.write_bytes(b'a'); b.write_bytes(b'b')
            folder = base / 'images'
            sync_image_links(folder, {'first.jpg': a})
            sync_image_links(folder, {'second.jpg': b})
            self.assertEqual([p.name for p in folder.iterdir()], ['second.jpg'])
            sync_image_links(folder, {'second.jpg': a})
            self.assertEqual((folder / 'second.jpg').resolve(), a.resolve())
            self.assertTrue(b.exists())

    def test_sync_preserves_regular_files_and_validates_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / 'source.jpg'; source.write_bytes(b'source')
            folder = base / 'images'; folder.mkdir()
            protected = folder / 'user.jpg'; protected.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                sync_image_links(folder, {'source.jpg': source})
            self.assertEqual(protected.read_bytes(), b'keep')
            self.assertFalse((folder / 'source.jpg').exists())

    def test_c2w_conversion_preserves_centre_and_flips_axes(self):
        centre = np.array([1., 2., 3.])
        R, t = convert_c2w(np.column_stack((np.eye(3), centre)))
        np.testing.assert_allclose(R, FLIP)
        np.testing.assert_allclose(camera_center((R, t)), centre)
        with self.assertRaises(ValueError):
            convert_c2w(np.zeros((3, 4)))

    def test_changing_uav_window_does_not_accumulate_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rgb = base / 'data/building-pixsfm/train/rgbs'; rgb.mkdir(parents=True)
            for i in range(4):
                (rgb / f'{i:03}.jpg').write_bytes(b'image')
            with patch('uav_dataset.read_pose', return_value=(np.eye(3), np.zeros(3))):
                folder, _, _ = build_uav_dataset(base / 'work', base / 'data', 'test', ['building'], 2)
                folder, _, labels = build_uav_dataset(base / 'work', base / 'data', 'test', ['building'], 2, start=2)
            self.assertEqual(set(labels), {'002.jpg', '003.jpg'})
            self.assertEqual({p.name for p in Path(folder).iterdir()}, set(labels))

    def test_pose_check_rejects_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                check_poses(tmp)


class ReportingAndCliTests(unittest.TestCase):
    def test_strict_json_nulls_for_unavailable_values(self):
        cleaned = json_ready({'values': np.array([np.nan, np.inf, 1.]), 'n': np.int64(2)})
        self.assertEqual(json.loads(json.dumps(cleaned, allow_nan=False)), {'values': [None, None, 1.], 'n': 2})

    def test_report_contains_hashes_config_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); images = root / 'images'; images.mkdir()
            (images / 'a.jpg').write_bytes(b'test')
            args = argparse.Namespace(run_dir=tmp, images_dir=str(images), seed=42)
            labels = {'a.jpg': 'scene'}
            path = write_run_report(args, 'cpu', labels, labels, {}, status='started')
            path = write_run_report(args, 'cpu', labels, labels, {}, status='completed', metric=np.nan)
            report = json.loads(Path(path).read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['configuration']['seed'], 42)
            self.assertEqual(len(report['inputs']['a.jpg']['sha256']), 64)
            self.assertIsNone(report['metric'])
            self.assertIn('src/sfm_pipeline.py', report['source']['sha256'])

    def test_help_needs_no_third_party_packages(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'src/run_experiments.py'), '--help'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--shortlist-policy', result.stdout)

    def test_invalid_arguments_fail_before_dependency_import(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'src/run_experiments.py'),
                                 '--experiment', 'pipes', '--min-matches', '0'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('must be positive', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)


if __name__ == '__main__':
    unittest.main()
