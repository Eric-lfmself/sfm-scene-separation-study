"""Small real-database smoke tests; skipped when optional COLMAP/HDF5 are absent."""

import argparse
from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
AVAILABLE = find_spec('pycolmap') is not None and find_spec('h5py') is not None


@unittest.skipUnless(AVAILABLE, 'optional pycolmap and h5py are not installed')
class ColmapSmokeTests(unittest.TestCase):
    def setUp(self):
        import h5py
        import pycolmap
        from sfm_pipeline import import_into_colmap

        self.pycolmap = pycolmap
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / 'images'; self.images.mkdir()
        self.features = self.root / 'features'; self.features.mkdir()
        for name in ('a.png', 'b.png'):
            pycolmap.Bitmap.from_array(np.zeros((32, 32, 3), dtype=np.uint8)).write(str(self.images / name))
        with h5py.File(self.features / 'keypoints.h5', 'w') as handle:
            for name in ('a.png', 'b.png'):
                handle[name] = np.array([[3., 3.], [10., 10.], [20., 20.]], dtype=np.float32)
        with h5py.File(self.features / 'matches.h5', 'w') as handle:
            handle.require_group('a.png')['b.png'] = np.array([[0, 0], [1, 1], [2, 2]])
        self.database = str(self.root / 'colmap.db')
        self.ids, self.count = import_into_colmap(str(self.images), str(self.features), self.database)

    def test_import_rig_frame_schema_and_rejected_pair_denominator(self):
        from baseline_colmap import matching_precision, database_summary

        db = self.pycolmap.Database.open(self.database)
        self.assertEqual((db.num_images(), db.num_frames(), db.num_rigs()), (2, 2, 2))
        db.close()
        self.pycolmap.geometric_verification(self.database)
        precision = matching_precision(self.database, {'a.png': 'room', 'b.png': 'room'})
        self.assertEqual(precision['n_putative'], 3)
        self.assertEqual(precision['n_inliers'], 0)
        self.assertEqual(precision['inlier_ratio'], 0)
        actual = database_summary(self.database)
        self.assertEqual(actual['images']['a.png']['n_keypoints'], 3)
        self.assertEqual(actual['cameras'][1]['model'], 'SIMPLE_PINHOLE')

    def test_filter_copies_db_and_uses_filtered_path(self):
        from run_experiments import refilter_and_map

        db = self.pycolmap.Database.open(self.database)
        geometry = self.pycolmap.TwoViewGeometry()
        geometry.inlier_matches = np.array([[0, 0], [1, 1]], dtype=np.uint32)
        db.write_two_view_geometry(self.ids['a.png'], self.ids['b.png'], geometry)
        db.close()
        def fake_mapping(name, database, images, args, timing, tag=''):
            timing['mapping'] = 0.0
            return {}, {}, 0
        with patch('run_experiments.map_and_collect', side_effect=fake_mapping):
            _, _, _, timings, filtered = refilter_and_map('test', self.database, str(self.images), 3, None)
        original = self.pycolmap.Database.open(self.database)
        copied = self.pycolmap.Database.open(filtered)
        try:
            self.assertEqual(len(original.read_two_view_geometries()[0]), 1)
            self.assertEqual(len(copied.read_two_view_geometries()[0]), 0)
        finally:
            original.close(); copied.close()
        self.assertEqual(timings['mapping'], 0)

    def test_mapper_options_supported_by_pinned_api(self):
        from run_experiments import map_and_collect

        args = argparse.Namespace(run_dir=str(self.root), seed=11, min_model_size=3, max_num_models=25)
        with patch.object(self.pycolmap, 'incremental_mapping', return_value={}) as mapping:
            self.assertEqual(map_and_collect('test', self.database, str(self.images), args, {}), ({}, {}, 0))
        options = mapping.call_args.kwargs['options']
        self.assertEqual(options.mapper.random_seed, 11)
        self.assertEqual(options.triangulation.random_seed, 11)

    def test_empty_shortlist_does_not_invoke_matching(self):
        from baseline_colmap import run_colmap_baseline

        # Real CPU feature extraction, with both matchers spied to catch fall-through.
        with patch.object(self.pycolmap, 'match_exhaustive') as exhaustive, \
             patch.object(self.pycolmap, 'match_image_pairs') as imported:
            database, _ = run_colmap_baseline(str(self.images), str(self.root / 'sift'),
                                              images=[str(p) for p in self.images.iterdir()],
                                              pairs=[], gpu=False, max_image_size=32,
                                              max_num_features=16)
        exhaustive.assert_not_called(); imported.assert_not_called()
        self.assertTrue(Path(database).exists())
