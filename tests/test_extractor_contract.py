"""Regression checks at the optional learned-extractor boundary, without ML imports."""

from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sfm_pipeline import detect_aliked


class ExtractorContractTests(unittest.TestCase):
    def test_nondefault_resize_reaches_extraction_and_preserves_input_coordinates(self):
        # Pinned LightGlue v0.2 applies preprocessing kwargs in Extractor.extract;
        # constructor kwargs do not override ALIKED.preprocess_conf['resize'].
        keypoints = np.array([[123.5, 456.5]], dtype=np.float32)
        descriptors = np.ones((1, 128), dtype=np.float32)

        def tensor(array):
            value = MagicMock()
            value.detach.return_value = value
            value.cpu.return_value = value
            value.numpy.return_value = array
            return value

        kp = tensor(keypoints)
        kp.reshape.return_value = kp
        desc = MagicMock()
        desc.__getitem__.return_value = tensor(descriptors)
        extractor = MagicMock()
        extractor.eval.return_value = extractor
        extractor.to.return_value = extractor
        extractor.extract.return_value = {'keypoints': kp, 'descriptors': desc}
        aliked = MagicMock(return_value=extractor)
        image = MagicMock()
        image.to.return_value = image
        stored = {}

        def open_feature_file(path, mode):
            stored[Path(path).name] = {}
            return nullcontext(stored[Path(path).name])

        optional_modules = {
            'torch': SimpleNamespace(float32=object(), inference_mode=nullcontext,
                                     cuda=SimpleNamespace(empty_cache=lambda: None)),
            'lightglue': SimpleNamespace(ALIKED=aliked),
            'h5py': SimpleNamespace(File=open_feature_file),
            'tqdm': SimpleNamespace(tqdm=lambda items, **kwargs: items),
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sys.modules, optional_modules), \
             patch('sfm_pipeline.load_torch_image', return_value=image):
            detect_aliked(['input/photo.jpg'], tmp, num_features=256, resize_to=1600)

        extractor.extract.assert_called_once_with(image, resize=1600)
        self.assertNotIn('resize', aliked.call_args.kwargs)
        np.testing.assert_array_equal(stored['keypoints.h5']['photo.jpg'], keypoints)
        np.testing.assert_array_equal(stored['descriptors.h5']['photo.jpg'], descriptors)


if __name__ == '__main__':
    unittest.main()
