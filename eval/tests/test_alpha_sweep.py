import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from eval import run_alpha_sweep as sweep
from eval.metrics.fid import FIDMetric
from eval.pipeline import evaluate_records
from eval.tests.mocks import MockFeatures


class AlphaSweepTests(unittest.TestCase):
    def test_only_selected_images_are_scored_from_larger_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt, pred = root/'gt', root/'pred'
            gt.mkdir(); pred.mkdir()
            for index in range(3):
                image = np.random.default_rng(index).integers(0, 256, (13+index, 19, 3), dtype=np.uint8)
                Image.fromarray(image).save(gt/f'image_{index}.png')
                Image.fromarray(255-image).save(pred/f'image_{index}.png')

            def evaluate_with_test_features(records, config, **kwargs):
                return evaluate_records(records, replace(config, device='cpu'),
                                        fid_metric=FIDMetric(network=MockFeatures(), dims=3), **kwargs)

            with patch.multiple(sweep, SOURCE=gt, OUTPUT=root/'out', LOCAL_OUTPUT=root/'local',
                                COUNT=2, SAMPLE_IDS=('image_0', 'image_1')), \
                 patch.object(sweep, 'evaluate_records', side_effect=evaluate_with_test_features):
                report = sweep.score(pred, 1.0)
            self.assertEqual(report['coverage']['ground_truth_images'], 3)
            self.assertEqual(report['coverage']['unevaluated_ground_truth_ids'], ['image_2'])
            self.assertEqual(report['fid']['real_count'], 2)
            self.assertEqual(report['fid']['generated_count'], 2)
            self.assertEqual([row['image_id'] for row in report['per_image']], ['image_0', 'image_1'])
            self.assertEqual({row['sample_id'] for row in report['per_sample']}, {'single'})


if __name__ == '__main__':
    unittest.main()
