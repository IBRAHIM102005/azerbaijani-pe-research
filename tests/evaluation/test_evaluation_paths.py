import tempfile
from pathlib import Path
import unittest
from scripts.discover_evaluation_paths import discover

class PathDiscoveryTests(unittest.TestCase):
    def test_server_layout_and_overlapping_search_roots(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);repo=root/'azerbaijani-pe-research';(repo/'scripts').mkdir(parents=True)
            (repo/'Evaluation.ipynb').touch();(repo/'scripts/evaluate.py').touch()
            exports=root/'m3/exports';exports.mkdir(parents=True)
            for n in ['5','10','20','50']:(exports/f'M3_CHECKPOINTS_{n}M.tar').touch()
            data=root/'inputs';data.mkdir();(data/'validation(3).parquet').touch();(data/'test(5).parquet').touch()
            loose=root/'m3/results/runs/run1/checkpoints/milestones';loose.mkdir(parents=True);(loose/'50m_model.pt').touch()
            result=discover([root,exports])
            self.assertEqual(result['package_roots'],[str(repo)])
            self.assertEqual(result['checkpoint_directories'],[str(exports)])
            self.assertEqual(result['loose_checkpoint_roots'],[str(root/'m3/results/runs')])
            self.assertEqual(len(result['archives']),4)
            self.assertEqual(result['parquet_directories'],[str(data)])

    def test_missing_search_root_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            result=discover([Path(d)/'absent'])
            self.assertTrue(result['warnings']);self.assertFalse(result['package_roots'])
