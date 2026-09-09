"""Regressions for shared-filesystem metadata disappearance during preparation."""
import errno,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.scene_token.run_training import preparation_state,read_preparation_json

class PreparationPollTests(unittest.TestCase):
    def test_missing_then_visible(self):
        with patch.object(Path,'read_text',side_effect=[FileNotFoundError(),'{"done": 4}']),patch('tools.scene_token.run_training.time.sleep'):
            self.assertEqual(read_preparation_json(Path('progress.json')),{'done':4})

    def test_missing_keeps_original_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmp,patch('tools.scene_token.run_training.time.sleep'):
            previous={'done':10,'updated':100.}
            summary,progress=preparation_state(Path(tmp),previous,90.,now=150.)
            self.assertIsNone(summary);self.assertEqual(progress,previous)
            with self.assertRaisesRegex(RuntimeError,'heartbeat stale'):
                preparation_state(Path(tmp),progress,90.,now=401.)

    def test_initial_absence_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp,patch('tools.scene_token.run_training.time.sleep'):
            self.assertEqual(preparation_state(Path(tmp),{},100.,now=200.),(None,{}))
            with self.assertRaisesRegex(RuntimeError,'unavailable'):
                preparation_state(Path(tmp),{},100.,now=401.)

    def test_json_replacement_retry_and_persistent_corruption(self):
        with patch.object(Path,'read_text',side_effect=['','{"done": 5}']),patch('tools.scene_token.run_training.time.sleep'):
            self.assertEqual(read_preparation_json(Path('progress.json')),{'done':5})
        with patch.object(Path,'read_text',return_value='bad'),patch('tools.scene_token.run_training.time.sleep'):
            with self.assertRaises(json.JSONDecodeError):read_preparation_json(Path('progress.json'))

    def test_transient_io_retries_but_permission_failure_does_not(self):
        with patch.object(Path,'read_text',side_effect=[OSError(errno.ESTALE,'stale'),'{"done": 6}']),patch('tools.scene_token.run_training.time.sleep'):
            self.assertEqual(read_preparation_json(Path('progress.json')),{'done':6})
        with patch.object(Path,'read_text',side_effect=PermissionError()) as read:
            with self.assertRaises(PermissionError):read_preparation_json(Path('progress.json'))
            self.assertEqual(read.call_count,1)

    def test_completed_or_failed_summary_preserved_without_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for passed in [True,False]:
                value={'passed':passed,'state':'ready' if passed else 'failed'}
                (root/'summary.json').write_text(json.dumps(value))
                summary,progress=preparation_state(root,{},100.,now=1000.)
                self.assertEqual(summary,value);self.assertEqual(progress,{})

if __name__=='__main__':unittest.main()
