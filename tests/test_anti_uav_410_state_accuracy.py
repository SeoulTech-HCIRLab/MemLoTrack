import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from trackit.data.components.result_collector.handler.one_pass_evaluation_compatible import (
    _expand_to_full_xyxy_time_conf,
)
from trackit.data.components.result_collector.state_accuracy import (
    encode_antiuav410_state_accuracy_absence,
)
from trackit.data.components.result_collector.handler.utils.compatibility import (
    ExternalToolkitCompatibilityHelper,
)
from trackit.data.protocol import SequenceInfo
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from trackit.datasets.SOT.datasets.Anti_UAV_410 import Anti_UAV_410_Seed


ANTI_UAV_TOOLKIT_PATH = Path(__file__).resolve().parents[1] / 'Anti-UAV410'
sys.path.insert(0, str(ANTI_UAV_TOOLKIT_PATH))
from Evaluation_for_SA import evaluate_directory  # noqa: E402
from utils.state_accuracy import evaluate_state_accuracy  # noqa: E402


class _FrameConstructor:
    def __init__(self, frame):
        self.frame = frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def set_path(self, path, size):
        self.frame['path'] = path
        self.frame['size'] = size

    def set_bounding_box(self, bounding_box, validity):
        self.frame['bounding_box'] = bounding_box
        self.frame['validity'] = validity


class _SequenceConstructor:
    def __init__(self, sequence):
        self.sequence = sequence

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def set_name(self, name):
        self.sequence['name'] = name

    def new_frame(self):
        frame = {}
        self.sequence['frames'].append(frame)
        return _FrameConstructor(frame)


class _DatasetConstructor:
    def __init__(self):
        self.sequences = []
        self.total_sequences = None

    def set_bounding_box_format(self, value):
        self.bounding_box_format = value

    def set_bounding_box_coordinate_system(self, value):
        self.bounding_box_coordinate_system = value

    def set_category_id_name_map(self, value):
        self.category_id_name_map = value

    def set_total_number_of_sequences(self, value):
        self.total_sequences = value

    def new_sequence(self, category_id):
        sequence = {'category_id': category_id, 'frames': []}
        self.sequences.append(sequence)
        return _SequenceConstructor(sequence)


class AntiUAV410SeedTest(unittest.TestCase):
    def _create_sequence(self, root: Path, frame_indices=(1, 2, 3)) -> Path:
        sequence_path = root / 'test' / 'IR_001'
        sequence_path.mkdir(parents=True)
        labels = {
            'exist': [1, 0, 1],
            'gt_rect': [[1, 2, 3, 4], [0, 0, 0, 0], [5, 6, 2, 2]],
        }
        (sequence_path / 'IR_label.json').write_text(json.dumps(labels), encoding='utf-8')
        for frame_index in frame_indices:
            Image.new('RGB', (16, 12)).save(sequence_path / f'{frame_index:06d}.jpg')
        return sequence_path

    def test_state_accuracy_mode_keeps_all_frames_and_raw_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_sequence(root)
            seed = Anti_UAV_410_Seed(
                root_path=str(root),
                data_split='test',
                evaluation_mode='state_accuracy',
            )
            constructor = _DatasetConstructor()

            seed.construct(constructor)

            self.assertEqual(seed.version, 4)
            self.assertEqual(constructor.sequences[0]['name'], 'IR_001')
            frames = constructor.sequences[0]['frames']
            self.assertEqual(len(frames), 3)
            self.assertEqual([frame['validity'] for frame in frames], [True, False, True])
            self.assertEqual(frames[1]['bounding_box'], [0.0, 0.0, 0.0, 0.0])

    def test_standard_mode_keeps_only_target_present_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_sequence(root)
            seed = Anti_UAV_410_Seed(root_path=str(root), data_split='test')
            constructor = _DatasetConstructor()

            seed.construct(constructor)

            self.assertEqual(seed.version, 1)
            self.assertEqual(constructor.sequences[0]['name'], 'Anti_UAV_410_IR_001')
            self.assertEqual(len(constructor.sequences[0]['frames']), 2)

    def test_state_accuracy_mode_rejects_frame_gaps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_sequence(root, frame_indices=(1, 3))
            seed = Anti_UAV_410_Seed(
                root_path=str(root),
                data_split='test',
                evaluation_mode='state_accuracy',
            )

            with self.assertRaisesRegex(ValueError, 'contiguous'):
                seed.construct(_DatasetConstructor())

    def test_state_accuracy_mode_rejects_annotation_length_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_sequence(root, frame_indices=(1, 2, 3, 4))
            seed = Anti_UAV_410_Seed(
                root_path=str(root),
                data_split='test',
                evaluation_mode='state_accuracy',
            )

            with self.assertRaisesRegex(ValueError, 'length mismatch'):
                seed.construct(_DatasetConstructor())

    def test_invalid_evaluation_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, 'evaluation_mode'):
                Anti_UAV_410_Seed(
                    root_path=temp_dir,
                    data_split='test',
                    evaluation_mode='unknown',
                )


class StateAccuracyMetricTest(unittest.TestCase):
    def test_evaluation_script_uses_dataset_split_and_writes_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sequence_path = root / 'dataset' / 'test' / 'IR_001'
            prediction_path = root / 'predictions'
            sequence_path.mkdir(parents=True)
            prediction_path.mkdir()
            labels = {
                'exist': [1, 0],
                'gt_rect': [[10, 10, 5, 5], [0, 0, 0, 0]],
            }
            (sequence_path / 'IR_label.json').write_text(
                json.dumps(labels), encoding='utf-8'
            )
            (prediction_path / 'IR_001.txt').write_text(
                '10 10 5 5\n0 0 0 0\n', encoding='utf-8'
            )
            report_path = root / 'reports' / 'sa.txt'

            score = evaluate_directory(
                root / 'dataset', prediction_path, 'test', 1, report_path
            )

            self.assertEqual(score, 1.0)
            self.assertIn('[Overall]', report_path.read_text(encoding='utf-8'))

    def test_zero_box_encodes_absence(self):
        labels = {
            'exist': [1, 0],
            'gt_rect': [[10, 10, 5, 5], [0, 0, 0, 0]],
        }
        predictions = [[10, 10, 5, 5], [0, 0, 0, 0]]
        self.assertEqual(evaluate_state_accuracy(predictions, labels), 1.0)

    def test_prediction_length_mismatch_is_rejected(self):
        labels = {
            'exist': [1, 0],
            'gt_rect': [[10, 10, 5, 5], [0, 0, 0, 0]],
        }
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            evaluate_state_accuracy([[10, 10, 5, 5]], labels)

    def _make_evaluation_result(self, sequence_name='IR_001'):
        result = SequenceEvaluationResult_SOT(
            id=1,
            sequence_info=SequenceInfo(
                'Anti_UAV_410', ('test',), 'Anti_UAV_410-test', sequence_name, 2, None
            ),
            evaluated_frame_indices=np.array([0, 1]),
            groundtruth_box=np.array([[0, 0, 5, 5], [np.nan] * 4]),
            groundtruth_object_existence_flag=np.array([True, False]),
            groundtruth_mask=None,
            output_box=np.array([[0, 0, 5, 5], [20, 20, 25, 25]], dtype=float),
            output_confidence=np.array([1.0, 0.5]),
            output_mask=None,
            time_cost=np.array([0.1, 0.1]),
            batch_size=np.array([1, 1]),
        )
        return result

    def test_state_accuracy_output_encodes_groundtruth_absence(self):
        result = self._make_evaluation_result()
        encoded_result = encode_antiuav410_state_accuracy_absence(result)

        _, aligned_boxes, _, _, _ = _expand_to_full_xyxy_time_conf(
            encoded_result, encoded_result.output_box
        )

        np.testing.assert_array_equal(aligned_boxes[1], [0, 0, 0, 0])

    def test_standard_output_does_not_encode_groundtruth_absence(self):
        result = self._make_evaluation_result('Anti_UAV_410_IR_001')
        encoded_result = encode_antiuav410_state_accuracy_absence(result)
        np.testing.assert_array_equal(
            encoded_result.output_box[1], [20, 20, 25, 25]
        )

    def test_state_accuracy_output_rejects_alignment_mismatch(self):
        result = self._make_evaluation_result()._replace(
            groundtruth_object_existence_flag=np.array([True])
        )
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            encode_antiuav410_state_accuracy_absence(result)

    def test_antiuav_sequence_name_is_not_prefixed(self):
        helper = ExternalToolkitCompatibilityHelper()
        name, _ = helper.adjust_for_pytracking(
            'Anti_UAV_410', 'IR_001', np.zeros((1, 4), dtype=float)
        )
        self.assertEqual(name, 'IR_001')

    def test_standard_antiuav_sequence_name_keeps_legacy_prefix(self):
        helper = ExternalToolkitCompatibilityHelper()
        name, _ = helper.adjust_for_pytracking(
            'Anti_UAV_410',
            'Anti_UAV_410_IR_001',
            np.zeros((1, 4), dtype=float),
        )
        self.assertEqual(name, 'uav_Anti_UAV_410_IR_001')


if __name__ == '__main__':
    unittest.main()
