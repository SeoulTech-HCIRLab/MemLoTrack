import numpy as np

from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT


def encode_antiuav410_state_accuracy_absence(
    evaluation_result: SequenceEvaluationResult_SOT,
) -> SequenceEvaluationResult_SOT:
    """Encode annotated absent frames as zero boxes for Anti-UAV410 SA output.

    State-accuracy sequences use the official raw sequence names. Standard OPE
    sequences keep the historical ``Anti_UAV_410_`` prefix and are left
    untouched. A length mismatch is an alignment error and must not be hidden
    by truncation or padding.
    """
    sequence_info = evaluation_result.sequence_info
    if sequence_info.dataset_name != 'Anti_UAV_410':
        return evaluation_result
    if sequence_info.sequence_name.startswith('Anti_UAV_410_'):
        return evaluation_result

    predicted_boxes = evaluation_result.output_box
    existence_flags = evaluation_result.groundtruth_object_existence_flag
    if predicted_boxes is None or existence_flags is None:
        return evaluation_result

    predicted_boxes = np.asarray(predicted_boxes)
    existence_flags = np.asarray(existence_flags, dtype=bool).reshape(-1)
    if len(predicted_boxes) != len(existence_flags):
        raise ValueError(
            'Anti-UAV410 SA prediction/existence length mismatch: '
            f'{len(predicted_boxes)} != {len(existence_flags)}'
        )

    encoded_boxes = predicted_boxes.copy()
    encoded_boxes[~existence_flags] = 0.0
    return evaluation_result._replace(output_box=encoded_boxes)
