from numbers import Real
from typing import Mapping, Sequence

import numpy as np


def prediction_is_absent(prediction) -> bool:
    """Return whether a prediction encodes the target-absent state.

    The official Anti-UAV format uses an empty row or a one-value row for
    absence. Plain numeric text files cannot represent variable-width rows
    reliably, so this implementation also accepts an invalid XYWH box (zero
    or negative width/height), including ``0 0 0 0``.
    """
    if prediction is None:
        return True

    try:
        values = list(prediction)
    except TypeError:
        values = [prediction]

    if len(values) in (0, 1):
        return True
    if len(values) != 4:
        return False
    if not all(isinstance(value, Real) for value in values):
        return False

    _, _, width, height = values
    return not np.isfinite(values).all() or width <= 0 or height <= 0


def intersection_over_union_xywh(bbox1, bbox2) -> float:
    bbox1 = np.asarray(bbox1, dtype=float)
    bbox2 = np.asarray(bbox2, dtype=float)
    if bbox1.shape != (4,) or bbox2.shape != (4,):
        return 0.0

    x1, y1, width1, height1 = bbox1
    x2, y2, width2, height2 = bbox2
    if width1 <= 0 or height1 <= 0 or width2 <= 0 or height2 <= 0:
        return 0.0

    intersection_left = max(x1, x2)
    intersection_top = max(y1, y2)
    intersection_right = min(x1 + width1, x2 + width2)
    intersection_bottom = min(y1 + height1, y2 + height2)

    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height
    union_area = width1 * height1 + width2 * height2 - intersection_area
    return float(intersection_area / union_area) if union_area > 0 else 0.0


def convert_xyxy_predictions_to_xywh(predictions: Sequence) -> list:
    converted = []
    for prediction in predictions:
        try:
            values = list(prediction)
        except TypeError:
            values = [prediction]
        if len(values) == 4:
            x1, y1, x2, y2 = values
            converted.append([x1, y1, x2 - x1, y2 - y1])
        else:
            converted.append(values)
    return converted


def evaluate_state_accuracy(
    predictions: Sequence,
    labels: Mapping,
    *,
    require_same_length: bool = True,
) -> float:
    groundtruth_rects = list(labels.get('gt_rect', ()))
    existence_flags = list(labels.get('exist', ()))
    predictions = list(predictions)

    if len(groundtruth_rects) != len(existence_flags):
        raise ValueError(
            'Anti-UAV410 gt_rect/exist length mismatch: '
            f'{len(groundtruth_rects)} != {len(existence_flags)}'
        )
    if require_same_length and len(predictions) != len(groundtruth_rects):
        raise ValueError(
            'Anti-UAV410 prediction/ground-truth length mismatch: '
            f'{len(predictions)} != {len(groundtruth_rects)}'
        )

    frame_scores = []
    for prediction, groundtruth, exists in zip(
        predictions, groundtruth_rects, existence_flags
    ):
        if not bool(exists):
            frame_scores.append(1.0 if prediction_is_absent(prediction) else 0.0)
            continue

        if not isinstance(groundtruth, (list, tuple)) or len(groundtruth) != 4:
            continue
        if not all(isinstance(value, Real) for value in groundtruth):
            continue
        if groundtruth[2] <= 0 or groundtruth[3] <= 0:
            continue

        if prediction_is_absent(prediction):
            frame_scores.append(0.0)
        else:
            frame_scores.append(intersection_over_union_xywh(prediction, groundtruth))

    if not frame_scores:
        raise ValueError('Anti-UAV410 sequence has no evaluable frames')
    return float(np.mean(frame_scores))
