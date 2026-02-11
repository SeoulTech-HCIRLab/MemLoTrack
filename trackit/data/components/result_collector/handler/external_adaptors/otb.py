from typing import Optional, Sequence
import zipfile
import io
import os
import numpy as np

from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh

from ...progress_tracer import EvaluationProgress
from .. import EvaluationResultHandler
from ..utils.compatibility import ExternalToolkitCompatibilityHelper


class OTBEvaluationToolTrackingResultWriter:
    def __init__(self, output_folder: str, file_name: str):
        self._zip_file = zipfile.ZipFile(os.path.join(output_folder, file_name + '.zip'), 'w', zipfile.ZIP_DEFLATED)
        self._duplication_check = set()

    def write(self, tracker_name: str, repeat_index: Optional[int], sequence_name: str, predicted_bboxes_xywh: np.ndarray, expected_len: int):
        assert (tracker_name, repeat_index, sequence_name) not in self._duplication_check
        self._duplication_check.add((tracker_name, repeat_index, sequence_name))
        tracker_folder_path = tracker_name
        if repeat_index is not None:
            tracker_folder_path += f'_{repeat_index:03d}_tracking_result'
        else:
            tracker_folder_path += '_tracking_result'
        with io.StringIO() as f:
            # 공백 구분(괄호/콤마 없음)
            np.savetxt(f, predicted_bboxes_xywh.astype(np.float32), fmt='%.3f')
            print(f"[DBG] OTB write: {sequence_name}.txt lines={predicted_bboxes.shape[0]}", flush=True)


            self._zip_file.writestr(f'{tracker_folder_path}/{sequence_name}.txt', f.getvalue())
        # 디버그
        print(f"[DUMMY-ENFORCER][OTB] wrote {sequence_name}.txt: lines={predicted_bboxes_xywh.shape[0]} expected={expected_len}", flush=True)
        assert predicted_bboxes_xywh.shape[0] == expected_len, f"[OTB] line-count mismatch: got {predicted_bboxes_xywh.shape[0]} vs expected {expected_len}"

    def close(self):
        self._zip_file.close()


def _extract_len_from_sequence_info(si) -> Optional[int]:
    for name in ('sequence_length', 'track_length', 'length', 'num_frames', 'frame_count', 'nframes'):
        if hasattr(si, name):
            v = getattr(si, name)
            if isinstance(v, (int, np.integer)) and v > 0:
                return int(v)
    if hasattr(si, '_fields'):
        ints = []
        for name in getattr(si, '_fields'):
            try:
                v = getattr(si, name)
            except Exception:
                continue
            if isinstance(v, (int, np.integer)) and v > 0:
                ints.append(int(v))
        if ints:
            return int(max(ints))
    try:
        ints = [int(v) for v in si if isinstance(v, (int, np.integer)) and int(v) > 0]
        if ints:
            return int(max(ints))
    except TypeError:
        pass
    return None


def _infer_total_length(e: SequenceEvaluationResult_SOT) -> int:
    si_len = _extract_len_from_sequence_info(e.sequence_info)
    if si_len is not None:
        return si_len
    if e.groundtruth_object_existence_flag is not None:
        return int(len(e.groundtruth_object_existence_flag))
    if e.groundtruth_box is not None:
        return int(len(e.groundtruth_box))
    idx = np.asarray(e.evaluated_frame_indices, dtype=int)
    return int(idx.max()) + 1 if idx.size > 0 else 0


def _align_full_xywh_with_dummy(e: SequenceEvaluationResult_SOT, pred_xyxy: Optional[np.ndarray]) -> np.ndarray:
    T = _infer_total_length(e)
    out_xyxy = np.zeros((T, 4), dtype=np.float32)
    idx = np.asarray(e.evaluated_frame_indices, dtype=int)
    if pred_xyxy is not None and pred_xyxy.size > 0 and idx.size > 0:
        out_xyxy[idx] = pred_xyxy.astype(np.float32)[:len(idx)]
    flag = e.groundtruth_object_existence_flag
    if flag is not None and len(flag) == T:
        exist = np.asarray(flag, dtype=bool)
        out_xyxy[~exist] = 0.0
    return bbox_xyxy_to_xywh(out_xyxy).astype(np.float32), T


class OTBEvaluationToolAdaptor(EvaluationResultHandler):
    def __init__(self, tracker_name: str, output_folder: str, file_name: str, rasterize_bbox: bool = True):
        self._writer = OTBEvaluationToolTrackingResultWriter(output_folder, file_name)
        self._tracker_name = tracker_name
        self._result_adjuster = ExternalToolkitCompatibilityHelper()
        self._rasterize_bbox = rasterize_bbox

    def close(self):
        self._writer.close()

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT], evaluation_progresses: Sequence[EvaluationProgress]):
        for e, prog in zip(evaluation_results, evaluation_progresses):
            # 호환 보정
            pred_xyxy = self._result_adjuster.adjust(e.sequence_info.dataset_name, e.output_box)

            if self._rasterize_bbox and pred_xyxy is not None:
                pred_xyxy = bbox_rasterize(pred_xyxy)

            pred_xywh_full, T = _align_full_xywh_with_dummy(e, pred_xyxy)

            repeat_index = prog.repeat_index
            if prog.this_dataset is not None:
                repeat_index = repeat_index if prog.this_dataset.total_repeat_times > 1 else None

            self._writer.write(self._tracker_name, repeat_index, e.sequence_info.sequence_name, pred_xywh_full, T)
