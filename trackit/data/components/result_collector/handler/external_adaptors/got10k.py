import os
import io
import zipfile
from typing import Optional, Sequence
import numpy as np

from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT

from ...progress_tracer import EvaluationProgress
from .. import EvaluationResultHandler


class GOT10kEvaluationToolFileWriter:
    def __init__(self, output_folder: str, output_file_name: str):
        self._output_file_path_prefix = os.path.join(output_folder, output_file_name)
        self._zipfile = zipfile.ZipFile(self._output_file_path_prefix + '.zip', 'w', zipfile.ZIP_DEFLATED)
        self._single_run_zipfiles = {}
        self._duplication_check = set()

    def write(self, tracker_name: str, sequence_name: str,
              predicted_bboxes: np.ndarray, time_costs: np.ndarray,
              repeat_index: Optional[int] = None):
        single_run_zipfile = None
        if repeat_index is not None:
            if repeat_index not in self._single_run_zipfiles:
                self._single_run_zipfiles[repeat_index] = zipfile.ZipFile(
                    self._output_file_path_prefix + f'{repeat_index + 1:03d}.zip',
                    'w',
                    zipfile.ZIP_DEFLATED
                )
            single_run_zipfile = self._single_run_zipfiles[repeat_index]

        if repeat_index is None:
            repeat_index = 0

        assert (tracker_name, repeat_index, sequence_name) not in self._duplication_check, "duplicated sequence name detected"
        self._duplication_check.add((tracker_name, repeat_index, sequence_name))

        # ===== bbox 파일 (공백 구분, 각 줄 4숫자, dummy 포함) =====
        with io.BytesIO() as result_file_content:
            # 구분자 명시하지 않음 → 공백만 사용됨
            np.savetxt(result_file_content, predicted_bboxes, fmt='%.3f')
            print(f"[DBG] GOT10k write: {sequence_name}_{repeat_index + 1:03d}.txt lines={predicted_bboxes.shape[0]}", flush=True)


            self._zipfile.writestr(
                '/'.join((tracker_name, sequence_name, f'{sequence_name}_{repeat_index + 1:03d}.txt')),
                result_file_content.getvalue()
            )
            if single_run_zipfile is not None:
                single_run_zipfile.writestr(
                    '/'.join((tracker_name, sequence_name, f'{sequence_name}_001.txt')),
                    result_file_content.getvalue()
                )

        # ===== time 파일 (길이도 전프레임에 맞춤, 공백 구분) =====
        if repeat_index == 0 or single_run_zipfile is not None:
            with io.BytesIO() as time_file_content:
                np.savetxt(time_file_content, time_costs.astype(np.float32).reshape(-1), fmt='%.8f')
                print(f"[DBG] GOT10k write: {sequence_name}_{repeat_index + 1:03d}.txt lines={predicted_bboxes.shape[0]}", flush=True)

                if repeat_index == 0:
                    self._zipfile.writestr(
                        '/'.join((tracker_name, sequence_name, f'{sequence_name}_time.txt')),
                        time_file_content.getvalue()
                    )
                if single_run_zipfile is not None:
                    single_run_zipfile.writestr(
                        '/'.join((tracker_name, sequence_name, f'{sequence_name}_time.txt')),
                        time_file_content.getvalue()
                    )

    def close(self):
        for zip_file in self._single_run_zipfiles.values():
            zip_file.close()
        self._zipfile.close()


# ===== 내부 유틸: 총 프레임 길이 추정 =====
def _infer_total_length(evaluation_result: SequenceEvaluationResult_SOT) -> int:
    if evaluation_result.groundtruth_object_existence_flag is not None:
        return int(len(evaluation_result.groundtruth_object_existence_flag))
    if evaluation_result.groundtruth_box is not None:
        return int(len(evaluation_result.groundtruth_box))
    # SequenceInfo 안의 필드들 중 존재하는 것 우선 사용
    seq_info = evaluation_result.sequence_info
    for attr in ('sequence_length', 'track_length'):
        if hasattr(seq_info, attr):
            val = getattr(seq_info, attr)
            if val is not None:
                return int(val)
    idx = np.asarray(evaluation_result.evaluated_frame_indices, dtype=int)
    return int(idx.max()) + 1 if idx.size > 0 else 0


# ===== 내부 유틸: bbox/time을 프레임 전수로 정렬 + dummy 처리 =====
def _build_full_xywh_and_time(evaluation_result: SequenceEvaluationResult_SOT,
                              predicted_bboxes_xyxy: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(evaluation_result.evaluated_frame_indices, dtype=int)
    total = _infer_total_length(evaluation_result)

    full_xyxy = np.zeros((total, 4), dtype=np.float32)
    if predicted_bboxes_xyxy is not None and predicted_bboxes_xyxy.size > 0 and idx.size > 0:
        # idx 길이에 맞춰 슬라이스 후 채움
        full_xyxy[idx] = predicted_bboxes_xyxy.astype(np.float32)[:len(idx)]

    # 객체 미존재 프레임은 무조건 dummy(0 0 0 0)
    flag = evaluation_result.groundtruth_object_existence_flag
    if flag is not None and len(flag) == total:
        exist = np.asarray(flag, dtype=bool)
        full_xyxy[~exist] = 0.0

    full_xywh = bbox_xyxy_to_xywh(full_xyxy).astype(np.float32)

    # time_cost도 전프레임으로 확장(미평가/미존재 프레임은 0)
    tc = getattr(evaluation_result, 'time_cost', None)
    full_time = np.zeros((total,), dtype=np.float32)
    if tc is not None and np.size(tc) > 0 and idx.size > 0:
        tc = np.asarray(tc).reshape(-1).astype(np.float32)
        full_time[idx] = tc[:len(idx)]

    return full_xywh, full_time


class GOT10KEvaluationToolAdaptor(EvaluationResultHandler):
    def __init__(self, tracker_name: str,  output_folder: str, file_name: str = 'GOT10k', rasterize_bbox: bool = True):
        self._writer = GOT10kEvaluationToolFileWriter(output_folder, file_name)
        self._tracker_name = tracker_name
        self._rasterize_bbox = rasterize_bbox

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT], evaluation_progresses: Sequence[EvaluationProgress]):
        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            predicted_bboxes_xyxy = evaluation_result.output_box
            if self._rasterize_bbox and predicted_bboxes_xyxy is not None:
                predi
