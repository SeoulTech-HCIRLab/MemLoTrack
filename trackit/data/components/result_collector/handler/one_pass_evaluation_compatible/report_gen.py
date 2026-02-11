import csv
import numpy as np
import json
import pickle
from typing import Optional, Dict
from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from ..utils.writer import FolderWriter
from .ope_metrics import DatasetOPEMetricsList, OPEMetrics


def _infer_total_length(
    frame_indices: np.ndarray,
    groundtruth_object_existence: Optional[np.ndarray],
    groundtruth_bounding_boxes: Optional[np.ndarray],
) -> int:
    if groundtruth_object_existence is not None:
        return int(len(groundtruth_object_existence))
    if groundtruth_bounding_boxes is not None:
        return int(len(groundtruth_bounding_boxes))
    idx = np.asarray(frame_indices, dtype=int)
    return int(idx.max()) + 1 if idx.size > 0 else 0


def _build_full_xyxy_with_dummy(
    frame_indices: np.ndarray,
    predicted_bboxes_xyxy: Optional[np.ndarray],
    groundtruth_object_existence: Optional[np.ndarray],
    groundtruth_bounding_boxes: Optional[np.ndarray],
) -> np.ndarray:
    """
    전체 프레임 길이에 맞춰 XYXY를 만들고,
    - 평가되지 않은 프레임: 0으로 둠
    - GT 미존재 프레임: 0 0 0 0 으로 강제
    """
    total = _infer_total_length(frame_indices, groundtruth_object_existence, groundtruth_bounding_boxes)
    full_xyxy = np.zeros((total, 4), dtype=np.float32)

    if predicted_bboxes_xyxy is not None and np.size(predicted_bboxes_xyxy) > 0:
        idx = np.asarray(frame_indices, dtype=int)
        full_xyxy[idx] = np.asarray(predicted_bboxes_xyxy, dtype=np.float32)[: len(idx)]

    if groundtruth_object_existence is not None and len(groundtruth_object_existence) == total:
        exist = np.asarray(groundtruth_object_existence, dtype=bool)
        full_xyxy[~exist] = 0.0

    return full_xyxy



def dump_sequence_tracking_results_with_groundtruth(folder_writer: FolderWriter,
                                                    tracker_name: str,
                                                    repeat_index: Optional[int],
                                                    dataset_name: str, sequence_name: str,
                                                    frame_indices: np.ndarray,
                                                    prediction_confidence: Optional[np.ndarray],
                                                    predicted_bboxes: Optional[np.ndarray],
                                                    groundtruth_object_existence: Optional[np.ndarray],
                                                    groundtruth_bounding_boxes: Optional[np.ndarray],
                                                    time_costs: Optional[np.ndarray],
                                                    iou_of_frames: Optional[np.ndarray]):
    """
    호환 리포트 + 항상 {sequence}.txt 생성 (풀-프레임 + dummy).
    """
    path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name, sequence_name)

    # 1) eval.pkl (원래 기능 유지)
    with folder_writer.open_binary_file_handle((*path, 'eval.pkl')) as f:
        pickle.dump(
            {
                'frame_index': frame_indices,
                'confidence': prediction_confidence,
                'bounding_box': predicted_bboxes,
                'time': time_costs
            }, f)

    # 2) eval.csv (원래 기능 유지: 부분 프레임 기준, 호환성 유지)
    seq_len_partial = len(frame_indices)
    eval_matrix = np.empty((seq_len_partial, 12), dtype=np.float64)
    eval_matrix[:, 0] = frame_indices.astype(np.float64)

    if groundtruth_object_existence is None:
        gt_exist_partial = np.ones((seq_len_partial,), dtype=np.float64)
    else:
        gt_exist_partial = np.asarray(groundtruth_object_existence, dtype=np.float64)[frame_indices]
    eval_matrix[:, 1] = gt_exist_partial

    if prediction_confidence is None:
        pred_conf_partial = np.zeros((seq_len_partial,), dtype=np.float64)
    else:
        pred_conf_partial = np.asarray(prediction_confidence, dtype=np.float64)[:seq_len_partial]
    eval_matrix[:, 2] = pred_conf_partial

    pred_xywh_partial = bbox_xyxy_to_xywh(
        np.asarray(predicted_bboxes, dtype=np.float64) if predicted_bboxes is not None
        else np.zeros((seq_len_partial, 4), dtype=np.float64)
    ).astype(np.float64)
    eval_matrix[:, 3:7] = pred_xywh_partial

    if groundtruth_bounding_boxes is None:
        gt_xywh_all = np.zeros((seq_len_partial, 4), dtype=np.float64)
    else:
        gt_xywh_all = bbox_xyxy_to_xywh(np.asarray(groundtruth_bounding_boxes, dtype=np.float64)).astype(np.float64)[frame_indices]
    eval_matrix[:, 7:11] = gt_xywh_all

    eval_matrix[:, 11] = (
        np.asarray(iou_of_frames, dtype=np.float64)[:seq_len_partial]
        if iou_of_frames is not None else
        np.zeros((seq_len_partial,), dtype=np.float64)
    )

    with folder_writer.open_text_file_handle((*path, 'eval.csv')) as f:
        np.savetxt(
            f, eval_matrix, fmt='%.3f', delimiter=',',
            header=','.join((
                'ind', 'gt_obj_exist', 'pred_conf',
                'pred_x', 'pred_y', 'pred_w', 'pred_h',
                'gt_x', 'gt_y', 'gt_w', 'gt_h', 'iou'
            ))
        )

    # 3) {sequence}.txt (신규/중요): 전체 프레임 + dummy(0 0 0 0) — 공백 구분, 괄호/콤마 없음
    full_xyxy = _build_full_xyxy_with_dummy(frame_indices, predicted_bboxes,
                                            groundtruth_object_existence, groundtruth_bounding_boxes)
    full_xywh = bbox_xyxy_to_xywh(full_xyxy).astype(np.float32)

    # (a) 툴킷식 경로
    with folder_writer.open_text_file_handle((*path, f'{sequence_name}.txt')) as f:
        np.savetxt(f, full_xywh, fmt='%.3f')
        # print(f"[DBG] OPEC(report_gen) write: {dataset_name}/{sequence_name}.txt lines={full_xywh.shape[0]}", flush=True)


    # (b) 루트 레벨에도 한 번 더 써줌 (일부 파이프라인은 이 경로만 읽음)
    with folder_writer.open_text_file_handle((f'{sequence_name}.txt',)) as f:
        np.savetxt(f, full_xywh, fmt='%.3f')
        # print(f"[DBG] OPEC(report_gen) write(root): {sequence_name}.txt lines={full_xywh.shape[0]}", flush=True)



def generate_sequence_one_pass_evaluation_report(
        folder_writer: FolderWriter, tracker_name: str,
        repeat_index: Optional[int],
        dataset_name: str, sequence_name: str,
        ope_metrics: OPEMetrics):
    path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name, sequence_name)

    sequence_report = {
        'success_score': ope_metrics.success_score,
        'precision_score': ope_metrics.precision_score,
        'normalized_precision_score': ope_metrics.normalized_precision_score,
        # 'average_overlap': ope_metrics.average_overlap,
        'success_rate_at_overlap_0.5': ope_metrics.success_rate_at_overlap_0_5,
        'success_rate_at_overlap_0.75': ope_metrics.success_rate_at_overlap_0_75,
        'success_curve': ope_metrics.success_curve.tolist(),
        'precision_curve': ope_metrics.precision_curve.tolist(),
        'normalized_precision_curve': ope_metrics.normalized_precision_curve.tolist(),
        'fps': ope_metrics.get_fps()
    }
    with folder_writer.open_text_file_handle((*path, 'performance.json')) as f:
        json.dump(sequence_report, f, indent=2)


def generate_dataset_one_pass_evaluation_report(
        folder_writer: FolderWriter, tracker_name: str,
        repeat_index: Optional[int], dataset_name: str,
        all_sequences_ope_metrics: DatasetOPEMetricsList,
        dataset_summary_ope_metrics: Optional[OPEMetrics] = None):
    if dataset_summary_ope_metrics is None:
        dataset_summary_ope_metrics = all_sequences_ope_metrics.get_mean()

    path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name)
    with folder_writer.open_text_file_handle((*path, 'sequences_performance.csv')) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(('Sequence Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
                              'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
        for sequence_name, ope_metrics in all_sequences_ope_metrics:
            csv_writer.writerow((sequence_name,
                                 ope_metrics.success_score, ope_metrics.precision_score,
                                 ope_metrics.normalized_precision_score,
                                 ope_metrics.success_rate_at_overlap_0_5,
                                 ope_metrics.success_rate_at_overlap_0_75,
                                 ope_metrics.get_fps()))

    dataset_report = {'success_score': dataset_summary_ope_metrics.success_score,
                      'precision_score': dataset_summary_ope_metrics.precision_score,
                      'normalized_precision_score': dataset_summary_ope_metrics.normalized_precision_score,
                    #   'average_overlap': dataset_summary_ope_metrics.average_overlap,
                      'success_rate_at_overlap_0.5': dataset_summary_ope_metrics.success_rate_at_overlap_0_5,
                      'success_rate_at_overlap_0.75': dataset_summary_ope_metrics.success_rate_at_overlap_0_75,
                      'success_curve': dataset_summary_ope_metrics.success_curve.tolist(),
                      'precision_curve': dataset_summary_ope_metrics.precision_curve.tolist(),
                      'normalized_precision_curve': dataset_summary_ope_metrics.normalized_precision_curve.tolist(),
                      'fps': dataset_summary_ope_metrics.get_fps()}
    with folder_writer.open_text_file_handle((*path, 'performance.json')) as f:
        json.dump(dataset_report, f, indent=2)


def generate_one_pass_evaluation_summary_report(folder_writer: FolderWriter, tracker_name: str,
                                                repeat_index: Optional[int],
                                                datasets_summary_ope_metrics: Dict[str, OPEMetrics]):
    with folder_writer.open_text_file_handle(
            (f'{tracker_name}_performance.csv' if repeat_index is None else f'{tracker_name}_{repeat_index:03d}_performance.csv',)) as f:
        writer = csv.writer(f)
        writer.writerow(('Dataset Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
                          'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
        for dataset_name, dataset_summary_ope_metrics in datasets_summary_ope_metrics.items():
            writer.writerow((dataset_name,
                             dataset_summary_ope_metrics.success_score,
                             dataset_summary_ope_metrics.precision_score,
                             dataset_summary_ope_metrics.normalized_precision_score,
                            #  dataset_summary_ope_metrics.average_overlap,
                             dataset_summary_ope_metrics.success_rate_at_overlap_0_5,
                             dataset_summary_ope_metrics.success_rate_at_overlap_0_75,
                             dataset_summary_ope_metrics.get_fps()))


# """
# 아래의 version 은 원 MemLoTrack 저자의 github 코드
# """
# import csv
# import numpy as np
# import json
# import pickle
# from typing import Optional, Dict
# from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
# from ..utils.writer import FolderWriter
# from .ope_metrics import DatasetOPEResults, OPEMetrics


# def _infer_total_length(
#     frame_indices: np.ndarray,
#     groundtruth_object_existence: Optional[np.ndarray],
#     groundtruth_bounding_boxes: Optional[np.ndarray],
# ) -> int:
#     if groundtruth_object_existence is not None:
#         return int(len(groundtruth_object_existence))
#     if groundtruth_bounding_boxes is not None:
#         return int(len(groundtruth_bounding_boxes))
#     idx = np.asarray(frame_indices, dtype=int)
#     return int(idx.max()) + 1 if idx.size > 0 else 0


# def _build_full_xyxy_with_dummy(
#     frame_indices: np.ndarray,
#     predicted_bboxes_xyxy: Optional[np.ndarray],
#     groundtruth_object_existence: Optional[np.ndarray],
#     groundtruth_bounding_boxes: Optional[np.ndarray],
# ) -> np.ndarray:
#     """
#     전체 프레임 길이에 맞춰 XYXY를 만들고,
#     - 평가되지 않은 프레임: 0으로 둠
#     - GT 미존재 프레임: 0 0 0 0 으로 강제
#     """
#     total = _infer_total_length(frame_indices, groundtruth_object_existence, groundtruth_bounding_boxes)
#     full_xyxy = np.zeros((total, 4), dtype=np.float32)

#     if predicted_bboxes_xyxy is not None and np.size(predicted_bboxes_xyxy) > 0:
#         idx = np.asarray(frame_indices, dtype=int)
#         full_xyxy[idx] = np.asarray(predicted_bboxes_xyxy, dtype=np.float32)[: len(idx)]

#     if groundtruth_object_existence is not None and len(groundtruth_object_existence) == total:
#         exist = np.asarray(groundtruth_object_existence, dtype=bool)
#         full_xyxy[~exist] = 0.0

#     return full_xyxy



# def dump_sequence_tracking_results_with_groundtruth(folder_writer: FolderWriter,
#                                                     tracker_name: str,
#                                                     repeat_index: Optional[int],
#                                                     dataset_name: str, sequence_name: str,
#                                                     frame_indices: np.ndarray,
#                                                     prediction_confidence: np.ndarray,
#                                                     predicted_bboxes: np.ndarray,
#                                                     groundtruth_object_existence: np.ndarray,
#                                                     groundtruth_bounding_boxes: np.ndarray,
#                                                     time_costs: np.ndarray,
#                                                     iou_of_frames: np.ndarray):
#     path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name, sequence_name)

#     with folder_writer.open_binary_file_handle((*path, 'eval.pkl')) as f:
#         pickle.dump(
#             {'frame_index': frame_indices, 'confidence': prediction_confidence, 'bounding_box': predicted_bboxes,
#              'time': time_costs}, f)

#     sequence_length = len(frame_indices)
#     eval_result_matrix = np.empty((sequence_length, 12), dtype=np.float64)
#     eval_result_matrix[:, 0] = frame_indices.astype(np.float64)
#     eval_result_matrix[:, 1] = groundtruth_object_existence.astype(np.float64)
#     eval_result_matrix[:, 2] = prediction_confidence.astype(np.float64)
#     eval_result_matrix[:, 3:7] = bbox_xyxy_to_xywh(predicted_bboxes).astype(np.float64)
#     eval_result_matrix[:, 7:11] = bbox_xyxy_to_xywh(groundtruth_bounding_boxes).astype(np.float64)
#     eval_result_matrix[:, 11] = iou_of_frames.astype(np.float64)

#     with folder_writer.open_text_file_handle((*path, 'eval.csv')) as f:
#         np.savetxt(f, eval_result_matrix, fmt='%.3f', delimiter=',',
#                    header=','.join(('ind', 'gt_obj_exist', 'pred_conf', 'pred_x', 'pred_y', 'pred_w', 'pred_h', 'gt_x',
#                                     'gt_y', 'gt_w', 'gt_h', 'iou')))


# def generate_sequence_one_pass_evaluation_report(
#         folder_writer: FolderWriter, tracker_name: str,
#         repeat_index: Optional[int],
#         dataset_name: str, sequence_name: str,
#         ope_metrics: OPEMetrics):
#     path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name, sequence_name)

#     sequence_report = {
#         'success_score': ope_metrics.success_score,
#         'precision_score': ope_metrics.precision_score,
#         'normalized_precision_score': ope_metrics.normalized_precision_score,
#         'success_rate_at_overlap_0.5': ope_metrics.success_rate_at_overlap_0_5,
#         'success_rate_at_overlap_0.75': ope_metrics.success_rate_at_overlap_0_75,
#         'success_curve': ope_metrics.success_curve.tolist(),
#         'precision_curve': ope_metrics.precision_curve.tolist(),
#         'normalized_precision_curve': ope_metrics.normalized_precision_curve.tolist(),
#         'fps': ope_metrics.get_fps()
#     }
#     with folder_writer.open_text_file_handle((*path, 'performance.json')) as f:
#         json.dump(sequence_report, f, indent=2)


# def generate_dataset_one_pass_evaluation_report(
#         folder_writer: FolderWriter, tracker_name: str,
#         repeat_index: Optional[int], dataset_name: str,
#         all_sequences_ope_metrics: DatasetOPEResults,
#         dataset_summary_ope_metrics: Optional[OPEMetrics] = None):
#     if dataset_summary_ope_metrics is None:
#         dataset_summary_ope_metrics = all_sequences_ope_metrics.get_mean()

#     path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name)
#     with folder_writer.open_text_file_handle((*path, 'sequences_performance.csv')) as f:
#         csv_writer = csv.writer(f)
#         csv_writer.writerow(('Sequence Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
#                              'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
#         for sequence_name, ope_metrics in all_sequences_ope_metrics:
#             csv_writer.writerow((sequence_name,
#                                  ope_metrics.success_score, ope_metrics.precision_score,
#                                  ope_metrics.normalized_precision_score,
#                                  ope_metrics.success_rate_at_overlap_0_5,
#                                  ope_metrics.success_rate_at_overlap_0_75,
#                                  ope_metrics.get_fps()))

#     dataset_report = {'success_score': dataset_summary_ope_metrics.success_score,
#                       'precision_score': dataset_summary_ope_metrics.precision_score,
#                       'normalized_precision_score': dataset_summary_ope_metrics.normalized_precision_score,
#                       'success_rate_at_overlap_0.5': dataset_summary_ope_metrics.success_rate_at_overlap_0_5,
#                       'success_rate_at_overlap_0.75': dataset_summary_ope_metrics.success_rate_at_overlap_0_75,
#                       'success_curve': dataset_summary_ope_metrics.success_curve.tolist(),
#                       'precision_curve': dataset_summary_ope_metrics.precision_curve.tolist(),
#                       'normalized_precision_curve': dataset_summary_ope_metrics.normalized_precision_curve.tolist(),
#                       'fps': dataset_summary_ope_metrics.get_fps()}
#     with folder_writer.open_text_file_handle((*path, 'performance.json')) as f:
#         json.dump(dataset_report, f, indent=2)


# def generate_one_pass_evaluation_summary_report(folder_writer: FolderWriter, tracker_name: str,
#                                                 repeat_index: Optional[int],
#                                                 datasets_summary_ope_metrics: Dict[str, OPEMetrics]):
#     with folder_writer.open_text_file_handle(
#             (f'{tracker_name}_performance.csv' if repeat_index is None else f'{tracker_name}_{repeat_index:03d}_performance.csv',)) as f:
#         writer = csv.writer(f)
#         writer.writerow(('Dataset Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
#                          'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
#         for dataset_name, dataset_summary_ope_metrics in datasets_summary_ope_metrics.items():
#             writer.writerow((dataset_name,
#                              dataset_summary_ope_metrics.success_score,
#                              dataset_summary_ope_metrics.precision_score,
#                              dataset_summary_ope_metrics.normalized_precision_score,
#                              dataset_summary_ope_metrics.success_rate_at_overlap_0_5,
#                              dataset_summary_ope_metrics.success_rate_at_overlap_0_75,
#                              dataset_summary_ope_metrics.get_fps()))