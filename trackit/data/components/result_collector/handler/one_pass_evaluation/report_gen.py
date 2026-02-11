import csv
import numpy as np
import json
import pickle
from typing import Optional, Dict, Tuple
from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from ..utils.writer import FolderWriter
# from .plot_ope_metric import draw_success_plot, draw_precision_plot, draw_normalized_precision_plot
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
    total = _infer_total_length(frame_indices, groundtruth_object_existence, groundtruth_bounding_boxes)
    full_xyxy = np.zeros((total, 4), dtype=np.float32)

    if predicted_bboxes_xyxy is not None and np.size(predicted_bboxes_xyxy) > 0:
        idx = np.asarray(frame_indices, dtype=int)
        full_xyxy[idx] = np.asarray(predicted_bboxes_xyxy, dtype=np.float32)[: len(idx)]

    # 객체 미존재 프레임은 무조건 dummy(0 0 0 0)
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
    기존 기능(픽클/CSV) + 추가 기능:
    - {sequence_name}.txt 를 항상 생성
    - txt는 전체 프레임 길이와 동일한 줄 수
    - 객체 미존재 프레임은 0 0 0 0 (공백 구분, 괄호/콤마 없음)
    """
    path = (tracker_name if repeat_index is None else f'{tracker_name}_{repeat_index:03d}', dataset_name, sequence_name)

    # === eval.pkl (원래 기능 유지) ===
    with folder_writer.open_binary_file_handle((*path, 'eval.pkl')) as f:
        pickle.dump(
            {'frame_index': frame_indices,
             'confidence': prediction_confidence,
             'bounding_box': predicted_bboxes,
             'time': time_costs}, f)

    # === CSV (원래 기능 유지) ===
    # 주의: 기존 코드는 len(frame_indices)를 길이로 삼았는데,
    #       여기서는 기존 동작을 보존하되, TXT는 "전체 길이"로 별도 생성한다.
    sequence_length_partial = len(frame_indices)
    eval_result_matrix = np.empty((sequence_length_partial, 12), dtype=np.float64)
    eval_result_matrix[:, 0] = frame_indices.astype(np.float64)
    if groundtruth_object_existence is None:
        gt_exist_partial = np.ones((sequence_length_partial,), dtype=np.float64)
    else:
        # frame_indices 위치의 gt_exist만 추출
        gt_exist_partial = np.asarray(groundtruth_object_existence, dtype=np.float64)[frame_indices]
    eval_result_matrix[:, 1] = gt_exist_partial

    if prediction_confidence is None:
        pred_conf_partial = np.zeros((sequence_length_partial,), dtype=np.float64)
    else:
        pred_conf_partial = np.asarray(prediction_confidence, dtype=np.float64)[:sequence_length_partial]
    eval_result_matrix[:, 2] = pred_conf_partial

    pred_xywh_partial = bbox_xyxy_to_xywh(
        np.asarray(predicted_bboxes, dtype=np.float64) if predicted_bboxes is not None else
        np.zeros((sequence_length_partial, 4), dtype=np.float64)
    ).astype(np.float64)
    eval_result_matrix[:, 3:7] = pred_xywh_partial

    gt_xywh_partial = bbox_xyxy_to_xywh(
        np.asarray(groundtruth_bounding_boxes, dtype=np.float64) if groundtruth_bounding_boxes is not None else
        np.zeros((sequence_length_partial, 4), dtype=np.float64)
    ).astype(np.float64)[frame_indices]
    eval_result_matrix[:, 7:11] = gt_xywh_partial

    eval_result_matrix[:, 11] = (np.asarray(iou_of_frames, dtype=np.float64)[:sequence_length_partial]
                                 if iou_of_frames is not None else
                                 np.zeros((sequence_length_partial,), dtype=np.float64))

    with folder_writer.open_text_file_handle((*path, 'eval.csv')) as f:
        np.savetxt(f, eval_result_matrix, fmt='%.3f', delimiter=',',
                   header=','.join(('ind', 'gt_obj_exist', 'pred_conf', 'pred_x', 'pred_y', 'pred_w', 'pred_h', 'gt_x',
                                    'gt_y', 'gt_w', 'gt_h', 'iou')))

    # === TXT (신규 추가): 전체 프레임 + dummy 반영 ===
    full_xyxy = _build_full_xyxy_with_dummy(frame_indices, predicted_bboxes,
                                            groundtruth_object_existence, groundtruth_bounding_boxes)
    full_xywh = bbox_xyxy_to_xywh(full_xyxy).astype(np.float32)

    # 툴킷 스타일 경로: <tracker or tracker_###>/<dataset>/<sequence>/<sequence>.txt
    with folder_writer.open_text_file_handle((*path, f'{sequence_name}.txt')) as f:
        # 공백 구분, 괄호/콤마 없음
        np.savetxt(f, full_xywh, fmt='%.3f')
        print(f"[DBG] OPE(report_gen) write: {dataset_name}/{sequence_name}.txt lines={full_xywh.shape[0]}", flush=True)


    # 루트 레벨에도 {sequence}.txt 추가 저장 (일부 파이프라인이 이 경로만 읽을 수 있음)
    with folder_writer.open_text_file_handle((f'{sequence_name}.txt',)) as f:
        np.savetxt(f, full_xywh, fmt='%.3f')
        print(f"[DBG] OPE(report_gen) write: {dataset_name}/{sequence_name}.txt lines={full_xywh.shape[0]}", flush=True)



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
        'average_overlap': ope_metrics.average_overlap,
        'success_rate_at_iou_0.5': ope_metrics.success_rate_at_iou_0_5,
        'success_rate_at_iou_0.75': ope_metrics.success_rate_at_iou_0_75,
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
                             'Average Overlap', 'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
        for sequence_name, ope_metrics in all_sequences_ope_metrics:
            csv_writer.writerow((sequence_name,
                                 ope_metrics.success_score, ope_metrics.precision_score,
                                 ope_metrics.normalized_precision_score,
                                 ope_metrics.average_overlap, ope_metrics.success_rate_at_iou_0_5,
                                 ope_metrics.success_rate_at_iou_0_75,
                                 ope_metrics.get_fps()))

    # with folder_writer.open_binary_file_handle((*path, 'success_plot.pdf')) as f:
    #     draw_success_plot(np.expand_dims(dataset_summary_ope_metrics.success_curve, axis=0), (tracker_name,), f)
    # with folder_writer.open_binary_file_handle((*path, 'precision_plot.pdf')) as f:
    #     draw_precision_plot(np.expand_dims(dataset_summary_ope_metrics.precision_curve, axis=0), (tracker_name,), f)
    # with folder_writer.open_binary_file_handle((*path, 'norm_precision_plot.pdf')) as f:
    #     draw_normalized_precision_plot(np.expand_dims(dataset_summary_ope_metrics.normalized_precision_curve, axis=0),
    #                                    (tracker_name,), f)

    dataset_report = {'success_score': dataset_summary_ope_metrics.success_score,
                      'precision_score': dataset_summary_ope_metrics.precision_score,
                      'normalized_precision_score': dataset_summary_ope_metrics.normalized_precision_score,
                      'average_overlap': dataset_summary_ope_metrics.average_overlap,
                      'success_rate_at_iou_0.5': dataset_summary_ope_metrics.success_rate_at_iou_0_5,
                      'success_rate_at_iou_0.75': dataset_summary_ope_metrics.success_rate_at_iou_0_75,
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
                         'Average Overlap', 'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
        for dataset_name, dataset_summary_ope_metrics in datasets_summary_ope_metrics.items():
            writer.writerow((dataset_name,
                             dataset_summary_ope_metrics.success_score,
                             dataset_summary_ope_metrics.precision_score,
                             dataset_summary_ope_metrics.normalized_precision_score,
                             dataset_summary_ope_metrics.average_overlap,
                             dataset_summary_ope_metrics.success_rate_at_iou_0_5,
                             dataset_summary_ope_metrics.success_rate_at_iou_0_75,
                             dataset_summary_ope_metrics.get_fps()))
