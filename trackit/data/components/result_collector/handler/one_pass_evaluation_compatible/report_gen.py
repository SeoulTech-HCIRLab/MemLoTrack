import csv
import numpy as np
import json
import pickle
from typing import Optional, Dict
from ..utils.writer import FolderWriter
from .ope_metrics import DatasetOPEMetricsList, OPEMetrics



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
    """Write the detailed pickle and CSV reports for one sequence."""
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
