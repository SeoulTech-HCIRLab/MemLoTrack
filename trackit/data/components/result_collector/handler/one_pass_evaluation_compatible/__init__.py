import os.path
from typing import Optional, Sequence, Mapping, Tuple, MutableMapping

import numpy as np
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from ..utils.writer import FolderWriter, ZipfileWriter
from ...progress_tracer import EvaluationProgress
from .. import EvaluationResultHandler
from .ope_metrics import (
    OPEMetrics,
    DatasetOPEMetricsListBuilder,
    DatasetOPEMetricsList,
    compute_OPE_metrics_mean,
    compute_one_pass_evaluation_metrics,
)
from .report_gen import (
    generate_dataset_one_pass_evaluation_report,
    generate_one_pass_evaluation_summary_report,
    dump_sequence_tracking_results_with_groundtruth,
    generate_sequence_one_pass_evaluation_report,
)
from ..utils.compatibility import ExternalToolkitCompatibilityHelper


# =========================
# Helpers (full-length + dummy)
# =========================

def _extract_length_from_sequence_info(si) -> Optional[int]:
    # 1) common field names
    for name in ("sequence_length", "track_length", "length", "num_frames", "frame_count", "nframes"):
        if hasattr(si, name):
            v = getattr(si, name)
            if isinstance(v, (int, np.integer)) and v > 0:
                return int(v)
    # 2) NamedTuple fields
    if hasattr(si, "_fields"):
        ints = []
        for name in getattr(si, "_fields"):
            try:
                v = getattr(si, name)
            except Exception:
                continue
            if isinstance(v, (int, np.integer)) and v > 0:
                ints.append(int(v))
        if ints:
            return int(max(ints))
    # 3) Iterable ints
    try:
        ints = [int(v) for v in si if isinstance(v, (int, np.integer)) and int(v) > 0]
        if ints:
            return int(max(ints))
    except TypeError:
        pass
    return None


def _infer_total_length(e: SequenceEvaluationResult_SOT) -> int:
    si_len = _extract_length_from_sequence_info(e.sequence_info)
    if si_len is not None:
        return si_len
    if e.groundtruth_object_existence_flag is not None:
        return int(len(e.groundtruth_object_existence_flag))
    if e.groundtruth_box is not None:
        return int(len(e.groundtruth_box))
    idx = np.asarray(e.evaluated_frame_indices, dtype=int)
    return int(idx.max()) + 1 if idx.size > 0 else 0


def _expand_to_full_xyxy_time_conf(
    e: SequenceEvaluationResult_SOT,
    pred_xyxy: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      full_idx  : (T,)   [0..T-1]
      full_xyxy : (T,4)  predicted XYXY, un-evaluated frames = 0, non-exist frames forced to 0 0 0 0
      full_xywh : (T,4)  converted from full_xyxy
      full_time : (T,)   padded with 0
      full_conf : (T,)   padded with 0
    """
    total = _infer_total_length(e)
    full_idx = np.arange(total, dtype=int)
    idx = np.asarray(e.evaluated_frame_indices, dtype=int)

    # bbox
    full_xyxy = np.zeros((total, 4), dtype=(pred_xyxy.dtype if pred_xyxy is not None else np.float32))
    if pred_xyxy is not None and pred_xyxy.size > 0 and idx.size > 0:
        full_xyxy[idx] = pred_xyxy[: len(idx)]

    # force dummy for non-exist frames
    flag = e.groundtruth_object_existence_flag
    if flag is not None and len(flag) == total:
        exist = np.asarray(flag, dtype=bool)
        full_xyxy[~exist] = 0.0

    # time
    full_time = np.zeros((total,), dtype=np.float32)
    if e.time_cost is not None and np.size(e.time_cost) > 0 and idx.size > 0:
        tc = np.asarray(e.time_cost, dtype=np.float32).reshape(-1)
        full_time[idx] = tc[: len(idx)]

    # confidence
    full_conf = np.zeros((total,), dtype=np.float32)
    if e.output_confidence is not None and np.size(e.output_confidence) > 0 and idx.size > 0:
        cf = np.asarray(e.output_confidence, dtype=np.float32).reshape(-1)
        full_conf[idx] = cf[: len(idx)]

    full_xywh = bbox_xyxy_to_xywh(full_xyxy).astype(np.float32)
    return full_idx, full_xyxy, full_xywh, full_time, full_conf


def _write_sequence_txt_variants(
    folder_writer: FolderWriter,
    tracker_name: str,
    repeat_index: Optional[int],
    dataset_full_name: str,
    sequence_name: str,
    pred_xywh_full: np.ndarray,
) -> None:
    """
    Write predicted bbox text files in two common layouts:
    - <tracker or tracker_###>/<dataset_full_name>/<sequence_name>/<sequence_name>.txt
    - <sequence_name>.txt (root-level)
    Space-separated, four numbers per line, no brackets/commas.
    """
    if folder_writer is None:
        return
    base = tracker_name if repeat_index is None else f"{tracker_name}_{repeat_index:03d}"

    # Toolkit-style nested path
    with folder_writer.open_text_file_handle((base, dataset_full_name, sequence_name, f"{sequence_name}.txt")) as f:
        np.savetxt(f, pred_xywh_full.astype(np.float32), fmt="%.3f")

    # Root-level convenience file
    with folder_writer.open_text_file_handle((f"{sequence_name}.txt",)) as f:
        np.savetxt(f, pred_xywh_full.astype(np.float32), fmt="%.3f")


# =========================
# Main Handler
# =========================

class EvaluationResultPersistenceWithOPEMetricsCompatibleWithExternalTools(EvaluationResultHandler):
    def __init__(self, tracker_name: str, output_path: Optional[str], file_name: Optional[str], rasterize_bbox: bool):
        self._tracker_name = tracker_name
        self._folder_writer = None
        if output_path is not None and file_name is not None:
            self._folder_writer = ZipfileWriter(os.path.join(output_path, file_name + '.zip'))

        self._progress_aware_sub_handler = EvaluationResultPersistenceWithOPEMetrics_ProgressAware(rasterize_bbox)
        self._live_feed_sub_handler = EvaluationResultPersistenceWithOPEMetrics_LiveFeed(rasterize_bbox)
        self._collected_metrics = []
        self._is_closed = False

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT],
               evaluation_progresses: Sequence[EvaluationProgress]):
        assert not self._is_closed
        metrics = {}
        sub_handler_1_metrics = self._progress_aware_sub_handler(self._tracker_name, self._folder_writer, evaluation_results, evaluation_progresses)
        sub_handler_2_metrics = self._live_feed_sub_handler(self._tracker_name, self._folder_writer, evaluation_results, evaluation_progresses)
        if sub_handler_1_metrics is not None:
            metrics.update(sub_handler_1_metrics)
        if sub_handler_2_metrics is not None:
            metrics.update(sub_handler_2_metrics)
        if len(metrics) != 0:
            self._collected_metrics.extend(list(metrics.items()))

    def close(self):
        assert not self._is_closed
        metrics = {}
        sub_handler_1_metrics = self._progress_aware_sub_handler.finalize(self._tracker_name, self._folder_writer)
        sub_handler_2_metrics = self._live_feed_sub_handler.finalize(self._tracker_name, self._folder_writer)
        if sub_handler_1_metrics is not None:
            metrics.update(sub_handler_1_metrics)
        if sub_handler_2_metrics is not None:
            metrics.update(sub_handler_2_metrics)
        if len(metrics) != 0:
            self._collected_metrics.extend(list(metrics.items()))
        if self._folder_writer is not None:
            self._folder_writer.close()
            self._folder_writer = None
        self._is_closed = True

    def get_metrics(self) -> Optional[Sequence[Tuple[str, float]]]:
        return self._collected_metrics


class FinalOPEMetricsSummaryReportGenerator:
    def __init__(self):
        self._final_summary_metrics = {}

    def add(self, dataset_name: str, repeat_index: Optional[int], metrics: OPEMetrics):
        if repeat_index not in self._final_summary_metrics:
            self._final_summary_metrics[repeat_index] = {}
        this_repeat_summary_metrics = self._final_summary_metrics[repeat_index]
        assert dataset_name not in this_repeat_summary_metrics
        this_repeat_summary_metrics[dataset_name] = metrics

    def dump(self, folder_writer: FolderWriter, tracker_name: str):
        for repeat_index, this_repeat_summary_metrics in self._final_summary_metrics.items():
            sorted_metrics = dict(sorted(this_repeat_summary_metrics.items()))
            generate_one_pass_evaluation_summary_report(folder_writer, tracker_name, repeat_index, sorted_metrics)


def __get_summary_metric_name(metric_name: str, dataset_name: str, repeat_index: Optional[int]):
    if repeat_index is None:
        return f'{metric_name}_{dataset_name}'
    else:
        return f'{metric_name}_{dataset_name}_{repeat_index:03d}'


def _generate_dataset_summary_metrics_name_value_pair(dataset_name: str, repeat_index: Optional[int], metrics: OPEMetrics):
    return {
        __get_summary_metric_name('success_score', dataset_name, repeat_index): metrics.success_score,
        __get_summary_metric_name('precision_score', dataset_name, repeat_index): metrics.precision_score,
        __get_summary_metric_name('norm_precision_score', dataset_name, repeat_index): metrics.normalized_precision_score,
        __get_summary_metric_name('success_rate_at_overlap_0_5', dataset_name, repeat_index): metrics.success_rate_at_overlap_0_5,
        __get_summary_metric_name('success_rate_at_overlap_0_75', dataset_name, repeat_index): metrics.success_rate_at_overlap_0_75,
        __get_summary_metric_name('fps', dataset_name, repeat_index): metrics.get_fps(),
    }


class EvaluationResultPersistenceWithOPEMetrics_ProgressAware:
    def __init__(self, rasterize_bbox: bool):
        self._known_tracks_metric_cache = {}
        self._multi_run_dataset_metrics_cache = {}
        self._final_summary_report_generator = FinalOPEMetricsSummaryReportGenerator()
        self._compatibility_helper = ExternalToolkitCompatibilityHelper()
        self._rasterize_bbox = rasterize_bbox

    def __call__(self, tracker_name: str, folder_writer: Optional[FolderWriter],
                 evaluation_results: Sequence[SequenceEvaluationResult_SOT],
                 evaluation_progresses: Sequence[EvaluationProgress]) -> Optional[Mapping[str, float]]:
        summary_metrics = {}

        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            if evaluation_progress.this_dataset is None:
                continue

            assert evaluation_result.sequence_info.dataset_full_name is not None
            assert evaluation_result.sequence_info.sequence_name is not None

            pred_xyxy = evaluation_result.output_box
            if self._rasterize_bbox and pred_xyxy is not None:
                pred_xyxy = bbox_rasterize(pred_xyxy)

            # === Force full-length + dummy ===
            full_idx, full_xyxy, full_xywh, full_time, full_conf = _expand_to_full_xyxy_time_conf(evaluation_result, pred_xyxy)

            # Metrics with full-length predictions
            metrics, frames_iou = compute_one_pass_evaluation_metrics(
                evaluation_result.sequence_info.dataset_name,
                full_xyxy,
                evaluation_result.groundtruth_box,
                evaluation_result.groundtruth_object_existence_flag,
                full_time,
                self._compatibility_helper
            )

            print(f'{evaluation_result.sequence_info.sequence_name}: success {metrics.success_score:.04f}, prec {metrics.precision_score:.04f}, norm_pre {metrics.normalized_precision_score:.04f}')

            repeat_index = evaluation_progress.repeat_index
            if evaluation_progress.this_dataset.total_repeat_times == 1:
                repeat_index = None
            cache_key = evaluation_result.sequence_info.dataset_full_name, repeat_index
            if cache_key not in self._known_tracks_metric_cache:
                dataset_metrics_list_builder = DatasetOPEMetricsListBuilder()
                self._known_tracks_metric_cache[cache_key] = dataset_metrics_list_builder
            else:
                dataset_metrics_list_builder = self._known_tracks_metric_cache[cache_key]
            dataset_metrics_list_builder.append(evaluation_result.sequence_info.sequence_name, metrics)

            if folder_writer is not None:
                # Write CSV/PKL (report_gen) with full-length arrays
                dump_sequence_tracking_results_with_groundtruth(
                    folder_writer,
                    tracker_name, repeat_index,
                    evaluation_result.sequence_info.dataset_full_name,
                    evaluation_result.sequence_info.sequence_name,
                    full_idx,
                    full_conf,
                    full_xyxy,  # report_gen converts to xywh for CSV, but keeps full length
                    evaluation_result.groundtruth_object_existence_flag,
                    evaluation_result.groundtruth_box,
                    full_time,
                    frames_iou
                )
                generate_sequence_one_pass_evaluation_report(
                    folder_writer,
                    tracker_name, repeat_index,
                    evaluation_result.sequence_info.dataset_full_name,
                    evaluation_result.sequence_info.sequence_name,
                    metrics
                )
                # And explicitly write {sequence}.txt (space-separated, full length, includes dummy)
                _write_sequence_txt_variants(
                    folder_writer,
                    tracker_name, repeat_index,
                    evaluation_result.sequence_info.dataset_full_name,
                    evaluation_result.sequence_info.sequence_name,
                    full_xywh
                )

            if evaluation_progress.this_dataset.this_repeat_all_evaluated:
                dataset_metrics_list = dataset_metrics_list_builder.build()
                dataset_metrics_list = dataset_metrics_list.sort_by_sequence_name()

                del self._known_tracks_metric_cache[cache_key]

                dataset_summary_metrics = dataset_metrics_list.get_mean()
                if folder_writer is not None:
                    generate_dataset_one_pass_evaluation_report(
                        folder_writer,
                        tracker_name, repeat_index,
                        evaluation_result.sequence_info.dataset_full_name,
                        dataset_metrics_list,
                        dataset_summary_metrics
                    )
                self._final_summary_report_generator.add(evaluation_result.sequence_info.dataset_full_name, repeat_index, dataset_summary_metrics)
                summary_metrics.update(
                    _generate_dataset_summary_metrics_name_value_pair(
                        evaluation_result.sequence_info.dataset_full_name,
                        repeat_index,
                        dataset_summary_metrics
                    )
                )
                if evaluation_progress.this_dataset.total_repeat_times > 1:
                    if evaluation_result.sequence_info.dataset_full_name not in self._multi_run_dataset_metrics_cache:
                        dataset_all_runs_metrics = []
                        self._multi_run_dataset_metrics_cache[evaluation_result.sequence_info.dataset_full_name] = dataset_all_runs_metrics
                    else:
                        dataset_all_runs_metrics = self._multi_run_dataset_metrics_cache[evaluation_result.sequence_info.dataset_full_name]
                    dataset_all_runs_metrics.append(dataset_summary_metrics)

                    if evaluation_progress.this_dataset.all_evaluated:
                        dataset_all_runs_mean_metrics = compute_OPE_metrics_mean(dataset_all_runs_metrics)
                        del self._multi_run_dataset_metrics_cache[evaluation_result.sequence_info.dataset_full_name]
                        summary_metrics.update(
                            _generate_dataset_summary_metrics_name_value_pair(
                                evaluation_result.sequence_info.dataset_full_name, None, dataset_all_runs_mean_metrics
                            )
                        )
                        self._final_summary_report_generator.add(evaluation_result.sequence_info.dataset_full_name, None, dataset_all_runs_mean_metrics)

        return summary_metrics

    def finalize(self, tracker_name: str, folder_writer: Optional[FolderWriter]) -> Optional[Mapping[str, float]]:
        if folder_writer is not None:
            self._final_summary_report_generator.dump(folder_writer, tracker_name)
        return None


class EvaluationResultPersistenceWithOPEMetrics_LiveFeed:
    def __init__(self, rasterize_bbox: bool):
        self._metric_cache: MutableMapping[Tuple[str, int], DatasetOPEMetricsListBuilder] = {}
        self._compatibility_helper = ExternalToolkitCompatibilityHelper()
        self._rasterize_bbox = rasterize_bbox

    def __call__(self,
                 tracker_name: str, folder_writer: Optional[FolderWriter],
                 evaluation_results: Sequence[SequenceEvaluationResult_SOT],
                 evaluation_progresses: Sequence[EvaluationProgress]) -> Optional[Mapping[str, float]]:
        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            if evaluation_progress.this_dataset is not None:
                continue
            assert evaluation_result.sequence_info.dataset_full_name is not None
            assert evaluation_result.sequence_info.sequence_name is not None

            # BUGFIX: was "output.box" (typo) in original
            pred_xyxy = evaluation_result.output_box
            if self._rasterize_bbox and pred_xyxy is not None:
                pred_xyxy = bbox_rasterize(pred_xyxy)

            # Force full-length + dummy
            full_idx, full_xyxy, full_xywh, full_time, full_conf = _expand_to_full_xyxy_time_conf(evaluation_result, pred_xyxy)

            metrics, frames_iou = compute_one_pass_evaluation_metrics(
                evaluation_result.sequence_info.dataset_name,
                full_xyxy,
                evaluation_result.groundtruth_box,
                evaluation_result.groundtruth_object_existence_flag,
                full_time,
                self._compatibility_helper
            )

            if folder_writer is not None:
                dump_sequence_tracking_results_with_groundtruth(
                    folder_writer,
                    tracker_name, evaluation_progress.repeat_index,
                    evaluation_result.sequence_info.dataset_full_name,
                    evaluation_result.sequence_info.sequence_name,
                    full_idx,
                    full_conf,
                    full_xyxy,
                    evaluation_result.groundtruth_object_existence_flag,
                    evaluation_result.groundtruth_box,
                    full_time,
                    frames_iou
                )
                generate_sequence_one_pass_evaluation_report(
                    folder_writer,
                    tracker_name, evaluation_progress.repeat_index,
                    evaluation_result.sequence_info.dataset_full_name,
                    evaluation_result.sequence_info.sequence_name,
                    metrics
                )
                _write_sequence_txt_variants(
                    folder_writer,
                    tracker_name, evaluation_progress.repeat_index,
                    evaluation_result.sequence_info.dataset_full_name,
                    evaluation_result.sequence_info.sequence_name,
                    full_xywh
                )

            metric_cache_key = evaluation_result.sequence_info.dataset_full_name, evaluation_progress.repeat_index
            if metric_cache_key not in self._metric_cache:
                self._metric_cache[metric_cache_key] = DatasetOPEMetricsListBuilder()
            self._metric_cache[metric_cache_key].append(evaluation_result.sequence_info.sequence_name, metrics)
        return None

    def finalize(self, tracker_name: str, folder_writer: Optional[FolderWriter]) -> Optional[Mapping[str, float]]:
        summary_metrics = {}

        all_dataset_summary_metrics = {}

        for (dataset_full_name, repeat_index), metrics_list_builder in self._metric_cache.items():
            metrics_list = metrics_list_builder.build()
            dataset_summary_metrics = metrics_list.get_mean()
            if folder_writer is not None:
                generate_dataset_one_pass_evaluation_report(
                    folder_writer, tracker_name, repeat_index,
                    dataset_full_name, metrics_list.sort_by_sequence_name(),
                    dataset_summary_metrics
                )
            if dataset_full_name not in all_dataset_summary_metrics:
                all_dataset_summary_metrics[dataset_full_name] = []
            all_dataset_summary_metrics[dataset_full_name].append(dataset_summary_metrics)

        final_summary_report_generator = FinalOPEMetricsSummaryReportGenerator()

        for dataset_full_name, dataset_summary_metrics_list in all_dataset_summary_metrics.items():
            for repeat_index, dataset_summary_metrics in enumerate(dataset_summary_metrics_list):
                summary_metrics.update(_generate_dataset_summary_metrics_name_value_pair(dataset_full_name, repeat_index, dataset_summary_metrics))
                final_summary_report_generator.add(dataset_full_name, repeat_index, dataset_summary_metrics)
            if len(dataset_summary_metrics_list) > 1:
                dataset_multirun_averaged_metrics = compute_OPE_metrics_mean(dataset_summary_metrics_list)
            else:
                dataset_multirun_averaged_metrics = dataset_summary_metrics_list[0]
            summary_metrics.update(_generate_dataset_summary_metrics_name_value_pair(dataset_full_name, None, dataset_multirun_averaged_metrics))
            final_summary_report_generator.add(dataset_full_name, None, dataset_multirun_averaged_metrics)

        if folder_writer is not None:
            final_summary_report_generator.dump(folder_writer, tracker_name)

        return summary_metrics
