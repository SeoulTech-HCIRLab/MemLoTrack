from typing import Optional, Mapping, Sequence, Tuple
import itertools
from dataclasses import dataclass
from tabulate import tabulate
import numpy as np  # ★ added

from trackit.miscellanies.torch.distributed import is_main_process
from trackit.core.runtime.metric_logger import get_current_metric_logger, get_current_local_metric_logger
from trackit.core.runtime.context.epoch import get_current_epoch_context
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from trackit.data.utils.data_source_matcher import DataSourceMatcher
from trackit.data import HostDataPipeline
from trackit.data.context.variable.eval import DatasetEvaluationTask

from .progress_tracer.predefined import EvaluationTaskTracer_Predefined
from .progress_tracer.plain import EvaluationTaskTracer_Plain
from .handler import EvaluationResultHandlerAsyncWrapper
from .progress_tracer import EvaluationTaskTracer


def print_metrics(metrics: Sequence[Tuple[str, float]], log_as_general_summary: bool):
    if len(metrics) == 0:
        return
    metrics = dict(metrics)
    if log_as_general_summary:
        get_current_metric_logger().log_summary(metrics)
    else:
        get_current_local_metric_logger().log_summary(metrics)
        get_current_metric_logger().log(external=metrics)
    print(tabulate(metrics.items(), headers=('metric', 'value'), floatfmt=".4f"), flush=True)


# ===== Helper functions to expand to full sequence length with dummy bboxes =====

def _extract_length_from_sequence_info(si) -> Optional[int]:
    """
    SequenceInfo의 '총 프레임 수' 필드를 이름 모를 때도 찾아냅니다.
    1) 흔한 필드명 우선 검색
    2) NamedTuple._fields를 통해 int 타입 필드 후보 중 가장 큰 값 선택
    3) 시퀀스 객체가 순회 가능하면 int 항목 중 가장 큰 값 선택
    """
    # 1) 흔한 이름 우선
    for name in ('sequence_length', 'track_length', 'length', 'num_frames', 'frame_count', 'nframes'):
        if hasattr(si, name):
            v = getattr(si, name)
            if isinstance(v, (int, np.integer)) and v > 0:
                return int(v)

    # 2) NamedTuple 필드 스캔
    if hasattr(si, '_fields'):
        ints = []
        for name in getattr(si, '_fields'):
            try:
                v = getattr(si, name)
            except Exception:
                continue
            if isinstance(v, (int, np.integer)) and v > 0:
                ints.append(int(v))
        if len(ints) > 0:
            return int(max(ints))

    # 3) 순회 가능시 항목 스캔
    try:
        ints = [int(v) for v in si if isinstance(v, (int, np.integer)) and int(v) > 0]
        if len(ints) > 0:
            return int(max(ints))
    except TypeError:
        pass

    return None


def _infer_total_length_from_result(e: SequenceEvaluationResult_SOT) -> int:
    # SequenceInfo에서 먼저 시도 (가장 신뢰도 높음: worker에서 len(track) 넣어줌)
    si_len = _extract_length_from_sequence_info(e.sequence_info)
    if si_len is not None:
        return si_len

    # GT 길이로 추정
    if e.groundtruth_object_existence_flag is not None:
        return int(len(e.groundtruth_object_existence_flag))
    if e.groundtruth_box is not None:
        return int(len(e.groundtruth_box))

    # 마지막으로 평가된 인덱스 범위로 추정
    idx = np.asarray(e.evaluated_frame_indices, dtype=int)
    return int(idx.max()) + 1 if idx.size > 0 else 0


def _expand_eval_result_to_full_length(e: SequenceEvaluationResult_SOT) -> SequenceEvaluationResult_SOT:
    total = _infer_total_length_from_result(e)
    full_idx = np.arange(total, dtype=int)

    # --- output_box (XYXY) expand to full length ---
    pred_xyxy = e.output_box
    idx = np.asarray(e.evaluated_frame_indices, dtype=int)
    full_box = np.zeros((total, 4), dtype=(pred_xyxy.dtype if pred_xyxy is not None else np.float32))
    if pred_xyxy is not None and pred_xyxy.size > 0 and idx.size > 0:
        full_box[idx] = pred_xyxy[:len(idx)]

    # 객체 미존재 프레임은 무조건 0 0 0 0 (dummy)
    flag = e.groundtruth_object_existence_flag
    if flag is not None and len(flag) == total:
        exist = np.asarray(flag, dtype=bool)
        full_box[~exist] = 0.0

    # --- output_confidence expand ---
    conf = e.output_confidence
    full_conf = np.zeros((total,), dtype=(conf.dtype if conf is not None else np.float32))
    if conf is not None and np.size(conf) > 0 and idx.size > 0:
        full_conf[idx] = np.asarray(conf).reshape(-1)[:len(idx)]

    # --- time_cost expand ---
    tc = e.time_cost
    full_time = np.zeros((total,), dtype=np.float32)
    if tc is not None and np.size(tc) > 0 and idx.size > 0:
        full_time[idx] = np.asarray(tc, dtype=np.float32).reshape(-1)[:len(idx)]

    # --- batch_size expand (형태 맞춤) ---
    bs = e.batch_size
    full_bs = np.zeros((total,), dtype=(bs.dtype if bs is not None else np.int32))
    if bs is not None and np.size(bs) > 0 and idx.size > 0:
        full_bs[idx] = np.asarray(bs).reshape(-1)[:len(idx)]

    # 새 NamedTuple로 교체(전 구간 인덱스/출력으로 정규화)
    return SequenceEvaluationResult_SOT(
        id=e.id,
        sequence_info=e.sequence_info,
        evaluated_frame_indices=full_idx,
        groundtruth_box=e.groundtruth_box,
        groundtruth_object_existence_flag=e.groundtruth_object_existence_flag,
        groundtruth_mask=e.groundtruth_mask,
        output_box=full_box,
        output_confidence=full_conf,
        output_mask=e.output_mask,
        time_cost=full_time,
        batch_size=full_bs,
    )


class EvaluationResultCollector:
    def __init__(self, progress_tracer: EvaluationTaskTracer,
                 dispatcher: Mapping[DataSourceMatcher, Sequence[EvaluationResultHandlerAsyncWrapper]],
                 log_summary: bool):
        self._progress_tracer = progress_tracer
        self._dispatcher = dispatcher
        self._log_summary = log_summary

    def collect(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT]):
        # ★ 상류에서 전 프레임 길이로 강제 확장 + dummy 채움 (모든 핸들러에 일괄 적용)
        evaluation_results = tuple(_expand_eval_result_to_full_length(e) for e in evaluation_results)

        progresses = tuple(self._progress_tracer.submit(
            evaluation_result.sequence_info.dataset_full_name,
            evaluation_result.sequence_info.sequence_name
        ) for evaluation_result in evaluation_results)

        last_remaining_evaluation_results = evaluation_results
        last_remaining_progresses = progresses

        rest_evaluation_results = []
        rest_progresses = []

        for data_source_matcher, handlers in self._dispatcher.items():
            this_handler_evaluation_results = []
            this_handler_evaluation_progresses = []
            for evaluation_result, progress in zip(last_remaining_evaluation_results, last_remaining_progresses):
                if data_source_matcher(evaluation_result.sequence_info.dataset_name, evaluation_result.sequence_info.data_split):
                    this_handler_evaluation_results.append(evaluation_result)
                    this_handler_evaluation_progresses.append(progress)
                else:
                    rest_evaluation_results.append(evaluation_result)
                    rest_progresses.append(progress)

            if len(this_handler_evaluation_results) > 0:
                for handler in handlers:
                    handler.accept(this_handler_evaluation_results, this_handler_evaluation_progresses)
            last_remaining_evaluation_results = rest_evaluation_results
            last_remaining_progresses = rest_progresses
            rest_evaluation_results = []
            rest_progresses = []

        assert len(last_remaining_evaluation_results) == 0, "Some evaluation results are not handled."

    def get_metrics(self) -> Sequence[Tuple[str, float]]:
        metrics = []
        for handlers in self._dispatcher.values():
            for handler in handlers:
                this_handler_metrics = handler.get_unseen_metrics()
                if this_handler_metrics is not None:
                    metrics.extend(this_handler_metrics)
        return metrics

    def finalize(self):
        for handlers in self._dispatcher.values():
            for handler in handlers:
                handler.close()


@dataclass(frozen=True)
class SubHandlerBuildOptions:
    handler_type: str
    file_name: Optional[str]
    bbox_rasterize: bool


class EvaluationResultCollector_RuntimeIntegration(HostDataPipeline):
    def __init__(self, tracker_name: str, handler_build_options: Mapping[DataSourceMatcher, Sequence[SubHandlerBuildOptions]],
                 optional_predefined_evaluation_tasks: Optional[Sequence[DatasetEvaluationTask]], run_async: bool, log_summary: bool):
        tracker_name = tracker_name.replace('/', '-')
        tracker_name = tracker_name.replace('\\', '-')
        if is_main_process():
            self._predefined_evaluation_tasks = optional_predefined_evaluation_tasks
            self._tracker_name = tracker_name
            self._handler_build_options = handler_build_options
            self._run_async = run_async
            self._log_summary = log_summary

    def on_epoch_begin(self):
        if is_main_process():
            output_path = get_current_epoch_context().get_current_epoch_output_path()
            if self._predefined_evaluation_tasks is not None:
                progress_tracer = EvaluationTaskTracer_Predefined(self._predefined_evaluation_tasks)
            else:
                progress_tracer = EvaluationTaskTracer_Plain()
            dispatcher = {}
            for data_source_matcher, handler_build_options in self._handler_build_options.items():
                handlers = []
                for handler_build_option in handler_build_options:
                    handler_cls = None
                    if handler_build_option.handler_type == 'plain':
                        if output_path is not None or handler_build_option.file_name is not None:
                            from .handler.persistence import EvaluationResultPersistence
                            handler_cls = EvaluationResultPersistence
                    elif handler_build_option.handler_type == 'one_pass_evaluation':
                        from .handler.one_pass_evaluation import EvaluationResultPersistenceWithOPEMetrics
                        handler_cls = EvaluationResultPersistenceWithOPEMetrics
                    elif handler_build_option.handler_type == 'one_pass_evaluation_compatible':
                        from .handler.one_pass_evaluation_compatible import EvaluationResultPersistenceWithOPEMetricsCompatibleWithExternalTools
                        handler_cls = EvaluationResultPersistenceWithOPEMetricsCompatibleWithExternalTools
                    elif handler_build_option.handler_type == 'external/GOT10k':
                        if output_path is not None:
                            from .handler.external_adaptors.got10k import GOT10KEvaluationToolAdaptor
                            handler_cls = GOT10KEvaluationToolAdaptor
                    elif handler_build_option.handler_type == 'external/OTB':
                        if output_path is not None:
                            from .handler.external_adaptors.otb import OTBEvaluationToolAdaptor
                            handler_cls = OTBEvaluationToolAdaptor
                    elif handler_build_option.handler_type == 'external/PyTracking':
                        if output_path is not None:
                            from .handler.external_adaptors.pytracking import PyTrackingEvaluationToolAdaptor
                            handler_cls = PyTrackingEvaluationToolAdaptor
                    elif handler_build_option.handler_type == 'external/TrackingNet':
                        if output_path is not None:
                            from .handler.external_adaptors.trackingnet import TrackingNetEvaluationToolAdaptor
                            handler_cls = TrackingNetEvaluationToolAdaptor
                    else:
                        raise ValueError(f'Unknown handler type: {handler_build_option.handler_type}')
                    if handler_cls is not None:
                        handler = EvaluationResultHandlerAsyncWrapper(
                            handler_cls,
                            (self._tracker_name, output_path, handler_build_option.file_name, handler_build_option.bbox_rasterize),
                            self._run_async
                        )
                        handlers.append(handler)
                dispatcher[data_source_matcher] = handlers
            self._collector = EvaluationResultCollector(progress_tracer, dispatcher, self._log_summary)
            self._duplication_check = set()

        self._local_cache = []

    def post_process(self, output_data: Optional[dict]):
        if output_data is not None:
            self._local_cache.extend(output_data['evaluated_sequences'])
        if is_main_process():
            print_metrics(self._collector.get_metrics(), self._log_summary)
        return output_data

    def distributed_prepare_gathering(self) -> Sequence[SequenceEvaluationResult_SOT]:
        cached_sequences = self._local_cache
        self._local_cache = []
        return cached_sequences

    def distributed_on_gathered(self, evaluation_results_on_all_nodes: Sequence[Sequence[SequenceEvaluationResult_SOT]]) -> None:
        if is_main_process():
            evaluation_results = tuple(itertools.chain.from_iterable(evaluation_results_on_all_nodes))
            if len(evaluation_results) > 0:
                for evaluation_result in evaluation_results:
                    assert evaluation_result.id not in self._duplication_check
                    self._duplication_check.add(evaluation_result.id)
                self._collector.collect(evaluation_results)

    def on_epoch_end(self):
        assert len(self._local_cache) == 0
        if is_main_process():
            self._collector.finalize()
            print_metrics(self._collector.get_metrics(), self._log_summary)
            del self._collector
            del self._duplication_check
