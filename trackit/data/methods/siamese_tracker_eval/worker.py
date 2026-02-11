import concurrent.futures
import torch.utils.data.dataset
from typing import Sequence, Optional
import time
import torch

from trackit.core.runtime.metric_logger import get_current_local_metric_logger
from trackit.data import HostDataPipeline
from trackit.data.sampling.per_frame.distributed_dynamic_eval_task_scheduling.distributed_evaluation_task_scheduler import DistributedTrackerEvaluationTaskDynamicSchedulerClient, EvaluationTask
from trackit.miscellanies.torch.distributed import get_rank
from trackit.data.source import TrackingDataset, TrackingDataset_FrameInTrack
from trackit.data.protocol.eval_input import TrackerEvalData, SequenceInfo
from trackit.data.utils.frame_decode import get_frame_decoder
from trackit.data.context.worker import get_current_worker_info

from . import SiameseTrackerEvalDataWorker_FrameContext, SiameseTrackerEvalDataWorker_Task
from .transform import SiameseTrackerEval_DataTransform

def _prepare_frame_context(frame: TrackingDataset_FrameInTrack):
    gt_bbox = frame.get_bounding_box() if frame.get_existence_flag() else None
    image_getter_fn = get_frame_decoder(frame)
    return SiameseTrackerEvalDataWorker_FrameContext(frame.get_frame_index(), image_getter_fn, gt_bbox)


# === 추가: 시퀀스 총 프레임 수 안전 추출기 ===

def _safe_sequence_total_length(sequence, track):
    # A) 트랙 내부 프레임의 원본 frame_index 최댓값+1 (중간 non-exist 포함 복원)
    try:
        n = len(track)
        if n > 0 and hasattr(track[0], 'get_frame_index'):
            max_idx = -1
            min_idx = 1 << 30
            for i in range(n):
                fi = track[i].get_frame_index()
                if isinstance(fi, int):
                    if fi > max_idx: max_idx = fi
                    if fi < min_idx: min_idx = fi
            if max_idx >= 0:
                total_len = (max_idx - (min_idx if min_idx >= 0 else 0)) + 1
                if total_len > n:
                    return int(total_len)
    except Exception:
        pass


def _prepare(batch_index: int, evaluation_task: EvaluationTask, datasets: Sequence[TrackingDataset]):
    dataset = datasets[evaluation_task.dataset_index]
    sequence = dataset[evaluation_task.sequence_index]
    track = sequence.get_track_by_id(evaluation_task.track_id)

    total_len = _safe_sequence_total_length(sequence, track)
    track_context = SequenceInfo(
        dataset.get_name(),
        dataset.get_data_split(),
        dataset.get_full_name(),
        track.get_name(),       # 기존 코드와 동일하게 track 이름을 sequence_name 자리에 사용
        total_len,              # ★ 시퀀스 '총' 프레임 수를 넣음
        sequence.get_fps()
    ) if evaluation_task.do_task_creation else None

    # (디버그) 실제 길이 확인
    # print(f"[DBG][SeqLen] {dataset.get_name()}::{sequence.get_name()}::{track.get_name()}  "
    #       f"seq_total={total_len}  track_len={len(track)}", flush=True)

    init_frame_context = _prepare_frame_context(track[evaluation_task.do_init_frame_index]) \
        if evaluation_task.do_init_frame_index is not None else None
    if init_frame_context is not None:
        assert init_frame_context.gt_bbox is not None, \
            'bbox must be provided for init frame. ' + \
            f"dataset: {dataset.get_name()}, sequence: {sequence.get_name()}, track: {track.get_name()}, frame: {evaluation_task.do_init_frame_index}"

    track_frame_context = _prepare_frame_context(track[evaluation_task.do_track_frame_index]) \
        if evaluation_task.do_track_frame_index is not None else None

    return batch_index, SiameseTrackerEvalDataWorker_Task(
        evaluation_task.task_index,
        track_context,
        init_frame_context,
        track_frame_context,
        evaluation_task.do_task_finalization
    )


class SiameseTrackEvaluationDataInputWorker(torch.utils.data.dataset.Dataset):
    def __init__(self, datasets: Sequence[TrackingDataset],
                 dynamic_task_scheduler: DistributedTrackerEvaluationTaskDynamicSchedulerClient,
                 processor: SiameseTrackerEval_DataTransform,
                 num_io_threads: int):
        self._datasets = datasets
        self._dynamic_task_scheduler = dynamic_task_scheduler
        self._rank = get_rank()
        self._background_io_threads: Optional[concurrent.futures.ThreadPoolExecutor] = None
        assert num_io_threads >= 0
        self._num_io_threads = num_io_threads
        self._processor = processor

    def worker_init(self):
        if self._num_io_threads > 0:
            self._background_io_threads = concurrent.futures.ThreadPoolExecutor(self._num_io_threads)

    def worker_shutdown(self):
        if self._num_io_threads > 0:
            self._background_io_threads.shutdown()
        self._background_io_threads = None

    def __getitem__(self, iteration: int):
        if self._background_io_threads is None:
            self.worker_init()
        miscellanies = {'local_worker_index': get_current_worker_info().worker_id}
        evaluation_tasks = self._dynamic_task_scheduler.get_next_batch(self._rank, iteration)
        if evaluation_tasks is None:
            return TrackerEvalData((), miscellanies)

        if self._num_io_threads > 0:
            io_wait_time = 0
            begin_time = time.perf_counter()
            jobs = tuple(self._background_io_threads.submit(_prepare, batch_index, evaluation_task, self._datasets)
                         for batch_index, evaluation_task in enumerate(evaluation_tasks))

            batch = {}
            while len(jobs) > 0:
                io_begin_time = time.perf_counter()
                done_jobs, unfinished_jobs = concurrent.futures.wait(jobs, return_when=concurrent.futures.FIRST_COMPLETED)
                io_wait_time += time.perf_counter() - io_begin_time
                for done_job in done_jobs:
                    batch_index, evaluation_task_step = done_job.result()
                    batch[batch_index] = self._processor(evaluation_task_step)
                jobs = unfinished_jobs
            batch = tuple(batch[index] for index in sorted(batch.keys()))
            total_time = time.perf_counter() - begin_time
            miscellanies['io_wait'] = io_wait_time / total_time
        else:
            batch = tuple(self._processor(_prepare(batch_index, evaluation_task, self._datasets)[1])
                          for batch_index, evaluation_task in enumerate(evaluation_tasks))

        return TrackerEvalData(batch, miscellanies)

    # def __len__(self):
    #     # 전체 evaluation task 수를 반환하도록 구현
    #     # 예: dynamic scheduler가 관리하는 총 task 수
    #     return self._dynamic_task_scheduler.get_number_of_tasks()


class SiameseTrackEvaluationHostLoggingHook(HostDataPipeline):
    def __init__(self, num_io_threads: int):
        self._num_io_threads = num_io_threads

    def on_epoch_begin(self):
        if self._num_io_threads > 0:
            get_current_local_metric_logger().set_metric_format('io_wait', no_prefix=True)

    def pre_process(self, input_data: Optional[TrackerEvalData]) -> Optional[TrackerEvalData]:
        if input_data is None:
            return None
        if 'io_wait' in input_data.miscellanies:
            get_current_local_metric_logger().log({'io_wait': input_data.miscellanies['io_wait']})
        return input_data
