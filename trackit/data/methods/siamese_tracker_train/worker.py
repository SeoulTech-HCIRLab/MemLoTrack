import torch.utils.data
import numpy as np
import time
import concurrent.futures
from typing import Sequence, Optional
from trackit.core.runtime.metric_logger import get_current_local_metric_logger
from trackit.data.source import TrackingDataset
from trackit.data.context.worker import get_current_worker_info
from trackit.core.operator.numpy.bbox.format import bbox_cxcywh_to_xyxy
from trackit.data.utils.frame_decode import get_frame_decoder
from trackit.data.protocol.train_input import TrainData
from ... import HostDataPipeline
from ._types import  SOTFrameInfo, SiameseTrainingPairExtended
from .transform import SiameseTrackerTrain_DataTransform
from .siamese_training_pair_sampling import SamplingResult_Element, SiamFCTrainingPairSampler


def _decode(to_decode: SamplingResult_Element,
            datasets: Sequence[TrackingDataset],
            rng_engine: np.random.Generator,
            prefetch: bool) -> SOTFrameInfo:
    # 주어진 SamplingResult_Element를 디코딩하여 SOTFrameInfo 객체를 생성합니다.
    sequence = datasets[to_decode.dataset_index][to_decode.sequence_index]
    track = sequence.get_track_by_id(to_decode.track_id)
    frame = track[to_decode.frame_index]
    image_getter = get_frame_decoder(frame, prefetch)
    object_exists = frame.get_existence_flag()
    if object_exists:
        object_bbox = frame.get_bounding_box().astype(np.float64)
    else:
        object_bbox = rng_engine.random(4, dtype=np.float64)
        object_bbox = bbox_cxcywh_to_xyxy(object_bbox)
        object_bbox *= np.repeat(frame.get_frame_size(), 2)
    return SOTFrameInfo(image_getter, object_bbox, object_exists, sequence, track, frame)


def _decode_with_cache(name: str,
                       to_decode: SamplingResult_Element,
                       datasets: Sequence[TrackingDataset],
                       cache: dict,
                       result: dict,
                       rng_engine: np.random.Generator,
                       prefetch: bool) -> None:
    # 동일한 SamplingResult_Element에 대해 중복 디코딩을 피하기 위해 캐시를 사용합니다.
    if to_decode not in cache:
        cache[to_decode] = _decode(to_decode, datasets, rng_engine, prefetch)
    result[name] = cache[to_decode]


def _prepare_siamese_training_pair(global_job_index: int,
                                   batch_element_index: int,
                                   sampler_index: Optional[int],
                                   datasets: Sequence[TrackingDataset],
                                   siamese_training_pair_sampler: SiamFCTrainingPairSampler,
                                   rng_engine: np.random.Generator,
                                   prefetch: bool):
    # 샘플러로부터 받은 training_pair 정보를 바탕으로 DET와 SOT 모드를 분기하여 디코딩합니다.
    training_pair = siamese_training_pair_sampler(sampler_index, rng_engine)

    dataset_index = training_pair.z.dataset_index
    sequence = datasets[dataset_index][training_pair.z.sequence_index]
    track = sequence.get_track_by_id(training_pair.z.track_id)
    track_length = len(track)

    cache = {}
    result = {}

    # DET 모드: 단일 프레임인 경우
    if track_length == 1:
        # print(f"[Sampling Debug] DET mode: using frame {training_pair.z.frame_index} for both z and x.")
        _decode_with_cache('z', training_pair.z, datasets, cache, result, rng_engine, prefetch)
        _decode_with_cache('x', training_pair.x, datasets, cache, result, rng_engine, prefetch)
        decoded_pair = SiameseTrainingPairExtended(
            is_positive=training_pair.is_positive,
            template=result['z'],
            search=result['x'],
            memory_frames=[]
        )
        # print(f"[DET Worker Debug] z_bbox={decoded_pair.template.object_bbox}, x_bbox={decoded_pair.search.object_bbox}, memory_bboxes={[f.object_bbox for f in decoded_pair.memory_frames]}")
        return global_job_index, batch_element_index, (decoded_pair, False)

     # SOT 모드: sampler가 제공한 z/x 인덱스를 그대로 사용 → object_exists=True 보장
    _decode_with_cache('z', training_pair.z, datasets, cache, result, rng_engine, prefetch)
    _decode_with_cache('x', training_pair.x, datasets, cache, result, rng_engine, prefetch)

    # memory_frames는 이미 SiamFCTrainingPairSampler 에서 적절히 뽑아두었으므로 그대로 디코딩
    decoded_memory_frames = []
    for mem in training_pair.memory_frames or []:
        key = f"mem_{mem.frame_index}"
        _decode_with_cache(key, mem, datasets, cache, result, rng_engine, prefetch)
        decoded_memory_frames.append(result[key])

    decoded_pair = SiameseTrainingPairExtended(
        is_positive=training_pair.is_positive,
        template=result['z'],
        search=result['x'],
        memory_frames=decoded_memory_frames
    )
    # MemLoTrack BASE와 동일하게 z/x가 valid하지 않으면 drop
    if decoded_pair.is_positive:
        assert decoded_pair.template.object_exists
        assert decoded_pair.search.object_exists
    return global_job_index, batch_element_index, (decoded_pair, True)


class SiameseTrackerTrainingDataWorker(torch.utils.data.Dataset):
    def __init__(self, datasets: Sequence[TrackingDataset],
                 num_samples_per_epoch: int, batch_size: int,
                 siamese_training_pair_generator: SiamFCTrainingPairSampler,
                 data_transform: SiameseTrackerTrain_DataTransform,
                 num_io_threads: int):
        self.datasets = datasets
        self.num_samples_per_epoch = num_samples_per_epoch
        self.batch_size = batch_size
        self.siamese_training_pair_generator = siamese_training_pair_generator
        self.num_io_threads = num_io_threads
        self.background_io_threads: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.transform = data_transform

    def worker_init(self):
        rng_seed = get_current_worker_info().rng_seed
        if self.num_io_threads > 0:
            self.background_io_threads = concurrent.futures.ThreadPoolExecutor(self.num_io_threads)
        seeds = rng_seed.spawn(self.batch_size)
        self.batch_element_rng = tuple(np.random.default_rng(seed) for seed in seeds)
        self.transform_rng = np.random.default_rng(rng_seed.spawn(1)[0].generate_state(1).item())

    def worker_shutdown(self):
        if self.num_io_threads > 0:
            self.background_io_threads.shutdown()
        self.background_io_threads = None
        self.batch_element_rng = None
        self.transform_rng = None

    def __getitems__(self, job_indices: Sequence[int]):
        if self.background_io_threads is None:
            self.worker_init()

        if self.num_io_threads > 0:
            io_wait_time = 0
            begin_time = time.perf_counter()
            jobs = tuple(self.background_io_threads.submit(
                        _prepare_siamese_training_pair,
                        job_index, batch_element_index, job_index,
                        self.datasets, self.siamese_training_pair_generator,
                        self.batch_element_rng[batch_element_index], True)
                        for batch_element_index, job_index in enumerate(job_indices))
            batch = {}
            while len(jobs) > 0:
                io_begin_time = time.perf_counter()
                done_jobs, unfinished_jobs = concurrent.futures.wait(jobs, return_when=concurrent.futures.FIRST_COMPLETED)
                io_wait_time += time.perf_counter() - io_begin_time
                for job_future in done_jobs:
                    job_index, batch_element_index, pair_with_flag = job_future.result()
                    training_pair, use_memory = pair_with_flag
                    if training_pair is None:
                        # print(f"[DEBUG] Resubmitting job at index {job_index} because training_pair is None.")
                        job = self.background_io_threads.submit(
                            _prepare_siamese_training_pair,
                            job_index, batch_element_index, None,
                            self.datasets, self.siamese_training_pair_generator,
                            self.batch_element_rng[batch_element_index], True)
                        unfinished_jobs = list(unfinished_jobs)
                        unfinished_jobs.append(job)
                    else:
                        data = self.transform(training_pair, self.transform_rng)
                        if data is None:
                            job = self.background_io_threads.submit(
                                _prepare_siamese_training_pair,
                                job_index, batch_element_index, None,
                                self.datasets, self.siamese_training_pair_generator,
                                self.batch_element_rng[batch_element_index], True)
                            unfinished_jobs = list(unfinished_jobs)
                            unfinished_jobs.append(job)
                        else:
                            data['use_memory'] = use_memory
                            batch[job_index] = data
                jobs = unfinished_jobs
            batch = tuple(batch[index] for index in sorted(batch.keys()))
            total_time = time.perf_counter() - begin_time
            io_wait = io_wait_time / total_time
            return batch, io_wait
        else:
            batch = []
            for batch_element_index, job_index in enumerate(job_indices):
                pair_with_flag = _prepare_siamese_training_pair(
                                    job_index, batch_element_index, job_index,
                                    self.datasets, self.siamese_training_pair_generator,
                                    self.batch_element_rng[batch_element_index], False)[2]
                training_pair, use_memory = pair_with_flag
                data = self.transform(training_pair, self.transform_rng)
                while data is None:
                    pair_with_flag = _prepare_siamese_training_pair(
                                        job_index, batch_element_index, None,
                                        self.datasets, self.siamese_training_pair_generator,
                                        self.batch_element_rng[batch_element_index], False)[2]
                    training_pair, use_memory = pair_with_flag
                    data = self.transform(training_pair, self.transform_rng)
                data['use_memory'] = use_memory
                batch.append(data)
            return batch, None

    def __len__(self):
        return self.num_samples_per_epoch


class SiameseTrackerTrainingDataCollator:
    def __init__(self, transform_data_collator):
        self.transform_data_collator = transform_data_collator

    def __call__(self, data):
        batch, io_wait = data
        collated = TrainData()
        self.transform_data_collator(batch, collated)

        if len(batch) > 0:
            collated.input.update({'use_memory': batch[0].get('use_memory', False)})
            # print(f"[Collator Debug] use_memory flag: {batch[0].get('use_memory', False)}")
            # print(f"[FLAG_DBG] use_memory      = {collated.input['use_memory']}")
            
            if 'task_id' in batch[0]:
                task_ids = torch.tensor([sample['task_id'] for sample in batch],
                                        device=next(iter(collated.input.values())).device)
                collated.input.update({'task_ids': task_ids})
                # print("[Collator Debug] 'task_ids' added to collated input.")
            else:
                batch_size = len(batch)
                device = next(iter(collated.input.values())).device
                collated.input.update({'task_ids': torch.zeros(batch_size, dtype=torch.int, device=device)})
                # print("[Collator Debug] 'task_ids' not found; defaulting to zeros.")

            if 'frame_index' in batch[0]:
                frame_indices = torch.tensor([sample['frame_index'] for sample in batch],
                                            device=next(iter(collated.input.values())).device)
                collated.input.update({'frame_indices': frame_indices})
                # print("[Collator Debug] 'frame_indices' added to collated input.")
            else:
                batch_size = len(batch)
                device = next(iter(collated.input.values())).device
                collated.input.update({'frame_indices': torch.zeros(batch_size, dtype=torch.int, device=device)})
                # print("[Collator Debug] 'frame_indices' not found; defaulting to zeros.")

        else:
            collated.input.update({'use_memory': False})

        if io_wait is not None:
            collated.miscellanies['io_wait'] = io_wait
            # print("[Collator Debug] io_wait time:", io_wait)
        return collated


class SiameseTrackerTrainingHostLoggingHook(HostDataPipeline):
    def __init__(self, num_io_threads: int):
        self._num_io_threads = num_io_threads

    def on_epoch_begin(self):
        if self._num_io_threads > 0:
            get_current_local_metric_logger().set_metric_format('io_wait', no_prefix=True)

    def pre_process(self, input_data: TrainData) -> TrainData:
        if 'io_wait' in input_data.miscellanies:
            get_current_local_metric_logger().log({'io_wait': input_data.miscellanies['io_wait']})
        return input_data
