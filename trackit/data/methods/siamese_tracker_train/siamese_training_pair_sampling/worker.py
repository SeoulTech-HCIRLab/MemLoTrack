import numpy as np
import torch
import torch.nn as nn

from trackit.data.methods.siamese_tracker_train.siamese_training_pair_sampling._algos import (
    _get_random_track,
    get_random_positive_siamese_training_pair_from_track
)
from ._types import SiameseTrainingPairSamplingResult, SamplingResult_Element


class SiamFCTrainingPairSampler:
    def __init__(
        self,
        datasets: list,
        dataset_weights: np.ndarray,
        sequence_picker,
        siamese_sampling_frame_range: int,
        siamese_sampling_method,
        siamese_sampling_frame_range_auto_extend_step: int,
        siamese_sampling_frame_range_auto_extend_max_retry_count: int,
        siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found: bool,
        negative_sample_weight: float,
        negative_sample_generation_methods: list,
        negative_sample_generation_method_weights: np.ndarray,
        force_sequential_template: bool = True
    ):
        self.datasets = datasets
        self.dataset_weights = dataset_weights / dataset_weights.sum()
        self.sequence_picker = sequence_picker
        self.siamese_sampling_frame_range = siamese_sampling_frame_range
        self.siamese_sampling_method = siamese_sampling_method
        self.siamese_sampling_frame_range_auto_extend_step = siamese_sampling_frame_range_auto_extend_step
        self.siamese_sampling_frame_range_auto_extend_max_retry_count = siamese_sampling_frame_range_auto_extend_max_retry_count
        self.siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found = siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found
        self.negative_sample_weight = negative_sample_weight
        self.negative_sample_generation_methods = negative_sample_generation_methods
        self.negative_sample_generation_method_weights = negative_sample_generation_method_weights
        self.force_sequential_template = force_sequential_template
        # Minimum interval between z and x
        self.min_interval = 7

    def __call__(self, index: int, rng_engine: np.random.Generator) -> SiameseTrainingPairSamplingResult:
        # always positive
        is_positive = True

        # Select sequence
        dataset_index, sequence_index, _ = self.sequence_picker[index]
        dataset = self.datasets[int(dataset_index)]
        sequence = dataset[int(sequence_index)]

        # Select track
        track = _get_random_track(sequence, rng_engine)
        length = len(track)
        obj_id = track.get_object_id() or 0

        # DET Mode
        if length == 1:
            return SiameseTrainingPairSamplingResult(
                z=SamplingResult_Element(dataset_index, sequence_index, obj_id, 0),
                x=SamplingResult_Element(dataset_index, sequence_index, obj_id, 0),
                is_positive=True,
                memory_frames=None
            )


        frame_indices = get_random_positive_siamese_training_pair_from_track(
            track,
            self.siamese_sampling_frame_range,
            self.siamese_sampling_method,
            rng_engine,
            self.siamese_sampling_frame_range_auto_extend_step,
            self.siamese_sampling_frame_range_auto_extend_max_retry_count,
            self.siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found
        )
        if isinstance(frame_indices, int) or len(frame_indices) == 1:
            idx = frame_indices if isinstance(frame_indices, int) else frame_indices[0]
            z_idx = x_idx = idx
        else:
            z_idx, x_idx = frame_indices


        candidates = np.arange(z_idx + 1, x_idx)
        num_mem = self.sequence_picker.num_frames - 2



        if len(candidates) >= num_mem:
            mem_idxs = sorted(rng_engine.choice(candidates, size=num_mem, replace=False).tolist())
        else:
            mem_idxs = list(candidates)
            pad = candidates[-1] if candidates.size else z_idx + 1
            pad = min(pad, length-1)
            while len(mem_idxs) < num_mem:
                mem_idxs.append(pad)

        
        # Generate result
        memory_frames = [
            SamplingResult_Element(dataset_index, sequence_index, obj_id, int(mi))
            for mi in mem_idxs
        ]
        return SiameseTrainingPairSamplingResult(
            z=SamplingResult_Element(dataset_index, sequence_index, obj_id, int(z_idx)),
            x=SamplingResult_Element(dataset_index, sequence_index, obj_id, int(x_idx)),
            is_positive=True,
            memory_frames=memory_frames
        )
