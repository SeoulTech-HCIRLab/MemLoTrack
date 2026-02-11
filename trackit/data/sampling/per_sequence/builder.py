# trackit/data/sampling/per_sequence/builder.py
from typing import Sequence, Tuple
import numpy as np
from trackit.data.source import TrackingDataset
from trackit.core.runtime.build_context import BuildContext
from . import RandomAccessiblePerSequenceSampler

def build_per_sequence_sampler(datasets: Sequence[TrackingDataset], sequence_picking_config: dict, build_context: BuildContext) -> Tuple[RandomAccessiblePerSequenceSampler, np.ndarray]:
    if sequence_picking_config['type'] == 'random':
        from .random.builder import build_random_sequence_picker
        return build_random_sequence_picker(datasets, sequence_picking_config, build_context)
    elif sequence_picking_config['type'] == 'intra_sequence':
        # 여기서는 하나의 시퀀스만 사용하도록 선택합니다.
        # 예를 들어, 첫 번째 데이터셋의 첫 시퀀스를 선택합니다.
        selected_sequence = datasets[0][0]
        total_size = sequence_picking_config['samples_per_epoch']
        from .intra_sequence_sampler import IntraSequenceSampler
        sampler = IntraSequenceSampler(selected_sequence, total_size)
        # 전체 데이터셋에 대해 균일한 sampling weight를 반환합니다.
        sampling_weight = np.ones(len(datasets), dtype=np.float64) / len(datasets)
        return sampler, sampling_weight
    else:
        raise NotImplementedError('Unknown sequence picker type: {}'.format(sequence_picking_config['type']))
