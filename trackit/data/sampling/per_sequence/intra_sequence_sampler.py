# trackit/data/sampling/per_sequence/intra_sequence_sampler.py
from typing import Tuple, Iterator

class IntraSequenceSampler:
    def __init__(self, sequence: object, total_samples: int):
        """
        sequence: 하나의 시퀀스(비디오) 객체, 예를 들어 datasets[0][0] (고정)
        total_samples: 에폭 당 총 샘플 수
        """
        self.sequence = sequence
        self.total_samples = total_samples
        # 선택한 시퀀스에 대한 dataset index와 sequence index (여기서는 고정으로 0,0 사용)
        self.dataset_index = 0
        self.sequence_index = 0

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Tuple[int, int]:

        return self.dataset_index, self.sequence_index

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        for i in range(self.total_samples):
            yield self.__getitem__(i)
