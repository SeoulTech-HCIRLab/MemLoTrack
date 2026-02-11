from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List
import torch


class SiamesePairSamplingMethod(Enum):
    interval = auto()
    causal = auto()
    reverse_causal = auto()


class SiamesePairNegativeSamplingMethod(Enum):
    random = auto()
    random_semantic_object = auto()
    distractor = auto()


@dataclass(frozen=True)
class SamplingResult_Element:
    dataset_index: int
    sequence_index: int
    track_id: int
    frame_index: int


@dataclass(frozen=True)
class SiameseTrainingPairSamplingResult:
    z: SamplingResult_Element
    x: SamplingResult_Element
    is_positive: bool

    # SOT 모드일 경우 추가적으로 memory candidate 프레임들의 정보와 encoder attention mask를 전달
    memory_frames: Optional[List[SamplingResult_Element]] = None
    # attn_mask: Optional[torch.Tensor] = None