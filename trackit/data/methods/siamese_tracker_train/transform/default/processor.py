# trackit/data/methods/siamese_tracker_train/transform/default/processor.py

import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Mapping

import numpy as np
import torch

from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.operator.numpy.bbox.validity      import bbox_is_valid
from trackit.core.transforms.dataset_norm_stats     import get_dataset_norm_stats_transform
from trackit.core.utils.siamfc_cropping              import (
    prepare_siamfc_cropping_with_augmentation,
    apply_siamfc_cropping,
    apply_siamfc_cropping_to_boxes,
)
from trackit.data.protocol.train_input               import TrainData
from trackit.data.utils.collation_helper             import (
    collate_element_as_torch_tensor,
    collate_element_as_np_array,
)
from trackit.data                                  import HostDataPipeline
from trackit.core.runtime.metric_logger             import get_current_metric_logger

from ..._types import SiameseTrainingPair, SOTFrameInfo
from ..          import SiameseTrackerTrain_DataTransform
from .augmentation import AugmentationPipeline, AnnotatedImage
from .plugin       import ExtraTransform

import os
import sys
from PIL import Image

@dataclass(frozen=True)
class SiamFCCroppingParameter:
    output_size: np.ndarray
    area_factor: float
    scale_jitter_factor: float = 0.
    translation_jitter_factor: float = 0.
    output_min_object_size_in_pixel: np.ndarray = field(default_factory=lambda: np.array((0., 0.)))
    output_min_object_size_in_ratio: float = 0.
    output_max_object_size_in_pixel: np.ndarray = field(default_factory=lambda: np.array((float("inf"), float("inf"))))
    output_max_object_size_in_ratio: float = 1.
    interpolation_mode: str = 'bilinear'
    interpolation_align_corners: bool = False


class SiamTrackerTrainingPairProcessor(SiameseTrackerTrain_DataTransform):
    def __init__(
        self,
        template_param: SiamFCCroppingParameter,
        search_param:   SiamFCCroppingParameter,
        dynamic_pipeline: AugmentationPipeline,
        static_pipeline:  AugmentationPipeline,
        norm_stats_dataset_name: str,
        additional_processors: Optional[Sequence[ExtraTransform]] = None,
        visualize: bool = False,
        patch_size: Tuple[int, int] = None,
    ):
        # 1) SiamFC cropping 모듈
        self.template_crop = SiamFCCropping(template_param)
        self.search_crop   = SiamFCCropping(search_param)

        # 2) augmentation: SOT(dynamic) vs DET(static)
        self.dynamic_pipeline = dynamic_pipeline
        self.static_pipeline  = static_pipeline

        # 3) normalize
        self.image_normalize = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)

        self.additional_processors = additional_processors
        self.visualize            = visualize

        # memory 크롭용
        self.search_param = search_param
        self.patch_size   = patch_size


    def __call__(self, training_pair: SiameseTrainingPair, rng: np.random.Generator):
        # — 단계 1: context 초기화 —
        context = {
            'is_positive': training_pair.is_positive,
            'z_bbox':      training_pair.template.object_bbox.copy(),
            'x_bbox':      training_pair.search.object_bbox.copy(),
        }

        # — 단계 2: SiamFC cropping 준비 —
        assert self.template_crop.prepare('z', rng, context)
        if not self.search_crop.prepare('x', rng, context):
            return None

        # — 단계 3: 이미지 디코딩 (z, x) —
        cache = {}
        _decode_with_cache('z', training_pair.template, cache, context)
        _decode_with_cache('x', training_pair.search,  cache, context)
        is_same_image = (len(cache) == 1)
        del cache

        # — 단계 4: 실제 크롭 (z, x) —
        self.template_crop.do('z', context)
        self.search_crop.do('x', context)

        # — 단계 5: augmentation —
        aug_ctx = {
            'template':      [AnnotatedImage(context['z_cropped_image'], context['z_cropped_bbox'])],
            'search_region': [AnnotatedImage(context['x_cropped_image'], context['x_cropped_bbox'])]
        }


        # # ========== 시각화 디버깅 =========
        # # — 저장 디렉토리 준비 및 aug 전 저장 —
        # save_dir = "/home/park/Desktop/object_tracking/image"
        # os.makedirs(save_dir, exist_ok=True)
        # # z_pre
        # zp = (context['z_cropped_image'] * 255).byte().cpu().permute(1,2,0).numpy()
        # Image.fromarray(zp).save(os.path.join(save_dir, "z_pre.png"))
        # # x_pre
        # xp = (context['x_cropped_image'] * 255).byte().cpu().permute(1,2,0).numpy()
        # Image.fromarray(xp).save(os.path.join(save_dir, "x_pre.png"))
        # # — augmentation 전 이미지 저장 —


        if is_same_image:
            self.static_pipeline(aug_ctx, rng)
        else:
            self.dynamic_pipeline(aug_ctx, rng)

        tpl = aug_ctx['template'][0]
        sr  = aug_ctx['search_region'][0]
        context['z_cropped_image'], context['z_cropped_bbox'] = tpl.image, tpl.bbox
        context['x_cropped_image'], context['x_cropped_bbox'] = sr.image, sr.bbox

        # # ======== 시각화 디버깅 ========
        # # — aug 후 저장 —
        # # z_post
        # zp2 = (context['z_cropped_image'] * 255).byte().cpu().permute(1,2,0).numpy()
        # Image.fromarray(zp2).save(os.path.join(save_dir, "z_post.png"))
        # # x_post
        # xp2 = (context['x_cropped_image'] * 255).byte().cpu().permute(1,2,0).numpy()
        # Image.fromarray(xp2).save(os.path.join(save_dir, "x_post.png"))


        # — 단계 6: memory_frames 크롭 (augmentation 없음) —
        saved_cp = copy.deepcopy(context['x_cropping_parameter'])
        osz      = self.search_param.output_size
        mode     = self.search_param.interpolation_mode
        align    = self.search_param.interpolation_align_corners

        memory_patches = []
        # — MemLoTrack BASE에선 memory frame은 단순 cache 대상이므로,
        #    occlusion/out-of-view 시 object_exists=False인 프레임은 drop —
        for frame in getattr(training_pair, 'memory_frames', []):
            if not frame.object_exists:
                continue
            
            raw = frame.image()
            if raw.ndim == 2:
                raw = np.stack([raw]*3, -1)
            t = torch.from_numpy(raw).permute(2,0,1).float()
            cp_mem = copy.deepcopy(saved_cp)
            patch, _, _ = apply_siamfc_cropping(
                t, osz, cp_mem,
                interpolation_mode=mode,
                align_corners=align,
            )
            patch.div_(255.)
            self.image_normalize(patch)
            memory_patches.append(patch)

        # # =========시각화 디버깅 =========
        # # — memory frames 저장 —
        # for idx, patch in enumerate(memory_patches):
        #     pm = (patch * 255).byte().cpu().permute(1,2,0).numpy()
        #     Image.fromarray(pm).save(os.path.join(save_dir, f"mem_{idx}.png"))
        # # 일단 확인 후 종료
        # sys.exit(0)


        # — 단계 7: bbox clip → normalize —
        _bbox_clip_to_image_boundary_(context['z_cropped_bbox'], context['z_cropped_image'])
        _bbox_clip_to_image_boundary_(context['x_cropped_bbox'], context['x_cropped_image'])
        self.image_normalize(context['z_cropped_image'])
        self.image_normalize(context['x_cropped_image'])

        # — 단계 8: output dict 준비 —
        Hc, Wc = osz
        Gh = Hc // self.patch_size[1]
        Gw = Wc // self.patch_size[0]

        data = {
            'z_cropped_image': context['z_cropped_image'],
            'x_cropped_image': context['x_cropped_image'],
            'z_cropped_bbox':  torch.as_tensor(context['z_cropped_bbox'], dtype=torch.float32),
            'x_cropped_bbox':  torch.as_tensor(context['x_cropped_bbox'], dtype=torch.float32),
            'is_positive':     torch.as_tensor(context['is_positive'], dtype=torch.float32),
            'z':               context['z_cropped_image'],
            'x':               context['x_cropped_image'],
            'memory_frames':   memory_patches,
            'z_feat_mask':     torch.zeros(Gh*Gw, dtype=torch.long),
            'use_memory_mask': torch.tensor(len(memory_patches)>0, dtype=torch.bool),
        }

        # — 단계 9: plugin 적용 —
        if self.additional_processors:
            for proc in self.additional_processors:
                proc(training_pair, context, data, rng)

        # — 단계 10: 시각화(Optional) —
        if self.visualize:
            from trackit.data.context.worker import get_current_worker_info
            from .visualization import visualize_siam_tracker_training_pair_processor
            op = get_current_worker_info().get_output_path()
            if op:
                visualize_siam_tracker_training_pair_processor(op, training_pair, context, self.image_normalize.__name__)

        return data



class SiamFCCropping:
    def __init__(self, param: SiamFCCroppingParameter):
        self.param = param

    def prepare(self, name: str, rng: np.random.Generator, ctx: dict):
        cp, ok = prepare_siamfc_cropping_with_augmentation(
            ctx[f'{name}_bbox'],
            self.param.area_factor,
            self.param.output_size,
            self.param.scale_jitter_factor,
            self.param.translation_jitter_factor,
            rng,
            self.param.output_min_object_size_in_pixel,
            self.param.output_max_object_size_in_pixel,
            self.param.output_min_object_size_in_ratio,
            self.param.output_max_object_size_in_ratio,
        )
        if ok:
            ctx[f'{name}_cropping_parameter'] = cp
        ctx[f'{name}_cropping_is_success'] = ok
        return ok


    def do(self, name: str, ctx: dict, normalized: bool = True):
        # 1) cp_before 백업
        cp_before = ctx[f'{name}_cropping_parameter']
        img       = ctx[f'{name}_image']

        # 디버그
        # print(f"[DEBUG {name}] cp_before =\n{cp_before}")

        # 2) 실제 크롭 → cp_after: 보정된 파라미터
        cropped, img_mean, cp_after = apply_siamfc_cropping(
            img,
            self.param.output_size,
            cp_before,
            interpolation_mode=self.param.interpolation_mode,
            align_corners=self.param.interpolation_align_corners,
            image_mean=None,
        )
        # print(f"[DEBUG {name}] cp_after  =\n{cp_after}")

        if normalized:
            cropped.div_(255.)

        # 3) cp_after, 이미지 저장
        ctx[f'{name}_cropping_parameter'] = cp_after
        ctx[f'{name}_cropped_image']      = cropped

        # 4) **cp_after** 로 bbox 변환
        if ctx.get(f'{name}_cropping_is_success', True):
            bbox = apply_siamfc_cropping_to_boxes(
                ctx[f'{name}_bbox'],
                cp_after,   # ← 반드시 cp_after
            )
            # print(f"[DEBUG {name}] pre-clip bbox = {bbox}")
            ctx[f'{name}_cropped_bbox'] = bbox


def _decode_with_cache(name: str, frame: SOTFrameInfo, cache: dict, ctx: dict):
    # print(f"[DEBUG raw {name}] object_bbox =", frame.object_bbox)
    if frame.image in cache:
        ctx[f'{name}_image'] = cache[frame.image]
        return
    img = frame.image()
    if isinstance(img, np.ndarray) and img.ndim == 2:
        img = np.stack([img]*3, -1)
    t = torch.from_numpy(img).permute(2,0,1).contiguous().float()
    cache[frame.image] = t
    ctx[f'{name}_image'] = t


def _bbox_clip_to_image_boundary_(bbox: np.ndarray, img: torch.Tensor):
    h, w = img.shape[-2:]
    bbox_clip_to_image_boundary_(bbox, np.array((w, h)))
    assert bbox_is_valid(bbox), f'bbox:\n{bbox}\nimage_size:\n{img.shape}'


class SiamTrackerTrainingPairProcessorBatchCollator:
    def __init__(self, additional_collators=None):
        self.additional_collators = additional_collators

    def __call__(self, batch: Sequence[Mapping], collated: TrainData):
        collated.input.update({
            'z':             collate_element_as_torch_tensor(batch, 'z_cropped_image'),
            'x':             collate_element_as_torch_tensor(batch, 'x_cropped_image'),
            'z_feat_mask':   collate_element_as_torch_tensor(batch, 'z_feat_mask'),
            'memory_frames': [s['memory_frames'] for s in batch],
            'use_memory_mask': collate_element_as_torch_tensor(batch, 'use_memory_mask'),
        })
        dev = collated.input['z'].device
        collated.miscellanies.update({
            'is_positive': collate_element_as_np_array(batch, 'is_positive')
        })
        if self.additional_collators:
            for c in self.additional_collators:
                c(batch, collated)

        if 'num_positive_samples' not in collated.target:
            B = collated.input['z'].shape[0]
            collated.target.update({
                'num_positive_samples':             torch.ones(B, dtype=torch.float, device=dev),
                'boxes':                            collate_element_as_torch_tensor(batch, 'x_cropped_bbox').float(),
                'positive_sample_batch_dim_indices': torch.arange(B, dtype=torch.long, device=dev),
                'positive_sample_map_dim_indices':   torch.zeros(B, dtype=torch.long, device=dev),
            })
        return collated


class SiamTrackerTrainingPairProcessorMainProcessLoggingHook(HostDataPipeline):
    def pre_process(self, input_data: TrainData) -> TrainData:
        ip = input_data.miscellanies.get('is_positive')
        if ip is not None:
            get_current_metric_logger().log({'positive_pair': (ip.sum()/len(ip)).item()})
        return input_data
