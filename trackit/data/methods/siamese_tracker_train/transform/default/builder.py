from typing import Tuple
import numpy as np
from timm.layers import to_2tuple

from trackit.core.runtime.build_context import BuildContext
from trackit.miscellanies.pretty_format import pretty_format
from .plugin.builder import build_plugins
from .augmentation.builder import build_augmentation_pipeline
from .processor import (
    SiamTrackerTrainingPairProcessor,
    SiamTrackerTrainingPairProcessorBatchCollator,
    SiamTrackerTrainingPairProcessorMainProcessLoggingHook,
    SiamFCCroppingParameter,
)


def _build_siamfc_cropping_parameter(
    siamfc_cropping_config: dict,
    output_size: Tuple[int, int],
    interpolation_mode: str,
    interpolation_align_corners: bool
) -> SiamFCCroppingParameter:
    area_factor = siamfc_cropping_config['area_factor']
    scale_jitter_factor = siamfc_cropping_config.get('scale_jitter', 0.0)
    translation_jitter_factor = siamfc_cropping_config.get('translation_jitter', 0.0)
    output_min_object_size_in_pixel = np.array(
        to_2tuple(siamfc_cropping_config.get('min_object_size', (0.0, 0.0)))
    )
    output_max_object_size_in_pixel = np.array(
        to_2tuple(siamfc_cropping_config.get('max_object_size', (float("inf"), float("inf"))))
    )
    output_min_object_size_in_ratio = siamfc_cropping_config.get('min_object_ratio', 0.0)
    output_max_object_size_in_ratio = siamfc_cropping_config.get('max_object_ratio', 1.0)

    return SiamFCCroppingParameter(
        output_size=np.array(output_size),
        area_factor=area_factor,
        scale_jitter_factor=scale_jitter_factor,
        translation_jitter_factor=translation_jitter_factor,
        output_min_object_size_in_pixel=output_min_object_size_in_pixel,
        output_min_object_size_in_ratio=output_min_object_size_in_ratio,
        output_max_object_size_in_pixel=output_max_object_size_in_pixel,
        output_max_object_size_in_ratio=output_max_object_size_in_ratio,
        interpolation_mode=interpolation_mode,
        interpolation_align_corners=interpolation_align_corners,
    )


def build_siamese_tracker_training_data_processing_components(
    transform_config: dict,
    config: dict,
    build_context: BuildContext
):
    # 플러그인(추가 프로세서, 추가 콜레이터, 추가 호스트 파이프라인)
    additional_processors, additional_data_collators, additional_host_data_pipelines = \
        build_plugins(transform_config, config, build_context)

    common_config = config['common']
    interp_mode = common_config['interpolation_mode']
    interp_align = common_config['interpolation_align_corners']

    # z, x 크롭 파라미터
    template_param = _build_siamfc_cropping_parameter(
        transform_config['SiamFC_cropping']['template'],
        common_config['template_size'],
        interp_mode,
        interp_align,
    )
    search_param = _build_siamfc_cropping_parameter(
        transform_config['SiamFC_cropping']['search_region'],
        common_config['search_region_size'],
        interp_mode,
        interp_align,
    )

    # 모델 인스턴스 생성 및 patch/pos embed 추출
    if 'model_instance' not in build_context.variables:
        build_context.variables['model_instance'] = build_context.model.create(build_context.device)
    model_inst = build_context.variables['model_instance'].model

    # timm 방식으로 list 또는 단일 객체 처리
    # patch_embed나 pos_embed 모듈을 직접 넘기면 안 됩니다.
    # 대신 patch embedding의 patch 크기만 꺼내서 전달합니다.
    raw_patch_embed = model_inst.patch_embed[0] if isinstance(model_inst.patch_embed, list) else model_inst.patch_embed
    patch_size = tuple(raw_patch_embed.patch_size)


    all_augs = transform_config['augmentation']
    # static (DET) keeps joint=True (e.g. DeiT3)
    static_cfgs  = [c for c in all_augs if c.get('joint', True)]
    # dynamic (SOT) only joint=False
    dynamic_cfgs = [c for c in all_augs if not c.get('joint', True)]
    static_pipeline  = build_augmentation_pipeline(static_cfgs)
    dynamic_pipeline = build_augmentation_pipeline(dynamic_cfgs)

    processor = SiamTrackerTrainingPairProcessor(
        template_param,
        search_param,
        dynamic_pipeline,         # SOT: no joint crop
        static_pipeline,          # DET: includes DeiT3 joint crop
        common_config['normalization'],  # norm_stats_dataset_name
        additional_processors,
        transform_config.get('visualize', False),
        patch_size
     )
    # BatchCollator, HostLoggingHook
    batch_collator = SiamTrackerTrainingPairProcessorBatchCollator(additional_data_collators)

    print('transform config:\n', pretty_format(transform_config))
    return (
        processor,
        batch_collator,
        (SiamTrackerTrainingPairProcessorMainProcessLoggingHook(), *additional_host_data_pipelines)
    )
