import torch
from typing import Sequence, Dict

from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.builder import (
    build_siamfc_search_region_cropping_parameter_provider_factory
)

# Post-process & segmentation builders
from ....components.post_process.builder import build_post_process
from ....components.segmentation.builder import build_segmentify_post_processor

# Existing main pipeline
from . import OneStreamTracker_Evaluation_MainPipeline

# The *new* memory pipeline
from .with_memory_pipeline import OneStreamTracker_Evaluation_MainPipeline_WithMemory


def build_one_stream_tracker_pipeline(pipeline_config: dict, config: dict, device: torch.device) -> Sequence[OneStreamTracker_Evaluation_MainPipeline]:
    """
    Original function used when pipeline_config['type'] == 'one_stream_tracker'.
    """
    common_config = config['common']

    visualization = pipeline_config.get('visualization', False)
    print('pipeline: one stream tracker')
    print('visualization:', visualization)

    # Build post-process
    post_process_cfg = pipeline_config.get('post_process', None)
    post_process = None
    if post_process_cfg is not None:
        post_process = build_post_process(post_process_cfg, common_config, device)

    # Build segmentation post-process if defined
    segmentify_cfg = pipeline_config.get('segmentify', None)
    segmentify = None
    if segmentify_cfg is not None:
        segmentify = build_segmentify_post_processor(segmentify_cfg, common_config, device)

    # Build the main pipeline
    main_pipeline = OneStreamTracker_Evaluation_MainPipeline(
        device=device,
        template_image_size=common_config['template_size'],
        search_region_image_size=common_config['search_region_size'],
        search_curation_parameter_provider_factory=build_siamfc_search_region_cropping_parameter_provider_factory(
            pipeline_config['search_region_cropping']
        ),
        model_output_post_process=post_process,
        segmentify_post_process=segmentify,
        interpolation_mode=common_config['interpolation_mode'],
        interpolation_align_corners=common_config['interpolation_align_corners'],
        norm_stats_dataset_name=common_config['normalization'],
        visualization=visualization
    )

    pipelines = [main_pipeline]

    # Possibly add plugin pipelines
    if 'plugin' in pipeline_config:
        pipelines.extend(_build_plugins(pipeline_config['plugin'], config, device))

    return pipelines


def build_one_stream_tracker_with_memory_pipeline(pipeline_config: dict, config: dict, device: torch.device) -> Sequence[OneStreamTracker_Evaluation_MainPipeline_WithMemory]:
    """
    NEW function for pipeline_config['type'] == 'one_stream_tracker_with_memory'.
    Uses OneStreamTracker_Evaluation_MainPipeline_WithMemory from with_memory_pipeline.py
    """
    common_config = config['common']

    visualization = pipeline_config.get('visualization', False)
    print('pipeline: one_stream_tracker_with_memory')
    print('visualization:', visualization)

    # Build post-process
    post_process_cfg = pipeline_config.get('post_process', None)
    post_process = None
    if post_process_cfg is not None:
        post_process = build_post_process(post_process_cfg, common_config, device)

    # Build segmentation if any
    segmentify_cfg = pipeline_config.get('segmentify', None)
    segmentify = None
    if segmentify_cfg is not None:
        segmentify = build_segmentify_post_processor(segmentify_cfg, common_config, device)

    # memory/cross-attn hyperparams from pipeline config
    memory_max_size = pipeline_config.get('memory_max_size', 7)
    memory_conf_thresh = pipeline_config.get('memory_conf_thresh', 0.8)
    embed_dim = pipeline_config.get('embed_dim', 768)
    use_memory_attention = pipeline_config.get('use_memory_attention', True)

    main_pipeline = OneStreamTracker_Evaluation_MainPipeline_WithMemory(
        device=device,
        template_image_size=common_config['template_size'],
        search_region_image_size=common_config['search_region_size'],
        search_curation_parameter_provider_factory=build_siamfc_search_region_cropping_parameter_provider_factory(
            pipeline_config['search_region_cropping']
        ),
        model_output_post_process=post_process,
        segmentify_post_process=segmentify,
        interpolation_mode=common_config['interpolation_mode'],
        interpolation_align_corners=common_config['interpolation_align_corners'],
        norm_stats_dataset_name=common_config['normalization'],
        visualization=visualization,
        memory_max_size=memory_max_size,
        memory_conf_thresh=memory_conf_thresh,
        embed_dim=embed_dim,
        use_memory_attention=use_memory_attention
    )

    pipelines = [main_pipeline]

    if 'plugin' in pipeline_config:
        pipelines.extend(_build_plugins(pipeline_config['plugin'], config, device))

    return pipelines


def _build_plugins(plugins_config, config, device):
    pipelines = []
    for plugin_config in plugins_config:
        if plugin_config['type'] == 'template_foreground_indicating_mask_generation':
            from .._common.template_foreground_indicating_mask_generation import TemplateFeatForegroundMaskGeneration
            pipelines.append(TemplateFeatForegroundMaskGeneration(
                config['common']['template_size'],
                config['common']['template_feat_size'],
                device
            ))
        else:
            raise ValueError('Unknown plugin type: {}'.format(plugin_config['type']))
    return pipelines
