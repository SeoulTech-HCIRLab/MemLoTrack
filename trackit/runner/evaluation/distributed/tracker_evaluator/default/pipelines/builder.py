import torch
from typing import Sequence

from . import TrackerEvaluationPipeline


from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines import TrackerEvaluationPipeline
from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines.one_stream.builder import build_one_stream_tracker_pipeline
from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines.one_stream.builder import build_one_stream_tracker_with_memory_pipeline


def build_tracker_evaluator_data_pipeline(evaluator_config, config, device):
    pipeline_config = evaluator_config['pipeline']
    t = pipeline_config['type']
    if t == 'one_stream_tracker':
        from .one_stream.builder import build_one_stream_tracker_pipeline
        return build_one_stream_tracker_pipeline(pipeline_config, config, device)
    elif t == 'one_stream_tracker_with_memory':
        from .one_stream.builder import build_one_stream_tracker_with_memory_pipeline
        return build_one_stream_tracker_with_memory_pipeline(pipeline_config, config, device)
    else:
        raise NotImplementedError(f"Unknown pipeline type: {t}")