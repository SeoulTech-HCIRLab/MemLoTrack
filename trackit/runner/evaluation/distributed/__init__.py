from typing import Optional, Any

import torch
from torch import nn
import gc

from trackit.runner import Runner
from trackit.models import ModelManager
from trackit.models.compiling import InferenceEngine
from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.data.context import get_current_data_context
from trackit.data import HostDataPipeline
from .tracker_evaluator import TrackerEvaluator, run_tracker_evaluator


class DefaultTrackerEvaluationRunner(Runner):
    def __init__(self, tracker_evaluator: TrackerEvaluator,
                 inference_engine: InferenceEngine, device: torch.device):
        self.tracker_evaluator = tracker_evaluator
        self.inference_engine = inference_engine
        self._device = device

        self.data_pipeline_on_host = {}
        self.branch_name = None

        self.raw_model: Optional[nn.Module] = None
        self.optimized_model: Any = None

    def switch_task(self, task_name, is_train):
        self.branch_name = task_name
        assert not is_train, "Evaluator can only be run in evaluation mode"

    def epoch_begin(self, epoch: int, model_manager: ModelManager):
        max_batch_size = get_current_data_context().variables['batch_size']

        self.optimized_model, self.raw_model = self.inference_engine(model_manager, self._device, max_batch_size)
        self.tracker_evaluator.on_epoch_begin()
        # ── 여기에 raw_model을 evaluator의 global_shared_objects에 넣어줍니다.
        if hasattr(self.tracker_evaluator, 'global_shared_objects'):
            self.tracker_evaluator.global_shared_objects['model'] = self.raw_model

    def epoch_end(self, epoch: int, _):
        self.tracker_evaluator.on_epoch_end()
        # 평가 배치가 끝나면, MemoryBank만 초기화합니다.
        for pipeline in self.tracker_evaluator.pipelines:
            if hasattr(pipeline, 'memory_banks'):
                pipeline.memory_banks.clear()
            # 템플릿 캐시와 이미지 평균 캐시는 유지 (reset 호출 X)
        self.optimized_model = self.raw_model = None
        gc.collect()

    def run(self, data: TrackerEvalData):

        """
        DefaultTrackerEvaluationRunner의 run 메서드를 수정.
        post_process를 호출할 때마다, 'x_feat'등 기존 키를 유지/병합하도록 변경.
        또한 debug print를 넣어 어디서 x_feat가 날아가는지 추적한다.
        """
        # -------------------------
        # (1) pre_process
        # -------------------------
        if self.data_pipeline_on_host.get(self.branch_name) is not None:
            for i, data_pipeline in enumerate(self.data_pipeline_on_host[self.branch_name]):
                if hasattr(data_pipeline, 'pre_process'):
                    data = data_pipeline.pre_process(data)

        # -------------------------
        # (2) run_tracker_evaluator => pipeline.on_tracked
        # -------------------------
        outputs = run_tracker_evaluator(self.tracker_evaluator, 
                                        data, self.optimized_model, 
                                        self.raw_model
                                        )

        # -------------------------
        # (3) post_process: 여기서 병합 로직 + debug
        # -------------------------
        if self.data_pipeline_on_host.get(self.branch_name) is not None:
            # 거꾸로 iterate
            reversed_pipelines = reversed(self.data_pipeline_on_host[self.branch_name])
            for j, data_pipeline in enumerate(reversed_pipelines):
                if hasattr(data_pipeline, 'post_process'):
                    # print(f"[Runner Debug] BEFORE pipeline {j} post_process => outputs keys="
                        #  f"{list(outputs.keys())}")
                    
                    old_dict = dict(outputs) if hasattr(outputs, 'keys') else {}
                    processed = data_pipeline.post_process(outputs)

                    # 병합
                    if hasattr(outputs, 'clear'):
                        outputs.clear()
                    for k, v in old_dict.items():
                        outputs[k] = v
                    if processed is not None and hasattr(processed, 'items'):
                        for k, v in processed.items():
                            outputs[k] = v

                    # print(f"[Runner Debug] AFTER pipeline {j} post_process => outputs keys="
                    #     f"{list(outputs.keys())}")

        if outputs is None:
            # print("[Runner Debug] Warning: pipeline returned None, setting outputs to empty dict.")
            outputs = {}
        # print(f"[Runner Debug] final outputs keys={list(outputs.keys())}")
        return outputs
    

    def register_data_pipeline(self, task_name: str, data_pipeline: HostDataPipeline) -> None:
        if task_name not in self.data_pipeline_on_host:
            self.data_pipeline_on_host[task_name] = []
        self.data_pipeline_on_host[task_name].append(data_pipeline)
