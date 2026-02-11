from typing import Any, Iterable

import torch
import torch.nn as nn
from trackit.models import ModelManager, ModelImplSuggestions, SampleInputDataGeneratorInterface
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelExecutionPath:
    path: str
    model: nn.Module
    sample_input: Any

    
def iterate_model_execution_paths(model_manager: ModelManager, batch_size: int, device: torch.device,
                                  torch_jit_trace_compatible: bool) -> Iterable[ModelExecutionPath]:
    # ── MemLoTrack BASE 로직 복원: train → eval 두 경로 모두 yield ──
    input_generator = model_manager.sample_input_data_generator

    # 1) Train 모델 (optimize_for_inference=False)
    train_suggestions = ModelImplSuggestions(
        torch_jit_trace_compatible=torch_jit_trace_compatible,
        optimize_for_inference=False
    )
    train_model = model_manager.create(device, train_suggestions)
    train_input = input_generator.get(batch_size, device)
    # memory attention 경로 활성화를 위한 플래그
    if isinstance(train_input, dict):
        train_input["use_memory_mask"] = torch.ones(batch_size, dtype=torch.bool, device=device)
    yield ModelExecutionPath('train', train_model.model.train(), train_input)

    # 2) Eval 모델 (optimize_for_inference=True)
    eval_suggestions = ModelImplSuggestions(
        torch_jit_trace_compatible=torch_jit_trace_compatible,
        optimize_for_inference=True
    )
    eval_model = model_manager.create(device, eval_suggestions)
    eval_input = input_generator.get(batch_size, device)
    if isinstance(eval_input, dict):
        eval_input["use_memory_mask"] = torch.ones(batch_size, dtype=torch.bool, device=device)
    yield ModelExecutionPath('eval', eval_model.model.eval(), eval_input)


# def iterate_model_execution_paths(model_manager: ModelManager, batch_size: int, device: torch.device,
#                                   torch_jit_trace_compatible: bool) -> Iterable[ModelExecutionPath]:
#     input_generator = model_manager.sample_input_data_generator
#     # 평가용 flop analysis에서는 torch_jit_trace_compatible 옵션을 강제 False로 전달합니다.
    
#     train_suggestions = ModelImplSuggestions(torch_jit_trace_compatible=False, optimize_for_inference=False)
#     eval_suggestions = ModelImplSuggestions(torch_jit_trace_compatible=False, optimize_for_inference=True)

#     if isinstance(input_generator, SampleInputDataGeneratorInterface):
#         train_model = model_manager.create(device, train_suggestions)
#         sample_input = input_generator.get(batch_size, device)

#         if train_model.build_string != model_manager.get_build_string(eval_suggestions):
#             yield ModelExecutionPath('train', train_model.model.train(), sample_input)
#             del train_model
#             eval_model = model_manager.create(device, eval_suggestions)
#             yield ModelExecutionPath('eval', eval_model.model.eval(), sample_input)
#         else:
#             yield ModelExecutionPath('', train_model.model.eval(), sample_input)
#     else:
#         train_paths = input_generator.get_path_names(with_train=True, with_eval=False)
#         eval_paths = input_generator.get_path_names(with_train=False, with_eval=True)

#         for train_path in train_paths:
#             train_model = model_manager.create(device, train_suggestions)
#             sample_input = input_generator.get(train_path, batch_size, device)

#             if train_path in eval_paths:
#                 yield ModelExecutionPath(f'{train_path}-train', train_model.model, sample_input)
#                 del train_model
#                 eval_model = model_manager.create(device, eval_suggestions)
#                 yield ModelExecutionPath(f'{train_path}-eval', eval_model.model.eval(), sample_input)
#             else:
#                 yield ModelExecutionPath(train_path, train_model.model, sample_input)

#         for eval_path in eval_paths:
#             if eval_path in train_paths:
#                 continue
#             eval_model = model_manager.create(device, eval_suggestions)
#             sample_input = input_generator.get(eval_path, batch_size, device)
#             yield ModelExecutionPath(eval_path, eval_model.model.eval(), sample_input)
