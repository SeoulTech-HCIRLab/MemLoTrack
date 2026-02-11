import torch
import torch.nn as nn
from typing import Optional, Any
from torch.nn.parallel import DistributedDataParallel
from trackit.models.schema.input.auto_unpack import auto_unpack_and_call
from ..common.distributed_data_parallel import DistributedDataParallelOption
from .utils import criterion_has_parameters


class ModelWithCriterion(nn.Module):
    def __init__(self, model, criterion):
        super(ModelWithCriterion, self).__init__()
        self.model = model
        self.criterion = criterion #loss 


    def forward(self, samples, targets) -> dict:
        # raw_output: 모델의 원시 출력 (예: dict 형태로 x_feat, final_box, final_conf 포함)
        raw_output = auto_unpack_and_call(samples, self.model)
        
        # CriterionOutput을 생성 (loss 및 metrics 계산)
        criterion_output = self.criterion(raw_output, targets)
    

        # ── 디버그: NaN/Inf 손실 감지 ───────────────────────────────
        loss = criterion_output.loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("[ANOMALY] Detected NaN/Inf in loss!")
            print("  cls_loss:", criterion_output.metrics.get(f"Loss/{criterion_output.cls_loss_display_name}", None),
                  " reg_loss:", criterion_output.metrics.get(f"Loss/{criterion_output.bbox_reg_loss_display_name}", None))
            # raw score map 확인 (metrics에서 꺼낼 수 없다면 모델 출력을 직접 찍어야 합니다)
        # ────────────────────────────────────────────────────────────

        out_dict = {}
        if isinstance(raw_output, dict):
            # x_feat, final_box, final_conf가 있다면 복사합니다.
            for key in ['x_feat', 'final_box', 'final_conf']:
                out_dict[key] = raw_output.get(key, None)
            # ── 메모리 프레임 출력 추가 ─────────────────────────────
            # (모델이 생산하는 memory_frames 텐서를 next-memory-update-hook에 넘깁니다)
            out_dict['memory_frames'] = raw_output.get('memory_frames', None)
            # ─────────────────────────────────────────────────────────
    
        else:
            # raw_output이 dict가 아니라면 에러를 발생시켜야 합니다.
            raise ValueError("Model output is not a dictionary. Cannot proceed with model_wrapper.")


        # 만약 final_box가 None이면, score_map과 boxes를 이용하여 계산합니다.
        if out_dict.get('final_box', None) is None:
            if 'score_map' in raw_output and 'boxes' in raw_output:
                B = raw_output['score_map'].shape[0]
                # Flatten score_map to (B, N) where N = H*W
                score_map_flat = raw_output['score_map'].view(B, -1)

                
                final_conf, indices = score_map_flat.max(dim=1)
                # Reshape boxes: (B, H, W, 4) -> flatten to (B, N, 4)
                boxes_flat = raw_output['boxes'].view(B, -1, 4)
                try:
                    final_box_list = [boxes_flat[i, indices[i], :] for i in range(B)]
                    final_box = torch.stack(final_box_list, dim=0)
                except Exception as e:
                    raise ValueError(f"Error computing final_box from score_map and boxes: {e}")
                out_dict['final_box'] = final_box
                out_dict['final_conf'] = final_conf.unsqueeze(1)
            else:
                raise ValueError("final_box is missing and raw_output does not contain 'score_map' and 'boxes'.")
        
        # 만약 어떤 키가 여전히 None이면 에러를 발생시킵니다.
        if out_dict.get('x_feat', None) is None:
            raise ValueError("x_feat is missing in model output.")
        if out_dict.get('final_box', None) is None:
            raise ValueError("final_box is missing in model output after processing.")
        if out_dict.get('final_conf', None) is None:
            raise ValueError("final_conf is missing in model output after processing.")
        
        # criterion_output의 loss 및 metrics도 저장 (원하는 경우, 따로 보관할 수 있음)
        out_dict['loss'] = criterion_output.loss
        out_dict['metrics'] = criterion_output.metrics
        out_dict['extra_metrics'] = criterion_output.extra_metrics

        # ── 디버그: memory_frames 포함 여부 확인 ───────────────────
        # print(f"[MODELWRAP_DBG] memory_frames in out_dict = {out_dict['memory_frames'] is not None}")
        # ─────────────────────────────────────────────────────────────
 
        return out_dict

def build_model_wrapper(model: nn.Module, criterion: nn.Module,
                        distributed_data_parallel_option: Optional[DistributedDataParallelOption],
                        torch_compile_options: Optional[dict]):
    in_computational_graph_criterion = criterion_has_parameters(criterion)
    wrapped_model = ModelWithCriterion(model, criterion)

    if distributed_data_parallel_option is not None:
        if distributed_data_parallel_option.convert_sync_batchnorm:
            if in_computational_graph_criterion:
                wrapped_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(wrapped_model)
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                wrapped_model.model = model
        if distributed_data_parallel_option.model_params_and_buffers_to_ignore is not None or distributed_data_parallel_option.criterion_params_and_buffers_to_ignore is not None:
            params_and_buffer_to_ignore = []
            if distributed_data_parallel_option.model_params_and_buffers_to_ignore is not None:
                if in_computational_graph_criterion:
                    params_and_buffer_to_ignore.extend('model.' + name for name in
                                                       distributed_data_parallel_option.model_params_and_buffers_to_ignore)
                else:
                    params_and_buffer_to_ignore.extend(name for name in
                                                       distributed_data_parallel_option.model_params_and_buffers_to_ignore)
            if distributed_data_parallel_option.criterion_params_and_buffers_to_ignore is not None:
                params_and_buffer_to_ignore.extend('criterion.' + name for name in
                                                   distributed_data_parallel_option.criterion_params_and_buffers_to_ignore)
            if in_computational_graph_criterion:
                DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(wrapped_model,
                                                                                    params_and_buffer_to_ignore)
            else:
                DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model,
                                                                                    params_and_buffer_to_ignore)
        if in_computational_graph_criterion:
            wrapped_model = DistributedDataParallel(wrapped_model,
                                                    find_unused_parameters=distributed_data_parallel_option.find_unused_parameters,
                                                    gradient_as_bucket_view=distributed_data_parallel_option.gradient_as_bucket_view,
                                                    static_graph=distributed_data_parallel_option.static_graph)
        else:
            model = DistributedDataParallel(model,
                                            find_unused_parameters=distributed_data_parallel_option.find_unused_parameters,
                                            gradient_as_bucket_view=distributed_data_parallel_option.gradient_as_bucket_view,
                                            static_graph=distributed_data_parallel_option.static_graph)
            wrapped_model.model = model
        print('DistributedDataParallel is enabled.')

    if torch_compile_options is not None:
        if in_computational_graph_criterion:
            wrapped_model = torch.compile(wrapped_model, **torch_compile_options)
        else:
            model = torch.compile(model, **torch_compile_options)
            wrapped_model.model = model
        message = 'torch.compile is enabled'
        if in_computational_graph_criterion:
            message += ' with criterion'
        message += '.'
        if len(torch_compile_options) > 0:
            message += f' parameters: {torch_compile_options}.'
        print(message)
    return wrapped_model


def build_model_wrapper_eval(model: nn.Module, criterion: nn.Module):
    return ModelWithCriterion(model, criterion)
