from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.parallel
from timm.scheduler.scheduler import Scheduler as timmScheduler

from trackit.core.runtime.metric_logger import get_current_local_metric_logger, get_current_metric_logger
from trackit.data import HostDataPipeline
from trackit.core.runtime.context.task import get_current_task_context
from trackit.runner import Runner
from trackit.models import ModelInstance
from trackit.data.protocol.train_input import TrainData
from trackit.criteria import CriterionOutput
from trackit.miscellanies.torch.distributed.reduce_dict import reduce_dict_async

from .model_wrapper import build_model_wrapper, build_model_wrapper_eval
from ..common.nan_dump import do_loss_nan_fault_dump
from ..common.distributed_data_parallel import DistributedDataParallelOption
from ..common.optimization import OptimizationModulesAndOptions
from .utils import criterion_has_parameters


class DefaultTrainer(Runner):
    def __init__(self, model: ModelInstance, criterion: nn.Module,
                 optimization_modules: OptimizationModulesAndOptions,
                 distributed_data_parallel_option: Optional[DistributedDataParallelOption],
                 torch_compile_options: Optional[dict],
                 detect_unused_parameters: bool = True):
        
        self._model_instance = model
        self._raw_model = model.model
        self._criterion = criterion
        self._ema = optimization_modules.ema

        self._model: Optional[nn.Module] = None
        self._wrapped_model_train: Optional[nn.Module] = None
        self._has_in_computational_graph_criterion = criterion_has_parameters(criterion)

        self._init = False
        self._distributed_data_parallel_option = distributed_data_parallel_option
        self._torch_compile_options = torch_compile_options



        self._optimizer = optimization_modules.optimizer



        self._is_apex_optimizer = optimization_modules.is_apex_optimizer
        self._lr_scheduler_per_iteration = optimization_modules.lr_scheduler_per_iteration
        self._lr_scheduler_per_epoch = optimization_modules.lr_scheduler_per_epoch
        self._wd_scheduler_per_iteration = optimization_modules.weight_decay_scheduler_per_iteration
        self._wd_scheduler_per_epoch = optimization_modules.weight_decay_scheduler_per_epoch

        self._parameter_updater = optimization_modules.parameter_updater
        self._amp_auto_cast_fn = optimization_modules.amp_auto_cast_fn
        self._autograd_detect_anomaly_fn = optimization_modules.autograd_detect_anomaly_fn

        self._grad_accumulation_steps = optimization_modules.grad_accumulation_steps
        self._zero_grad_set_to_none = optimization_modules.zero_grad_set_to_none

        self._detect_unused_parameters = detect_unused_parameters

        self.data_pipeline_on_host = {}
        self.task_name = None
        self.is_train = True
        self._iteration = 0


        self.memory_banks = {}          # 각 task 별 memory bank (훈련 시 video 데이터용)
        self.search_region_image_size = None  # 나중에 epoch_begin에서 설정
        self.patch_size = None                 # 나중에 epoch_begin에서 설정

    def _deferred_init(self):
        if self._init:
            return
        model = self._raw_model
        criterion = self._criterion
        distributed_data_parallel_option = self._distributed_data_parallel_option
        torch_compile_options = self._torch_compile_options
        self._wrapped_model_train = build_model_wrapper(model, criterion, distributed_data_parallel_option, torch_compile_options)
        self._init = True

    def register_data_pipeline(self, task_name: str, data_pipeline: HostDataPipeline) -> None:
        if task_name not in self.data_pipeline_on_host:
            self.data_pipeline_on_host[task_name] = []
        self.data_pipeline_on_host[task_name].append(data_pipeline)

    def switch_task(self, task_name, is_train):
        self.task_name = task_name
        self.is_train = is_train

    def epoch_begin(self, epoch: int, _):
        self._deferred_init()
        local_logger = get_current_local_metric_logger()

        self.memory_banks = {}  
        # 모델 인스턴스 또는 config에서 검색 영역 이미지 크기와 patch_size를 가져옵니다.
        # (여기서는 모델 인스턴스에 해당 속성이 있으면 사용, 없으면 기본값 (224,224)와 14로 설정)
        self.search_region_image_size = getattr(self._model_instance, 'search_region_image_size', (224, 224))
        self.patch_size = getattr(self._model_instance, 'patch_size', 14)
        self._epoch = epoch
        
        if local_logger is not None:
            if self.is_train:
                local_logger.set_metric_format('lr', window_size=1, format='{value:.6f}')
                local_logger.set_metric_format('weight_decay', window_size=1, format='{value:.6f}')
                if self._parameter_updater.is_grad_scaler_enabled():
                    local_logger.set_metric_format('loss_scale', window_size=1, format='{value:.2f}')
                if self._parameter_updater.has_grad_norm():
                    local_logger.set_metric_format('grad_norm', window_size=1, format='{value:.4f}')
            local_logger.set_metric_format('loss')

        self._raw_model.train(self.is_train)
        self._criterion.train(self.is_train)
        if self.is_train:
            self._model = self._wrapped_model_train
        else:
            self._model = build_model_wrapper_eval(self._raw_model, self._criterion)
        if self._ema is not None:
            self._ema.on_epoch_begin()
        self._epoch = epoch


    def epoch_end(self, epoch: int, model_manager):
        assert epoch == self._epoch
        if self.is_train:
            if self._lr_scheduler_per_epoch is not None:
                if isinstance(self._lr_scheduler_per_epoch, timmScheduler):
                    self._lr_scheduler_per_epoch.step(epoch)
                else:
                    self._lr_scheduler_per_epoch.step()
            if self._wd_scheduler_per_epoch is not None:
                self._wd_scheduler_per_epoch.step(epoch)
            if self._ema is None:
                self._model_instance.notify_update()  # 최신 state_dict를 ModelManager에 반영
            else:
                model_manager.load_state_dict(self._ema.get_model().state_dict(), print_missing=False)
         
        if hasattr(self._model_instance.model, 'memory_attention'):
            memory_attn = self._model_instance.model.memory_attention
            # logs 폴더가 없으면 생성 (os 모듈 필요)
            import os
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            for i, layer in enumerate(memory_attn.layers):
                file_path = os.path.join(log_dir, f"memory_attention_gradients_epoch{epoch}_layer{i}.csv")
                layer.save_gradients(file_path)
            
        del self._epoch
        self._model = None


        
    def run(self, data: TrainData):
        with self._autograd_detect_anomaly_fn():
            metrics = {}
            data_pipeline_on_host = self.data_pipeline_on_host.get(self.task_name, None)
            if data_pipeline_on_host is not None:
                for data_pipeline in data_pipeline_on_host:
                    data: TrainData = data_pipeline.pre_process(data)

            model_output = None

            criterion_output: Optional[dict] = None  # 모델은 dict를 반환한다고 가정
            
            
            if data.input is not None:
                with torch.set_grad_enabled(self.is_train), self._amp_auto_cast_fn():
                    # 모델 forward 호출
                    criterion_output = self._model(data.input, data.target)
                
                # print(f"[DefaultTrainer] model_output type: {type(criterion_output)}")
                loss = criterion_output['loss']
                # 1) score_map
                if 'score_map' in criterion_output:
                    sm = criterion_output['score_map']
                    # print(f"[LOSS Debug] score_map NaN? {sm.isnan().any().item()}, "
                    #     f"min={sm.min().item():.4f}, max={sm.max().item():.4f}")
                # 2) boxes
                if 'boxes' in criterion_output:
                    bx = criterion_output['boxes']
                    # print(f"[LOSS Debug] boxes NaN? {bx.isnan().any().item()}, "
                    #     f"min={bx.min().item():.4f}, max={bx.max().item():.4f}")
                # 3) target boxes
                tb = data.target.get('boxes', None)
                if tb is not None:
                    # print(f"[LOSS Debug] target boxes NaN? {tb.isnan().any().item()}, "
                    #     f"min={tb.min().item():.4f}, max={tb.max().item():.4f}")
                    pass
            
            
            # ── 디버그: memory hook 호출 조건 체크 ───────────────────────
            # print(f"[MEMHOOK_DBG] is_train={self.is_train}, use_memory={data.input.get('use_memory')}")
            # print(f"[MEMHOOK_DBG] criterion_output keys = {list(criterion_output.keys())}")
            
            
            # 여기서 update_memory_bank_training을 호출하기 전에 확인
            # if self.is_train and data.input.get('use_memory', False):
            #     # print("[MEMHOOK_DBG] Calling update_memory_bank_training")
            #     from trackit.runner.training.default.memory_update_hook import update_memory_bank_training
            #     update_memory_bank_training(criterion_output, data.input, self.memory_banks,
            #                                 self.search_region_image_size, self.patch_size,
            #                                 self._model_instance.device)
            # else:
            #     print("[MEMHOOK_DBG] Skipping memory_update_hook")
                

            # 이하 기존 run() 로직 계속...
            if criterion_output.get('metrics', None) is None:
                metrics['loss'] = criterion_output['loss'].item()
            else:
                metrics['loss'] = sum(criterion_output['metrics'].values())
                metrics.update(criterion_output['metrics'])
                if criterion_output.get('extra_metrics', None) is not None:
                    metrics.update(criterion_output['extra_metrics'])
                
                if not torch.isfinite(criterion_output['loss']):
                    output_path = get_current_task_context().get_output_path()
                    if output_path is not None:
                        do_loss_nan_fault_dump(self._model, self._optimizer,
                                            self._lr_scheduler_per_iteration, self._lr_scheduler_per_epoch,
                                            self._parameter_updater,
                                            data, model_output, metrics,
                                            output_path,
                                            self.task_name, self._epoch, self._iteration)
                    raise RuntimeError(f"Loss is {criterion_output['loss'].item()}, stopping training\n{metrics}")
            
                
            if criterion_output is not None and self.is_train:
                is_second_order = hasattr(self._optimizer, 'is_second_order') and self._optimizer.is_second_order
                update_grad = ((self._iteration + 1) % self._grad_accumulation_steps) == 0
                loss_scale, grad_norm = self._parameter_updater.backward_and_unscale(
                    criterion_output['loss'],
                    self._optimizer,
                    create_graph=is_second_order,
                    update_grad=update_grad)

                # ── MEMDBG: MemoryAttention 파라미터 grad norm 찍기 ────────────────
                from trackit.core.memory.memory_attention import MemoryAttention
                mem_attn = getattr(self._model, 'memory_attention', None)
                if isinstance(mem_attn, MemoryAttention):
                    for layer_i, layer in enumerate(mem_attn.layers):
                        for name, param in layer.cross_attn.named_parameters():
                            if param.grad is not None:
                                print(f"[MEM_GRAD] layer={layer_i} param={name} grad_norm={param.grad.norm().item():.6f}")
                            else:
                                print(f"[MEM_GRAD] layer={layer_i} param={name} grad=None")
                # ─────────────────────────────────────────────────────────────

                stat = {'lr': self._optimizer.param_groups[0]["lr"],
                        'weight_decay': self._optimizer.param_groups[0]["weight_decay"]}
                if loss_scale is not None:
                    stat['loss_scale'] = loss_scale
                if grad_norm is not None:
                    stat['grad_norm'] = grad_norm
                metrics.update(stat)
                metrics = reduce_dict_async(metrics)
                self._parameter_updater.step(self._optimizer, update_grad=update_grad)

                # # ------- 디버깅 용: 파라미터 이름과 grad norm 출력 -------
                # for name, param in self._model.named_parameters():
                #    if param.grad is not None:
                #         print(f"[GRADDBG] {name} grad norm = {param.grad.norm().item():.6f}")                    

                # # 3) parameter update
                # metrics = reduce_dict_async(metrics)
                # self._parameter_updater.step(self._optimizer, update_grad=update_grad)
                # # -------------


                if update_grad:
                    if self._detect_unused_parameters:
                        for name, param in self._model.named_parameters():
                            if param.grad is None and param.requires_grad:
                                print(f'unused parameter detected: {name}')
                    if self._is_apex_optimizer:
                        self._optimizer.zero_grad()
                    else:
                        self._optimizer.zero_grad(self._zero_grad_set_to_none)
                    if self._lr_scheduler_per_iteration is not None:
                        if isinstance(self._lr_scheduler_per_iteration, timmScheduler):
                            self._lr_scheduler_per_iteration.step_update(self._iteration)
                        else:
                            self._lr_scheduler_per_iteration.step()
                    if self._wd_scheduler_per_iteration is not None:
                        self._wd_scheduler_per_iteration.step_update(self._iteration)
                    if self._ema is not None:
                        self._ema.update_parameters(self._raw_model)
            else:
                metrics = reduce_dict_async(metrics)
            
            if data_pipeline_on_host is not None:
                for data_pipeline in reversed(data_pipeline_on_host):
                    model_output = data_pipeline.post_process(model_output)
            
            if self.is_train:
                self._iteration += 1
            metrics = metrics.get()
            if len(metrics) > 0:
                get_current_metric_logger().log(metrics)

    def get_state(self) -> Any:
        state_dict = {'optimizer': self._optimizer.state_dict(), 'iteration': self._iteration}

        if self._has_in_computational_graph_criterion:
            state_dict['criterion'] = self._criterion.state_dict()

        if self._lr_scheduler_per_iteration is not None:
            state_dict['lr_scheduler_per_iteration'] = self._lr_scheduler_per_iteration.state_dict()
        if self._lr_scheduler_per_epoch is not None:
            state_dict['lr_scheduler'] = self._lr_scheduler_per_epoch.state_dict()
        state_dict['amp_param_updater'] = self._parameter_updater.state_dict()
        if self._ema is not None:
            state_dict['ema'] = self._ema.state_dict()

        return state_dict

    def set_state(self, state_dict: Any) -> None:
        if self._distributed_data_parallel_option is not None:
            assert not self._init, "limitation: state cannot be updated when torch.nn.parallel.DistributedDataParallel is enabled"
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self._iteration = state_dict['iteration']
        if self._has_in_computational_graph_criterion:
            self._criterion.load_state_dict(state_dict['criterion'])
        if self._lr_scheduler_per_iteration is not None:
            self._lr_scheduler_per_iteration.load_state_dict(state_dict['lr_scheduler_per_iteration'])
        if self._lr_scheduler_per_epoch is not None:
            self._lr_scheduler_per_epoch.load_state_dict(state_dict['lr_scheduler'])
        if self._ema is not None:
            self._ema.load_state_dict(state_dict['ema'])
        self._parameter_updater.load_state_dict(state_dict['amp_param_updater'])
