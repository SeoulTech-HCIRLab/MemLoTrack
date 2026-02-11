from typing import Dict, Tuple, Callable, Any, Optional, List
import numpy as np
import torch
from dataclasses import dataclass, field

from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping, apply_siamfc_cropping_to_boxes, \
    reverse_siamfc_cropping_params, apply_siamfc_cropping_subpixel, scale_siamfc_cropping_params
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider import CroppingParameterProvider
from ....components.post_process import TrackerOutputPostProcess
from ....components.segmentation import Segmentify_PostProcessor
from ....components.tensor_cache import CacheService, TensorCache

from ... import TrackerEvaluationPipeline
@dataclass
class _LocalContext:
    reset_frame_indices: List[int] = field(default_factory=list)
    siamfc_cropping_params_provider: Optional[CroppingParameterProvider] = None


class OneStreamTracker_Evaluation_MainPipeline(TrackerEvaluationPipeline):
    def __init__(self, device: torch.device,
                 template_image_size: Tuple[int, int],
                 search_region_image_size: Tuple[int, int],  # W, H
                 search_curation_parameter_provider_factory: Callable[[], CroppingParameterProvider],
                 model_output_post_process: TrackerOutputPostProcess,
                 segmentify_post_process: Optional[Segmentify_PostProcessor],
                 interpolation_mode: str, interpolation_align_corners: bool,
                 norm_stats_dataset_name: str, visualization: bool):
        self.template_image_size = template_image_size
        self.search_region_image_size = search_region_image_size

        self.search_image_cropping_params_provider_factory = search_curation_parameter_provider_factory
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

        self.model_output_post_process = model_output_post_process
        self.segmentify_post_process = segmentify_post_process
        self.device = device

        self.image_normalization_transform_ = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.visualization = visualization

    def start(self, max_batch_size: int, global_shared_objects):
        self.max_batch_size = max_batch_size  # 여기서 max_batch_size를 인스턴스 변수로 저장
        max_capacity = max(max_batch_size, len(global_shared_objects.get('task_ids', [])))
        template_shape = (3, self.template_image_size[1], self.template_image_size[0])
        search_region_shape = (3, self.search_region_image_size[1], self.search_region_image_size[0])
        
        self.all_tracking_task_local_contexts: Dict[Any, _LocalContext] = {}
        self.all_tracking_template_cache = CacheService(max_capacity,
                                                        TensorCache(max_capacity, template_shape, self.device))
        self.all_tracking_template_image_mean_cache = CacheService(max_capacity,
                                                                    TensorCache(max_capacity, (3,), self.device))
        global_shared_objects['template_cache'] = self.all_tracking_template_cache
        global_shared_objects['template_image_mean_cache'] = self.all_tracking_template_image_mean_cache
        
        # 디버그: global_shared_objects에 저장된 캐시 확인
        # # print("[OneStreamTracker] Global shared objects keys:", list(global_shared_objects.keys()))
        # # print("[OneStreamTracker] Global template_cache id_list:", self.all_tracking_template_cache.id_list)
        # # print("[OneStreamTracker] Global template_image_mean_cache id_list:", self.all_tracking_template_image_mean_cache.id_list)

        self.cropping_parameter_cache = np.full((max_batch_size, 2, 2), float('nan'), dtype=np.float64)
        self.search_region_cache = torch.full((max_batch_size, *search_region_shape), float('nan'),
                                            dtype=torch.float, device=self.device)
        self.model_output_post_process.start()
        if self.segmentify_post_process is not None:
            self.segmentify_post_process.start(max_batch_size)


    def stop(self, global_shared_objects):
        if self.segmentify_post_process is not None:
            self.segmentify_post_process.stop()
        self.model_output_post_process.stop()
        # 평가가 끝나면 모든 task의 local context를 제거하여 추적 시퀀스가 모두 완료되었음을 보장합니다.
        self.all_tracking_task_local_contexts.clear()
        # 이제 assert는 통과합니다.
        assert len(self.all_tracking_task_local_contexts) == 0, "bug check: some tracking sequences are not finished"
        del self.cropping_parameter_cache
        del self.search_region_cache
        # 캐시들은 global_shared_objects에 보존되어 있으므로, 여기서 삭제하지 않고 유지합니다.
        # del self.all_tracking_template_cache
        # del self.all_tracking_template_image_mean_cache


    def begin(self, context):
        # 평가 데이터에 포함된 task 수
        num_tasks = len(context.input_data.tasks)
        max_capacity = max(self.max_batch_size, num_tasks)
        
        if hasattr(self, 'all_tracking_template_cache') and (self.all_tracking_template_cache is not None):
            current_capacity = len(self.all_tracking_template_cache.id_list)
            if current_capacity < max_capacity:
                # # print("[OneStreamTracker] Expanding template caches from", current_capacity, "to", max_capacity)
                template_shape = (3, self.template_image_size[1], self.template_image_size[0])
                self.all_tracking_template_cache = CacheService(
                    max_capacity, TensorCache(max_capacity, template_shape, self.device)
                )
                self.all_tracking_template_image_mean_cache = CacheService(
                    max_capacity, TensorCache(max_capacity, (3,), self.device)
                )
            # else:
            #     # 용량이 충분하다면 reset() 대신 보존하도록 처리 (Runner의 epoch_end()에서 reset()을 제거하므로 이 부분은 그대로 두면 됩니다)
            #     # print("[OneStreamTracker] Evaluation mode: Preserving existing template caches.")
        else:
            # # print("[OneStreamTracker] Template caches not found, initializing new caches with capacity:", max_capacity)
            template_shape = (3, self.template_image_size[1], self.template_image_size[0])
            self.all_tracking_template_cache = CacheService(
                max_capacity, TensorCache(max_capacity, template_shape, self.device)
            )
            self.all_tracking_template_image_mean_cache = CacheService(
                max_capacity, TensorCache(max_capacity, (3,), self.device)
            )
            
        # # print("[OneStreamTracker] Template cache capacity:", len(self.all_tracking_template_cache.id_list))
        
        for task in context.input_data.tasks:
            if task.task_creation_context is not None:
                self.all_tracking_task_local_contexts[task.id] = _LocalContext()



    def prepare_initialization(self, context, model_input_params):
        # 각 task에 대해 tracker_do_init_context의 입력 데이터를 디버깅 및 캐시에 저장
        for task in context.input_data.tasks:
            if task.tracker_do_init_context is not None:
                init_context = task.tracker_do_init_context
                
                # tracker_do_init_context의 input_data에 포함된 키들을 출력하여 'curated_image'와 'image_mean'이 있는지 확인
                init_keys = list(init_context.input_data.keys())
                # # print(f"[Debug] Task {task.id} tracker_do_init_context input_data keys: {init_keys}")

                # 'curated_image' 체크
                if 'curated_image' not in init_keys:
                    # print(f"[Error] Task {task.id} is missing 'curated_image' in tracker_do_init_context.input_data!")
                    pass
                else:
                    curated = init_context.input_data['curated_image']
                    if hasattr(curated, 'shape'):
                        # print(f"[Debug] Task {task.id} 'curated_image' shape: {curated.shape}")
                        pass
                    else:
                        # print(f"[Debug] Task {task.id} 'curated_image' type: {type(curated)}")
                        pass
                # 'image_mean' 체크
                if 'image_mean' not in init_keys:
                    # print(f"[Error] Task {task.id} is missing 'image_mean' in tracker_do_init_context.input_data!")
                    pass
                else:
                    image_mean = init_context.input_data['image_mean']
                    # print(f"[Debug] Task {task.id} 'image_mean' value: {image_mean}")

                # 캐시에 값을 저장 (중복 호출 제거: 한 번만 호출)
                try:
                    self.all_tracking_template_cache.put(task.id, init_context.input_data['curated_image'])
                    self.all_tracking_template_image_mean_cache.put(task.id, init_context.input_data['image_mean'])
                    # print(f"[Debug] Task {task.id}: Cached 'curated_image' and 'image_mean' successfully.")
                except Exception as e:
                    # print(f"[Error] Task {task.id}: Failed to cache template data: {e}")
                    pass   
                # Cropping parameter provider 초기화
                cropping_params_provider = self.search_image_cropping_params_provider_factory()
                cropping_params_provider.initialize(init_context.gt_bbox)
                task_context = self.all_tracking_task_local_contexts[task.id]
                task_context.siamfc_cropping_params_provider = cropping_params_provider
                task_context.reset_frame_indices.append(init_context.frame_index)
            else:
                # print(f"[Debug] Task {task.id} does not have tracker_do_init_context; skipping initialization for this task.")
                pass 
        # 캐시 상태 출력
        # print("[Debug] all_tracking_template_cache keys:", self.all_tracking_template_cache.id_list)
        # print("[Debug] all_tracking_template_image_mean_cache keys:", self.all_tracking_template_image_mean_cache.id_list)


    def prepare_tracking(self, context, model_input_params: dict):
        num_tracking_sequence = 0
        task_ids = []
        image_size_list = []
        frame_indices = []
        
        # 각 task에 대해 tracking context가 있는 경우만 처리
        for task in context.input_data.tasks:
            if task.tracker_do_tracking_context is not None:
                track_context = task.tracker_do_tracking_context

                # 템플릿 이미지 평균(또는 대체값) 가져오기
                template_image_mean = None
                if task.id in self.all_tracking_template_image_mean_cache.id_list:
                    template_image_mean = self.all_tracking_template_image_mean_cache.get(task.id)
                if template_image_mean is None:
                    # print(f"[OneStreamTracker] Warning: template_image_mean is None for task {task.id}; attempting to use template image from cache.")
                    # 템플릿 이미지 캐시에서 대체 템플릿을 가져옴
                    if task.id in self.all_tracking_template_cache.id_list:
                        template_image_mean = self.all_tracking_template_cache.get(task.id)
                        # print(f"[OneStreamTracker] Info: Using template image from all_tracking_template_cache for task {task.id}.")
                    else:
                        # print(f"[OneStreamTracker] Error: No template image available for task {task.id}; skipping this task.")
                        continue

                # cropping parameters provider 확보
                try:
                    cropping_params_provider = self.all_tracking_task_local_contexts[task.id].siamfc_cropping_params_provider
                except KeyError:
                    # print(f"[OneStreamTracker] Warning: No cropping_params_provider for task {task.id}; skipping this task.")
                    continue

                # 초기 cropping 파라미터 계산
                cropping_params = cropping_params_provider.get(np.array(self.search_region_image_size))
                # 검색 영역 이미지(x)를 float32로 변환
                x = track_context.input_data['image'].to(torch.float32)
                H, W = x.shape[-2:]
                image_size_list.append(np.array((W, H), dtype=np.int32))

                # cropping 파라미터 갱신 및 검색 영역 캐시에 저장
                try:
                    _, _, cropping_params = apply_siamfc_cropping(
                        x,
                        np.array(self.search_region_image_size),
                        cropping_params,
                        self.interpolation_mode,
                        self.interpolation_align_corners,
                        template_image_mean,
                        out_image=self.search_region_cache[num_tracking_sequence, ...]
                    )
                except Exception as e:
                    # print(f"[OneStreamTracker] Error during apply_siamfc_cropping for task {task.id}: {e}; skipping this task.")
                    continue

                # cropping 파라미터 캐시에 저장
                self.cropping_parameter_cache[num_tracking_sequence, ...] = cropping_params
                task_ids.append(task.id)
                frame_indices.append(track_context.frame_index)
                num_tracking_sequence += 1

        # 유효한 task가 하나도 없다면 경고 후 종료
        if num_tracking_sequence == 0:
            # print("[OneStreamTracker] Warning: No valid tracking sequences found in prepare_tracking.")
            return

        # temporary_objects에 관련 정보를 저장
        context.temporary_objects['task_ids'] = task_ids
        context.temporary_objects['x_frame_sizes'] = image_size_list
        context.temporary_objects['x_frame_indices'] = frame_indices
        context.temporary_objects['x_cropping_params'] = self.cropping_parameter_cache[:num_tracking_sequence, ...]

        # 템플릿 이미지 캐시에서 해당 task들의 템플릿 이미지를 가져옴
        z = self.all_tracking_template_cache.get_batch(task_ids)
        if z is None:
            # print("[OneStreamTracker] Error: Failed to retrieve template images from cache.")
            return
        
        # 검색 영역 캐시에서 이미지를 가져와 정규화 (255로 나누기)
        x = self.search_region_cache[:num_tracking_sequence, ...] / 255.
        self.image_normalization_transform_(x)
        
        # 모델 입력으로 사용할 'z'와 'x'가 유효한 텐서인지 확인
        if not isinstance(z, torch.Tensor) or not isinstance(x, torch.Tensor):
            # print("[OneStreamTracker] Error: 'z' or 'x' is not a valid tensor. Aborting prepare_tracking update.")
            return
        
        # model_input_params에 업데이트
        model_input_params.update({'z': z, 'x': x})
        # print("[OneStreamTracker] Info: Updated model_input_params with keys:", list(model_input_params.keys()))

    def on_tracked(self, model_outputs, context):
        # print("[OneStreamTracker Debug] Parent pipeline on_tracked START")
        
        if model_outputs is None:
            # print("[OneStreamTracker Debug]Parent pipeline on_tracked: model_outputs is None")
            return {}  # 빈 dict 반환하여 오류를 회피합니다.
        else:
            # print("[OneStreamTracker Debug]Parent pipeline on_tracked: model_outputs keys before post_process:", list(model_outputs.keys()))
            pass 

        old_dict = dict(model_outputs)

        outputs = self.model_output_post_process(model_outputs)
        # outputs에는 post-process가 만든
        #  {'confidence':..., 'box':..., 'mask':...} 등이 있을 수 있음
        #  x_feat는 보통 여기엔 포함되지 않음
        # print("[OneStreamTracker Debug] parent pipeline post_process done. outputs keys:", list(outputs.keys()))
        
        # print("[OneStreamTracker Debug] MY MERGE CODE: on_tracked *START* ")

        # 3) model_outputs를 비움
        model_outputs.clear()
        for k, v in old_dict.items():
            model_outputs[k] = v
        for k, v in outputs.items():
            model_outputs[k] = v

        # We retrieve these from 'prepare_tracking'
        task_ids = context.temporary_objects.get('task_ids', [])
        x_frame_sizes = context.temporary_objects.get('x_frame_sizes', [])
        x_frame_indices = context.temporary_objects.get('x_frame_indices', [])
        x_cropping_params = context.temporary_objects.get('x_cropping_params', [])


        all_predicted_score = outputs['confidence']         # shape: (num_tracking_sequence,)
        all_predicted_bounding_box = outputs['box']         # shape: (num_tracking_sequence, 4)
        all_predicted_mask = outputs.get('mask', None)      # optional

        # shape checks
        assert all_predicted_score.ndim == 1
        assert all_predicted_bounding_box.ndim == 2
        assert all_predicted_bounding_box.shape[1] == 4
        assert len(task_ids) == len(all_predicted_score) == len(all_predicted_bounding_box)
        if all_predicted_mask is not None:
            assert all_predicted_mask.ndim == 3
            assert all_predicted_mask.shape[0] == len(task_ids)

        # Convert them to CPU for some bounding-box transformations
        all_predicted_score = all_predicted_score.cpu()
        all_predicted_bounding_box = all_predicted_bounding_box.cpu()

        # check for validity
        assert torch.all(torch.isfinite(all_predicted_score))
        assert torch.all(torch.isfinite(all_predicted_bounding_box))

        # convert to numpy for siamfc cropping
        all_predicted_score_np = all_predicted_score.numpy()
        all_predicted_bbox_np = all_predicted_bounding_box.numpy()

        # bounding box: apply reverse siamfc cropping
        all_predicted_bbox_on_full_search_image = apply_siamfc_cropping_to_boxes(
            all_predicted_bbox_np, reverse_siamfc_cropping_params(x_cropping_params))

        # Clip to image boundary
        for pred_bbox, image_size in zip(all_predicted_bbox_on_full_search_image, x_frame_sizes):
            bbox_clip_to_image_boundary_(pred_bbox, image_size)

        # mask or segmentify
        all_predicted_mask_on_full_search_image = None
        if all_predicted_mask is not None:
            # shape => (N, H, W)
            all_predicted_mask_on_full_search_image = []
            for curr_mask, curr_image_size, curr_cropping_parameter in zip(
                    all_predicted_mask, x_frame_sizes, x_cropping_params):
                mask_h, mask_w = curr_mask.shape
                curr_cropping_parameter = scale_siamfc_cropping_params(
                    curr_cropping_parameter,
                    np.array(self.search_region_image_size),
                    np.array((mask_w, mask_h))
                )
                predicted_mask_on_full_search_image = apply_siamfc_cropping_subpixel(
                    curr_mask.to(torch.float32).unsqueeze(0),
                    np.array(curr_image_size),
                    reverse_siamfc_cropping_params(curr_cropping_parameter),
                    self.interpolation_mode, self.interpolation_align_corners
                )
                all_predicted_mask_on_full_search_image.append(
                    predicted_mask_on_full_search_image.squeeze(0).to(torch.bool).cpu().numpy()
                )
        else:
            if self.segmentify_post_process is not None:
                full_search_region_images = []
                for task in context.input_data.tasks:
                    if task.tracker_do_tracking_context is not None:
                        full_search_region_images.append(task.tracker_do_tracking_context.input_data['image'])
                all_predicted_mask_on_full_search_image = (
                    self.segmentify_post_process(
                        full_search_region_images,
                        all_predicted_bbox_on_full_search_image
                    )
                )

        # 5) context.result 에 결과 저장
        for idx, (task_id, image_size, frame_index) in enumerate(zip(task_ids, x_frame_sizes, x_frame_indices)):
            predicted_score = all_predicted_score_np[idx].item()
            predicted_bbox_on_full_search_image = all_predicted_bbox_on_full_search_image[idx]
            mask_on_full_search_image = (all_predicted_mask_on_full_search_image[idx]
                                         if all_predicted_mask_on_full_search_image is not None else None)

            # possibly update cropping param provider
            local_task_context = self.all_tracking_task_local_contexts[task_id]
            local_task_context.siamfc_cropping_params_provider.update(
                predicted_score,
                predicted_bbox_on_full_search_image,
                image_size
            )

            context.result.submit(
                task_id,
                predicted_bbox_on_full_search_image,
                predicted_score,
                mask_on_full_search_image
            )

        # 6) 최종 key ('final_box', 'final_conf') 생성하여 모델 출력에 추가
        final_box_tensor = torch.from_numpy(all_predicted_bbox_on_full_search_image).float().to(self.device)
        final_conf_tensor = torch.from_numpy(all_predicted_score_np).float().to(self.device)


        model_outputs['final_box'] = final_box_tensor
        model_outputs['final_conf'] = final_conf_tensor

        # print("[OneStreamTracker Debug] MY MERGE CODE: on_tracked *END* ====")
        # print("[OneStreamTracker Debug]final model_outputs keys=", list(model_outputs.keys()))
        
        # 4) final assertion
        assert context.result.is_all_submitted()

        return model_outputs