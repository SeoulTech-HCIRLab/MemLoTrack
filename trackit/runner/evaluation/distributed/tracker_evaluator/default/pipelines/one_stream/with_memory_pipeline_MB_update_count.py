
# trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines/one_stream/with_memory_pipeline.py
"""
With-memory pipeline
- MB 저장 소스: **포스트-패치임베딩(post-embed) 토큰** (encoder 출력 전)
- Motion-aware gating (Kalman): SAMURAI식 동작 일관성 기반 게이팅 추가

추가 기능 (통계):
- τ ∈ {0.1..0.9} 에 대해, accept_motion 이면서 conf ≥ τ 인 프레임 수("would-write") 히스토그램 집계
- 실제 저장(append/replace) 횟수는 MemoryBank.write_count 를 end() 시점에 수집
- stop() 시 rank별 JSON 파일로 기록 (환경변수 MB_WRITE_LOG_DIR 사용 가능)
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

from . import OneStreamTracker_Evaluation_MainPipeline
from trackit.core.memory.memory_bank import MemoryBank
from trackit.core.memory.memory_attention import MemoryAttention
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.utils.siamfc_cropping import (
    apply_siamfc_cropping_to_boxes,
    reverse_siamfc_cropping_params
)
from trackit.core.memory.motion_gate import MotionGateRegistry


class OneStreamTracker_Evaluation_MainPipeline_WithMemory(OneStreamTracker_Evaluation_MainPipeline):
    """
    Extends 'OneStreamTracker_Evaluation_MainPipeline' to store full-frame post-embedding tokens
    (search crop 전체; patch embedding 출력 == pre-encoder tokens) in the MemoryBank,
    and use them as K,V for MemoryAttention.
    + A방법론: K,V를 현재 Q 격자에 정렬해서 Cross-Attn에 투입.
    """
    debug_call_count = 0

    def __init__(
        self,
        device: torch.device,
        template_image_size: tuple,
        search_region_image_size: tuple,
        search_curation_parameter_provider_factory,
        model_output_post_process,
        segmentify_post_process,
        interpolation_mode: str,
        interpolation_align_corners: bool,
        norm_stats_dataset_name: str,
        visualization: bool,
        memory_max_size: int = 7,
        memory_conf_thresh: float = 0.1,
        embed_dim: int = 768,
        use_memory_attention: bool = True,
        # Motion-aware gate 하이퍼파라미터 (필요 시 조정)
        gate_process_var_pos: float = 50.0,
        gate_process_var_vel: float = 10.0,
        gate_process_var_size: float = 0.5,
        gate_meas_var_pos: float = 30.0,
        gate_meas_var_size: float = 0.2,
        gate_chi2_thr: float = 9.21,

    ):
        super().__init__(
            device=device,
            template_image_size=template_image_size,
            search_region_image_size=search_region_image_size,
            search_curation_parameter_provider_factory=search_curation_parameter_provider_factory,
            model_output_post_process=model_output_post_process,
            segmentify_post_process=segmentify_post_process,
            interpolation_mode=interpolation_mode,
            interpolation_align_corners=interpolation_align_corners,
            norm_stats_dataset_name=norm_stats_dataset_name,
            visualization=visualization,
        )

        self.memory_max_size = memory_max_size
        self.memory_conf_thresh = memory_conf_thresh
        self.memory_banks: Dict[Any, MemoryBank] = {}

        self.call_count = 0

        self.use_memory_attention = use_memory_attention
        if self.use_memory_attention:
            self.memory_attention = MemoryAttention(
                d_model=embed_dim,
                nhead=8,
                num_layers=4,
                dim_feedforward=768,
                dropout=0.1,
                batch_first=True,
            )
        else:
            self.memory_attention = None

        # e.g. patch_size=14 for "ViT-B/14"
        self.patch_size = 14

        # --- Motion-aware gate (per task) ---
        self.motion_gates = MotionGateRegistry(
            process_var_pos=gate_process_var_pos,
            process_var_vel=gate_process_var_vel,
            process_var_size=gate_process_var_size,
            meas_var_pos=gate_meas_var_pos,
            meas_var_size=gate_meas_var_size,
            chi2_thr=gate_chi2_thr,
        )

        # ---- MB write stats (per-run) ----
        self._tau_bins = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]
        # would-write histogram: "accept_motion and conf >= tau"
        self._mb_write_hist_by_tau = {f"{tau:.1f}": 0 for tau in self._tau_bins}
        # 실제 저장 횟수(append/replace 합계)는 각 MemoryBank.write_count에서 수집
        self._mb_writes_total = 0
        self._mb_writes_by_task: Dict[Any, int] = {}
        # 출력 디렉토리 지정(선택): export MB_WRITE_LOG_DIR=/path/to/save
        self._mb_write_log_dir = os.getenv("MB_WRITE_LOG_DIR", None)

    def start(self, max_batch_size: int, global_shared_objects: dict):
        super().start(max_batch_size, global_shared_objects)

    def stop(self, global_shared_objects: dict):
        # 통계 저장 후 정리
        self._dump_mb_write_stats(global_shared_objects)
        super().stop(global_shared_objects)
        self.memory_banks.clear()
        self.motion_gates.clear()

    def begin(self, context):
        super().begin(context)
        for task in context.input_data.tasks:
            if task.task_creation_context is not None:
                self.memory_banks[task.id] = MemoryBank(
                    max_size=self.memory_max_size,
                    conf_thresh=self.memory_conf_thresh
                )
            # Kalman Gate 초기화 (GT 초기 박스 사용 가능 시)
            if task.tracker_do_init_context is not None and task.task_creation_context is not None:
                init_ctx = task.tracker_do_init_context
                gt0 = getattr(init_ctx, "gt_bbox", None)
                if gt0 is not None:
                    try:
                        self.motion_gates.init_with_gt(task.id, np.asarray(gt0, dtype=np.float64), int(init_ctx.frame_index))
                    except Exception:
                        pass

    def prepare_tracking(self, context, model_input_params: dict):
        # 1) if the raw (un-JIT’d) model has a memory_attention submodule, use it
        raw_model = context.global_objects.get('model', None)
        if raw_model is not None and self.use_memory_attention and hasattr(raw_model, 'memory_attention'):
            self.model = raw_model
            self.memory_attention = raw_model.memory_attention

        # 2) delegate to base tracker to fill model_input_params['z'], ['x'], ...
        super(OneStreamTracker_Evaluation_MainPipeline_WithMemory, self).prepare_tracking(
            context,
            model_input_params
        )

        # 3) cache: pre-encoder input x (search crop)
        context.temporary_objects['x_tensor'] = model_input_params.get('x', None)
        # 현재 배치의 크롭 파라미터도 부모가 넣어둔 것을 그대로 사용
        # (부모 prepare_tracking에서 'x_cropping_params'를 설정함)

    # --- helper: robust pre-encoder tokenizer -----------------------
    def _pre_encode_tokens(self, x_bchw: torch.Tensor) -> torch.Tensor:
        """
        Return full-frame pre-encoder tokens (B, N, C) from x (B,3,H,W).
        Tries _x_feat -> embed_x -> backbone.patch_embed, then shape-normalizes.
        """
        # Preferred: project-specific pre-encoder
        if hasattr(self.model, '_x_feat'):
            out = self.model._x_feat(x_bchw)
        elif hasattr(self.model, 'embed_x'):
            out = self.model.embed_x(x_bchw)
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'patch_embed'):
            out = self.model.backbone.patch_embed(x_bchw)  # might be (B,C,H',W') or (B,N,C)
        else:
            raise RuntimeError(
                "No pre-encoder embedding found. Provide one of: _x_feat, embed_x, backbone.patch_embed"
            )

        # Normalize shape to (B, N, C)
        if out.dim() == 3:
            return out
        elif out.dim() == 4:
            B, C, Hf, Wf = out.shape
            out = out.flatten(2).transpose(1, 2)  # (B, N, C)
            return out
        else:
            raise RuntimeError(f"Unexpected pre-encoder output shape: {tuple(out.shape)}")

    # ---------------------------------------------------------------

    def on_tracked(self, model_outputs: dict, context):
        if model_outputs is None:
            return {}

        # 1) MAL 적용 + Head (제출 전 예측 생성)
        task_ids = context.temporary_objects.get('task_ids', [])
        qs = model_outputs.get('x_feat', None)  # encoder 출력 (B, Nq, C)  ← Q는 계속 encoder 출력 사용
        x_cropping_params = context.temporary_objects.get('x_cropping_params', None)  # (B,2,2)
        H_q = self.search_region_image_size[1] // self.patch_size
        W_q = self.search_region_image_size[0] // self.patch_size

        if self.use_memory_attention and hasattr(self, 'model') and qs is not None:
            with torch.no_grad():
                fused_list = []
                for i, tid in enumerate(task_ids):
                    q_i = qs[i:i+1]
                    mem_i = None
                    if tid in self.memory_banks:
                        bank = self.memory_banks[tid]
                        # 좌표 정렬 가능한 경우 aligned concat 사용
                        x_cps = context.temporary_objects.get('x_cropping_params', None)
                        if x_cps is not None and i < len(x_cps) and hasattr(bank, 'concat_features_aligned'):
                            curr_cp = x_cps[i]  # (2,2)
                            if torch.is_tensor(curr_cp):
                                curr_cp = curr_cp.detach().cpu().numpy()
                            Hf = self.search_region_image_size[1] // self.patch_size
                            Wf = self.search_region_image_size[0] // self.patch_size
                            mem_i = bank.concat_features_aligned(
                                curr_crop_params=curr_cp,
                                current_feat_hw=(Hf, Wf),
                                patch_size=self.patch_size,
                                align_corners=False
                            )
                        if mem_i is None:
                            mem_i = bank.concat_features()
                    if mem_i is not None and mem_i.size(1) > 0:
                        q_i = self.memory_attention(q_i, mem_i)  # K,V=memory, Q=encoder 출력
                    fused_list.append(q_i)
                fused = torch.cat(fused_list, dim=0) if fused_list else qs

                head_out = self.model.head(fused)  # {'score_map', 'boxes', ...}

            # 부모 후처리로 넘길 최소 키(+x_feat=fused는 안전용)
            model_outputs = {
                'score_map': head_out['score_map'],
                'boxes':     head_out['boxes'],
                'x_feat':    fused,
            }

        # 2) 제출(부모 on_tracked는 정확히 한 번만 호출)
        submitted = super().on_tracked(model_outputs, context) or {}

        # 3) (제출 후) 메모리 쓰기: conf ≥ threshold + motion gate 통과 프레임만 FIFO 저장
        try:
            final_confs = submitted.get('final_conf', model_outputs.get('final_conf'))
        except Exception:
            final_confs = None
        final_boxes  = submitted.get('final_box',  model_outputs.get('final_box'))

        x_tensor    = context.temporary_objects.get('x_tensor', None)  # (B,3,H,W) search crop
        frame_idxs  = context.temporary_objects.get('x_frame_indices', [])
        crop_params = context.temporary_objects.get('x_cropping_params', None)

        if (
            x_tensor is not None and
            isinstance(task_ids, (list, tuple)) and
            final_confs is not None and
            final_boxes is not None
        ):
            with torch.no_grad():
                B = len(task_ids)
                for i in range(B):
                    tid  = task_ids[i]
                    bank = self.memory_banks.get(tid, None)
                    if bank is None:
                        continue

                    conf_i = final_confs[i]
                    conf = float(conf_i.item()) if torch.is_tensor(conf_i) else float(conf_i)

                    # ---- Motion-aware gate (KF) ----
                    meas_box = final_boxes[i]
                    if torch.is_tensor(meas_box):
                        meas_box = meas_box.detach().cpu().numpy().astype(np.float64)
                    frame_idx = int(frame_idxs[i]) if i < len(frame_idxs) else 0
                    d2, accept_motion = self.motion_gates.gate(tid, meas_box, frame_idx)
                    # 게이트 통과 조건: confidence & motion 일관성
                    # chi2_thr 접근은 내부 gate 객체에 의존 (디버깅 출력용)
                    try:
                        chi2_thr = self.motion_gates._gates[tid].chi2_thr
                    except Exception:
                        chi2_thr = float('nan')
                    gate_pass = (conf >= self.memory_conf_thresh) and accept_motion

                    # would-write histogram (τ ∈ {0.1..0.9}) — 같은 실행에서 비교 가능
                    for tau in self._tau_bins:
                        if accept_motion and (conf >= tau):
                            self._mb_write_hist_by_tau[f"{tau:.1f}"] += 1

                    if not gate_pass:
                        continue

                    # (a) 포스트-패치임베딩(post-embed) 토큰 저장 (encoder 출력이 아님)
                    tokens = self._pre_encode_tokens(x_tensor[i:i+1])  # (1, N, C), N=Hf*Wf

                    # (b) 메타 포함 저장 (FIFO는 MemoryBank가 수행)
                    Hf = self.search_region_image_size[1] // self.patch_size
                    Wf = self.search_region_image_size[0] // self.patch_size
                    cp_i = crop_params[i] if (crop_params is not None and i < len(crop_params)) else None
                    if torch.is_tensor(cp_i):
                        cp_i = cp_i.detach().cpu().numpy()
                    stored = bank.add(tokens, conf, frame_idx, crop_params=cp_i, feat_hw=(Hf, Wf))
                    # stored는 True일 때만 해당 프레임이 실제 MemoryBank에 기록됨

        # 4) 최종 결과 반환
        return submitted

    def end(self, context) -> Any:
        parent_result = super().end(context)
        for task in context.input_data.tasks:
            if task.do_task_finalization:
                # release 전에 per-task write_count 누적
                bank = self.memory_banks.get(task.id, None)
                if bank is not None and hasattr(bank, 'write_count'):
                    c = int(getattr(bank, 'write_count', 0))
                    self._mb_writes_total += c
                    prev = self._mb_writes_by_task.get(task.id, 0)
                    self._mb_writes_by_task[task.id] = prev + c
                _ = self.memory_banks.pop(task.id, None)
                self.motion_gates.clear(task.id)
        return parent_result

    def do_custom_update(self, compiled_model, raw_model, context):
        # evaluation 시점에는 이미 raw_model.memory_attention 에 학습된 가중치가 로드되어 있음
        self.model = raw_model if raw_model is not None else compiled_model
        if self.use_memory_attention and hasattr(self.model, 'memory_attention'):
            self.memory_attention = self.model.memory_attention
            self.memory_attention.eval()

    # -------------------------------
    # Stats writer
    # -------------------------------
    def _dump_mb_write_stats(self, global_shared_objects: dict):
        try:
            tau = float(self.memory_conf_thresh)
            tau_label = f"tau_{int(round(tau * 10)):02d}"
            # 출력 디렉토리 추정(우선순위: ENV → global_shared_objects → CWD)
            out_dir = (
                self._mb_write_log_dir
                or (isinstance(global_shared_objects, dict) and (
                    global_shared_objects.get("result_dir")
                    or global_shared_objects.get("output_dir")
                    or global_shared_objects.get("save_dir")
                    or global_shared_objects.get("work_dir")
                ))
                or os.getcwd()
            )
            os.makedirs(out_dir, exist_ok=True)
            rank = os.getenv("RANK") or os.getenv("LOCAL_RANK") or os.getenv("SLURM_PROCID") or "0"
            fname = f"mb_write_stats_{tau_label}_rank{rank}.json"
            path = os.path.join(out_dir, fname)
            payload = {
                "tau": tau,
                "tau_label": tau_label,
                "memory_max_size": int(self.memory_max_size),
                # 실제 저장(현재 tau 기준) 합계
                "writes_total_actual": int(self._mb_writes_total),
                "writes_by_task_actual": {str(k): int(v) for k, v in self._mb_writes_by_task.items()},
                # 단일 실행에서 계산한 “would-write” 히스토그램 (accept_motion ∧ conf≥τ)
                "would_write_hist_by_tau": {k: int(v) for k, v in self._mb_write_hist_by_tau.items()},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[MB-STATS] saved → {path}", flush=True)
        except Exception as e:
            print(f"[MB-STATS][WARN] failed to write stats: {e}", flush=True)

    ########################################################
    # 아래 ROI 관련 헬퍼는 더 이상 사용하지 않지만, 잔존 의존성이 있으면 유지하세요.
    # (완전 삭제해도 동작에는 영향 없음)
    def _scale_bbox_to_feature_coords(
        self,
        box_xyxy: torch.Tensor,
        search_region_image_size: tuple
    ) -> tuple:
        patch_size = self.patch_size
        W_in, H_in = search_region_image_size
        W_feat = W_in // patch_size
        H_feat = H_in // patch_size

        x1, y1, x2, y2 = box_xyxy.tolist()
        if x1 < 0: x1=0
        if y1 < 0: y1=0
        if x2 > W_in: x2=W_in
        if y2 > H_in: y2=H_in

        scale_x = W_feat / float(W_in)
        scale_y = H_feat / float(H_in)
        fx1 = int(np.floor(x1 * scale_x))
        fy1 = int(np.floor(y1 * scale_y))
        fx2 = int(np.ceil(x2 * scale_x))
        fy2 = int(np.ceil(y2 * scale_y))

        if fx2 > W_feat: fx2 = W_feat
        if fy2 > H_feat: fy2 = H_feat
        if fx1 < 0: fx1=0
        if fy1 < 0: fy1=0
        return fx1, fy1, fx2, fy2

    def _gather_subtokens(
        self,
        x_feat_2d: torch.Tensor,
        fx1: int,
        fy1: int,
        fx2: int,
        fy2: int,
        search_region_image_size: tuple
    ) -> torch.Tensor:
        patch_size = self.patch_size
        W_in, H_in = search_region_image_size
        W_feat = W_in // patch_size

        tokens_list = []
        for row in range(fy1, fy2):
            for col in range(fx1, fx2):
                idx = row * W_feat + col
                tokens_list.append(x_feat_2d[idx:idx+1, :])

        if not tokens_list:
            return torch.empty((0, x_feat_2d.shape[1]), device=x_feat_2d.device, dtype=x_feat_2d.dtype)

        return torch.cat(tokens_list, dim=0)

    def _box_to_search_coords(self, box_xyxy: torch.Tensor, context) -> torch.Tensor:
        crop_params = context.temporary_objects.get('siamfc_cropping_params', None)
        try:
            if 'final_box_in_search' in context.temporary_objects:
                return context.temporary_objects['final_box_in_search']
            if crop_params is not None:
                boxes = box_xyxy.unsqueeze(0)
                search_space_boxes = apply_siamfc_cropping_to_boxes(boxes, crop_params)
                return search_space_boxes.squeeze(0)
        except Exception:
            pass
        return box_xyxy
