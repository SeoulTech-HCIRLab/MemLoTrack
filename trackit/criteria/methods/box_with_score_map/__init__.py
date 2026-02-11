import torch
import torch.nn as nn
from trackit.criteria import CriterionOutput
from trackit.criteria.modules.iou_loss import bbox_overlaps
from trackit.miscellanies.torch.distributed.reduce_mean import reduce_mean_


class SimpleCriteria(nn.Module):
    def __init__(self, cls_loss: nn.Module, bbox_reg_loss: nn.Module,
                 iou_aware_classification_score: bool,
                 cls_loss_weight: float, bbox_reg_loss_weight: float,
                 cls_loss_display_name: str, bbox_reg_loss_display_name: str):
        super().__init__()
        self.cls_loss = cls_loss
        self.bbox_reg_loss = bbox_reg_loss
        self.iou_aware_classification_score = iou_aware_classification_score
        self.cls_loss_weight = cls_loss_weight
        self.bbox_reg_loss_weight = bbox_reg_loss_weight
        self.cls_loss_display_name = cls_loss_display_name
        self.bbox_reg_loss_display_name = bbox_reg_loss_display_name
        self._across_all_nodes_normalization = True

    def forward(self, outputs: dict, targets: dict):
        # 1) num_positive_samples 처리 (MemLoTrack BASE와 동일)
        num_positive_samples = targets['num_positive_samples']
        assert isinstance(num_positive_samples, torch.Tensor)
        reduce_mean_(num_positive_samples)
        num_positive_samples.clamp_(min=1.)

        # 2) forward에 사용할 텐서들 준비
        predicted_score_map = outputs['score_map'].float()
        predicted_bboxes    = outputs['boxes'].float()
        groundtruth_bboxes  = targets['boxes'].float() # 이미 [0,1] 로 라벨 생성기에서 정규화됨

        # ── 디버그 시작: NaN/Inf & 스케일 체크 ─────────────────────────
        N, H, W = predicted_score_map.shape

        # print(f"[DBG SCALE] pred_score_map.shape={predicted_score_map.shape}, "
        #        f"gt_boxes(cell units) min/max = {groundtruth_bboxes.min():.4f}/{groundtruth_bboxes.max():.4f}")
        if torch.isnan(predicted_score_map).any():   print("[DBG NaN] score_map has NaN")
        if torch.isinf(predicted_score_map).any():  print("[DBG Inf] score_map has Inf")
        if torch.isnan(predicted_bboxes).any():     print("[DBG NaN] predicted_bboxes has NaN")

        # ────────────────────────────────────────────────────────────────

        # 3) positive sample 인덱스에 맞춰 flatten & select
        positive_batch_idx = targets['positive_sample_batch_dim_indices']
        positive_map_idx   = targets['positive_sample_map_dim_indices']
        has_pos = positive_batch_idx is not None
        if has_pos:
            predicted_bboxes = predicted_bboxes.view(N, H*W, 4)
            predicted_bboxes = predicted_bboxes[positive_batch_idx, positive_map_idx]
            groundtruth_bboxes = groundtruth_bboxes[positive_batch_idx]

        # 4) response map 생성
        with torch.no_grad():
            resp_map = torch.zeros((N, H*W), dtype=torch.float32, device=predicted_score_map.device)
            if self.iou_aware_classification_score:
                resp_map.index_put_(
                    (positive_batch_idx, positive_map_idx),
                    bbox_overlaps(groundtruth_bboxes, predicted_bboxes, is_aligned=True))
            else:
                resp_map[positive_batch_idx, positive_map_idx] = 1.

        # 5) cls & reg loss 계산 (MemLoTrack BASE와 완전 동일)
        cls_loss = self.cls_loss(predicted_score_map.view(N, -1), resp_map).sum() / num_positive_samples
        if has_pos:
            reg_loss = self.bbox_reg_loss(predicted_bboxes, groundtruth_bboxes).sum() / num_positive_samples
        else:
            reg_loss = predicted_bboxes.mean() * 0

        if self.cls_loss_weight != 1.:
            cls_loss = cls_loss * self.cls_loss_weight
        if self.bbox_reg_loss_weight != 1.:
            reg_loss = reg_loss * self.bbox_reg_loss_weight

        cls_loss_cpu = cls_loss.detach().cpu().item()
        reg_loss_cpu = reg_loss.detach().cpu().item()

        metrics = {
            f'Loss/{self.cls_loss_display_name}': cls_loss_cpu,
            f'Loss/{self.bbox_reg_loss_display_name}': reg_loss_cpu
        }
        extra_metrics = {
            f'Loss/{self.cls_loss_display_name}_unscale': cls_loss_cpu / self.cls_loss_weight,
            f'Loss/{self.bbox_reg_loss_display_name}_unscale': reg_loss_cpu / self.bbox_reg_loss_weight
        }

        return CriterionOutput(cls_loss + reg_loss, metrics, extra_metrics)
