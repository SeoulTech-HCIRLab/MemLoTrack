# trackit/core/memory/memory_bank.py
"""
MemoryBank with optional coordinate-aligned concatenation.

기능 요약
- 각 메모리 아이템이 생성될 때 사용된 search crop의 좌표계 파라미터 (scale S, translation T)를 함께 저장합니다.
- 현재 프레임(Q)의 그리드와 과거 메모리(K,V)의 그리드를 좌표 정렬(워핑)하여,
  동일한 (i,j) 토큰 위치가 같은 실제 영상 위치를 바라보도록 보정한 뒤 토큰을 연결(concat)할 수 있습니다.
- 정렬 정보가 없으면(파라미터가 None) 기존 방식처럼 단순 연결로 폴백합니다.

Tensor Shapes
- feat: (1, N, C)  # 토큰 시퀀스 형태 (배치=1 가정)
- 정렬 후 concat_features_aligned 결과: (1, num_items * H_q*W_q, C)
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MemoryItem:
    """
    한 프레임의 메모리 아이템 단위.

    Args:
        feat: (1, N, C) shaped tokens (e.g., encoder output flattened)
        conf: Confidence score (float).
        frame_idx: Which frame number this memory item corresponds to.
        crop_params: (2,2) numpy array
            row0: scale (sx, sy)
            row1: translation (tx, ty)
            이렇게 정의된 2x2 행렬로, SiamFC 스타일 크롭에서 out = S * x + T의 per-axis 버전.
        feat_hw: (H_feat, W_feat)  # 이 feat가 2D로 reshape될 때의 토큰 그리드 크기
    """
    def __init__(
        self,
        feat: torch.Tensor,
        conf: float,
        frame_idx: int,
        crop_params: Optional[np.ndarray] = None,
        feat_hw: Optional[Tuple[int, int]] = None
    ):
        self.feat = feat
        self.conf = conf
        self.frame_idx = frame_idx
        self.crop_params = crop_params
        self.feat_hw = feat_hw

    def __repr__(self) -> str:
        hw = None if self.feat_hw is None else f"{self.feat_hw}"
        has_cp = self.crop_params is not None
        return (f"MemoryItem(frame={self.frame_idx}, conf={self.conf:.3f}, "
                f"feat_shape={tuple(self.feat.shape)}, feat_hw={hw}, "
                f"crop_params={'Y' if has_cp else 'N'})")


class MemoryBank:
    """
    단순/정렬형 메모리 은행.

    Args:
        max_size: 최대 보관 개수 (FIFO)
        conf_thresh: 이 값 미만의 conf는 저장하지 않음
    """
    def __init__(
        self,
        max_size: int = 7,
        conf_thresh: float = 0.8
    ):
        assert max_size >= 1, "max_size must be >= 1"
        self.max_size = max_size
        self.conf_thresh = conf_thresh
        self.items: List[MemoryItem] = []
        self._DBG = (os.getenv("MB_ALIGN_DEBUG", "") == "1")

    # -----------------------------
    # Basic container utilities
    # -----------------------------
    def __len__(self) -> int:
        return len(self.items)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def clear(self) -> None:
        """메모리 아이템을 모두 삭제합니다."""
        self.items.clear()

    def _prune(self) -> None:
        """max_size를 초과하면 가장 오래된 아이템부터 제거(FIFO)."""
        while len(self.items) > self.max_size:
            self.items.pop(0)

    def _find_duplicate_index(self, frame_idx: int) -> int:
        """같은 frame_idx를 가진 아이템이 있으면 그 인덱스를, 없으면 -1을 반환."""
        for i, it in enumerate(self.items):
            if it.frame_idx == frame_idx:
                return i
        return -1

    # -----------------------------
    # Add / Get
    # -----------------------------
    def add(
        self,
        feat: torch.Tensor,
        conf: float,
        frame_idx: int,
        *,
        crop_params: Optional[np.ndarray] = None,
        feat_hw: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        메모리 추가 (동일 frame_idx 존재 시 conf 높은 쪽으로 교체).

        요구사항:
            - conf < conf_thresh 이면 drop
            - feat shape: (1, N, C)
        """
        if conf < self.conf_thresh:
            return

        assert torch.is_tensor(feat), "feat must be a torch.Tensor"
        assert feat.dim() == 3 and feat.shape[0] == 1, "feat must be (1, N, C)"

        dup_idx = self._find_duplicate_index(frame_idx)

        new_item = MemoryItem(feat, conf, frame_idx, crop_params, feat_hw)

        # 간단 검증: feat_hw가 있으면 N == H*W인지 확인
        if feat_hw is not None:
            Hm, Wm = feat_hw
            N = int(feat.shape[1])
            if N != Hm * Wm and self._DBG:
                print(f"[MB-WARN] frame={frame_idx} N({N}) != Hm*Wm({Hm*Wm}). "
                      f"정렬시 grid_sample에 재배치/보간이 과도할 수 있습니다.", flush=True)

        if dup_idx >= 0:
            if conf > self.items[dup_idx].conf:
                self.items[dup_idx] = new_item
            # conf가 낮으면 skip
            return

        self.items.append(new_item)
        if len(self.items) > self.max_size:
            self._prune()

    def get_items(self) -> List[MemoryItem]:
        """외부에서 읽기용으로 아이템 목록을 반환(참조 주의)."""
        return self.items

    # -----------------------------
    # Concatenation (no alignment)
    # -----------------------------
    @torch.no_grad()
    def concat_features(self) -> Optional[torch.Tensor]:
        """
        정렬 정보 없이 단순히 feature들을 토큰 차원으로 연결.

        Returns:
            None (비어있으면) 또는 (1, sum_N, C)
        """
        if len(self.items) == 0:
            return None

        feats = []
        for it in self.items:
            # (1, N, C)
            feats.append(it.feat)
        return torch.cat(feats, dim=1) if feats else None

    # -----------------------------
    # Concatenation (aligned)
    # -----------------------------
    @torch.no_grad()
    def concat_features_aligned(
        self,
        *,
        curr_crop_params: np.ndarray,                # (2,2) for current frame (the Q crop used now)
        current_feat_hw: Tuple[int, int],            # (H_q, W_q) tokens of current Q (e.g., 16x16)
        patch_size: int,                             # e.g., 14
        align_corners: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Warp each stored memory item from *its own crop* into the *current Q grid*,
        then concatenate along tokens: shape -> (1, H_q*W_q * num_items, C).

        Coordinate model:
          - Cropping uses affine y = S * x + T (per-axis scale+translation).
          - For current frame (c): out_c = S_c * x + T_c
          - For memory  frame (m): out_m = S_m * x + T_m
          - We want: for each Q-grid location out_c, find corresponding out_m:
              out_m = (S_m/S_c) * out_c + (T_m - (S_m/S_c) * T_c)
          - Then convert pixel coords (out_m) -> memory feature index coords (j_m_cont, i_m_cont)
            by dividing by patch_size and subtracting 0.5 (patch center).
          - Finally map to grid_sample's normalized coords [-1,1] with proper align_corners.

        Fallback:
          - 저장된 아이템 중 crop_params 또는 feat_hw 가 하나라도 없으면
            좌표 정렬이 불가능하므로 concat_features() 방식으로 폴백합니다.

        Returns:
            None (비어있으면) 또는 (1, num_items * H_q*W_q, C)
        """
        if len(self.items) == 0:
            return None

        # 좌표 정렬 가능한지 검사
        has_params = all((it.crop_params is not None and it.feat_hw is not None) for it in self.items)
        if not has_params:
            if self._DBG:
                miss = [it.frame_idx for it in self.items if (it.crop_params is None or it.feat_hw is None)]
                print(f"[MB-ALIGN][FALLBACK] meta missing for frames={miss}. Use concat_features().", flush=True)
            return self.concat_features()

        H_q, W_q = current_feat_hw

        # device, dtype 일관화
        device = self.items[0].feat.device
        dtype = self.items[0].feat.dtype

        # (i,j) grid for Q tokens
        js = torch.arange(W_q, device=device, dtype=dtype)  # 0..W_q-1  (x / column)
        is_ = torch.arange(H_q, device=device, dtype=dtype) # 0..H_q-1  (y / row)
        jj, ii = torch.meshgrid(js, is_, indexing='xy')     # jj: (W_q,H_q), ii: (W_q,H_q)

        # 현재 Q 그리드의 픽셀 중심좌표 (224x224 crop 좌표계)
        u_c = (jj + 0.5) * float(patch_size)                # (W_q,H_q)
        v_c = (ii + 0.5) * float(patch_size)                # (W_q,H_q)
        # grid_sample에 넣기 용이하도록 (H_q, W_q) 형태로 전치
        u_c = u_c.T.contiguous()
        v_c = v_c.T.contiguous()

        # 현재 프레임 crop params
        assert isinstance(curr_crop_params, np.ndarray) and curr_crop_params.shape == (2, 2)
        scale_c, trans_c = curr_crop_params
        sx_c, sy_c = float(scale_c[0]), float(scale_c[1])
        tx_c, ty_c = float(trans_c[0]), float(trans_c[1])

        warped_list: List[torch.Tensor] = []

        for it in self.items:
            feat = it.feat  # (1, N, C)
            Hm, Wm = it.feat_hw
            Cm = feat.shape[-1]

            # (1, C, Hm, Wm)
            feat_2d = feat.transpose(1, 2).reshape(1, Cm, Hm, Wm)

            scale_m, trans_m = it.crop_params
            sx_m, sy_m = float(scale_m[0]), float(scale_m[1])
            tx_m, ty_m = float(trans_m[0]), float(trans_m[1])

            # out_m = (S_m/S_c) * out_c + (T_m - (S_m/S_c) * T_c)
            fx = sx_m / sx_c
            fy = sy_m / sy_c
            bx = tx_m - fx * tx_c
            by = ty_m - fy * ty_c

            # 현재 Q 픽셀 중심좌표 -> 메모리 크롭 좌표
            u_m = fx * u_c + bx  # (H_q, W_q)
            v_m = fy * v_c + by  # (H_q, W_q)

            # 메모리 feature index 연속좌표 (열=j, 행=i), 패치 중심 정렬 고려
            j_m = (u_m / float(patch_size)) - 0.5  # (H_q, W_q)
            i_m = (v_m / float(patch_size)) - 0.5  # (H_q, W_q)

            # grid_sample용 정규화 [-1,1]
            if align_corners:
                grid_x = 2.0 * j_m / max(Wm - 1, 1) - 1.0
                grid_y = 2.0 * i_m / max(Hm - 1, 1) - 1.0
            else:
                grid_x = (2.0 * j_m + 1.0) / float(Wm) - 1.0
                grid_y = (2.0 * i_m + 1.0) / float(Hm) - 1.0

            grid = torch.stack([grid_x, grid_y], dim=-1)  # (H_q, W_q, 2)
            grid = grid.unsqueeze(0).to(dtype=dtype, device=device)

            # (1, C, H_q, W_q)
            warped = F.grid_sample(
                feat_2d, grid,
                mode='bilinear', padding_mode='zeros',
                align_corners=align_corners
            )

            # (1, H_q*W_q, C)
            warped_tokens = warped.permute(0, 2, 3, 1).reshape(1, H_q * W_q, Cm)
            warped_list.append(warped_tokens)

        if not warped_list:
            return None

        out = torch.cat(warped_list, dim=1)  # (1, num_items*H_q*W_q, C)
        if self._DBG:
            n_items = len(self.items)
            print(f"[MB-ALIGN][OK] items={n_items} -> aligned_tokens={out.shape[1]} "
                  f"(per={H_q*W_q}) HqWq={H_q}x{W_q}", flush=True)
        return out

    # (디버그 편의) 총 토큰 수
    def total_tokens(self) -> int:
        return sum(int(it.feat.shape[1]) for it in self.items)


