import os
import numpy as np

# 백엔드 고정(서버 무디스플레이 환경)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_map(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mx = x.max()
    if not np.isfinite(mx) or mx < eps:
        return np.zeros_like(x, dtype=np.float32)
    return x / (mx + eps)


def _save_heatmap_2d(attn_2d: np.ndarray, save_path: str, title: str = None) -> None:
    """
    attn_2d: (H_feat, W_feat) normalized to [0,1]
    """
    plt.figure(figsize=(3, 3), dpi=200)
    plt.imshow(attn_2d, interpolation="nearest")
    plt.axis("off")
    if title:
        plt.title(title, fontsize=8)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_attention_map_for_memory(
    attn_weights: np.ndarray,
    q_index: int,
    mem_slices: list,
    H_feat: int,
    W_feat: int,
    out_dir: str,
    fname_prefix: str,
    frame_selector: str = "last",
) -> None:
    """
    Parameters
    ----------
    attn_weights : np.ndarray
        Shape (Nq, Nk) *또는* (Nk,) (앞에서 Nq 인덱싱을 마친 벡터).
    q_index : int
        시각화할 Query 토큰 인덱스 (예: 중앙 토큰).
    mem_slices : list[tuple]
        [(start, end, frame_idx), ...] 로, concat된 memory에서 각 프레임의 토큰 범위.
    H_feat, W_feat : int
        메모리 프레임 토큰을 (H_feat, W_feat) 격자로 복원할 해상도.
    out_dir : str
        저장 디렉토리.
    fname_prefix : str
        파일명 접두사(예: "tid7_f0123").
    frame_selector : {"last", "all"}
        "last"  : 가장 최근 메모리 프레임만 저장
        "all"   : 메모리에 쌓여있는 모든 프레임별로 저장
    """
    _ensure_dir(out_dir)

    # (Nq, Nk) → (Nk,) 로 단일 쿼리 인덱스 선택
    if attn_weights.ndim == 2:
        # (Nq, Nk)
        if q_index < 0 or q_index >= attn_weights.shape[0]:
            # 방어: 범위를 벗어나면 중앙 토큰 재계산 없이 0번째로 대체
            q_index = 0
        vec = attn_weights[q_index]  # (Nk,)
    elif attn_weights.ndim == 1:
        vec = attn_weights  # 이미 (Nk,)
    else:
        # 예상치 못한 경우 저장하지 않음
        return

    def _save_one(slice_tuple):
        s, e, fid = slice_tuple
        sub = vec[s:e]
        if sub.size != (H_feat * W_feat):
            # 크기 불일치 시, 가능한 경우 가장 앞쪽 H*W 만큼만 사용
            need = H_feat * W_feat
            if sub.size >= need:
                sub = sub[:need]
            else:
                # 저장 스킵
                return
        sub = _normalize_map(sub)
        sub = sub.reshape(H_feat, W_feat)
        out_path = os.path.join(out_dir, f"{fname_prefix}_mem{int(fid)}_q{int(q_index)}.png")
        _save_heatmap_2d(sub, out_path)

    if mem_slices and len(mem_slices) > 0:
        if frame_selector == "last":
            _save_one(mem_slices[-1])
        else:
            for tup in mem_slices:
                _save_one(tup)
    else:
        # 메모리 프레임 구분 정보를 못 받았으면, 전체를 H×W로 가정
        if vec.size >= (H_feat * W_feat):
            sub = _normalize_map(vec[: H_feat * W_feat]).reshape(H_feat, W_feat)
            out_path = os.path.join(out_dir, f"{fname_prefix}_q{int(q_index)}.png")
            _save_heatmap_2d(sub, out_path)
        # else: 저장 스킵
