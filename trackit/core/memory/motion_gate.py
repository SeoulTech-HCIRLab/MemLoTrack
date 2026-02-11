# trackit/core/memory/motion_gate.py
from __future__ import annotations
import numpy as np

class _BBoxKF:
    """
    Constant-velocity Kalman filter on [cx, cy, vx, vy, logw, logh].
      - Measurement z = [cx, cy, logw, logh]
      - State      x = [cx, cy, vx, vy, logw, logh]

    강화된 게이팅 (hard + soft gate):
      1) Hard gates (측정 직전 필터링)
         - IoU(pred_box, meas_box) >= min_iou
         - size_ratio in [1/max_size_ratio, max_size_ratio]
         - center displacement <= max_center_speed_px * dt
      2) Soft gate
         - Mahalanobis d^2 < adaptive_chi2_thr
           (불확실도 작을수록 임계 감소 → 더 엄격)
    """
    def __init__(self,
                 process_var_pos: float = 50.0,
                 process_var_vel: float = 10.0,
                 process_var_size: float = 0.5,
                 meas_var_pos: float = 30.0,
                 meas_var_size: float = 0.2,
                 chi2_thr: float = 9.21,       # base (df=4, 95%)
                 # 강화 게이팅 하이퍼파라미터
                 min_iou: float = 0.10,
                 max_size_ratio: float = 2.0,  # w,h 비율 상한
                 max_center_speed_px: float = 160.0,  # px/frame
                 adaptive_chi2_beta: float = 1.5,     # 불확실도 높을수록 임계 ↑ (낮으면 엄격)
                 pos_unc_ref: float = 500.0,          # 위치 공분산 기준치     
                 ):
        self.Q_pos = process_var_pos
        self.Q_vel = process_var_vel
        self.Q_size = process_var_size
        self.R_pos = meas_var_pos
        self.R_size = meas_var_size
        self.chi2_thr = chi2_thr

        # 강화 게이팅 파라미터
        self.min_iou = float(min_iou)
        self.max_size_ratio = float(max_size_ratio)
        self.max_center_speed_px = float(max_center_speed_px)
        self.adaptive_chi2_beta = float(adaptive_chi2_beta)
        self.pos_unc_ref = float(pos_unc_ref)

        self.x = None   # (6,)
        self.P = None   # (6,6)
        self.last_f = None  # last accepted frame index

    @staticmethod
    def _bbox_to_z(box_xyxy: np.ndarray):
        x1, y1, x2, y2 = box_xyxy.astype(np.float64)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return np.array([cx, cy, np.log(w), np.log(h)], dtype=np.float64)

    @staticmethod
    def _z_to_bbox_xyxy(z: np.ndarray) -> np.ndarray:
        cx, cy, lw, lh = z
        w = float(np.exp(lw))
        h = float(np.exp(lh))
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = x1 + w
        y2 = y1 + h
        return np.array([x1, y1, x2, y2], dtype=np.float64)

    @staticmethod
    def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    def init(self, box_xyxy: np.ndarray, frame_idx: int):
        z = self._bbox_to_z(box_xyxy)
        cx, cy, lw, lh = z
        self.x = np.array([cx, cy, 0.0, 0.0, lw, lh], dtype=np.float64)

        # 초기 공분산: 위치/크기 크게, 속도 더 크게
        self.P = np.diag([1000.0, 1000.0, 500.0, 500.0, 5.0, 5.0]).astype(np.float64)
        self.last_f = frame_idx

    def _predict(self, dt: float):
        if self.x is None:
            return
        # 상태전이
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt  # cx += vx*dt
        F[1, 3] = dt  # cy += vy*dt

        # 프로세스 잡음
        Q = np.diag([
            self.Q_pos * dt*dt, self.Q_pos * dt*dt,
            self.Q_vel * dt,    self.Q_vel * dt,
            self.Q_size * dt,   self.Q_size * dt
        ]).astype(np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _innovation(self, z: np.ndarray):
        """
        z: [cx, cy, logw, logh]
        H: 4x6 (pick cx, cy, lw, lh)
        """
        H = np.zeros((4, 6), dtype=np.float64)
        H[0, 0] = 1.0  # cx
        H[1, 1] = 1.0  # cy
        H[2, 4] = 1.0  # lw
        H[3, 5] = 1.0  # lh

        R = np.diag([self.R_pos, self.R_pos, self.R_size, self.R_size]).astype(np.float64)

        y = z - (H @ self.x)                     # innovation
        S = H @ self.P @ H.T + R                 # innovation cov
        return H, R, y, S

    def _adaptive_chi2_thr(self) -> float:
        """
        위치 불확실도(Pxx, Pyy)가 낮을수록 더 엄격(임계 ↓),
        높을수록 완화(임계 ↑)되도록 조절.
        """
        pos_unc = 0.5 * (self.P[0, 0] + self.P[1, 1])
        scale = min(1.0, max(0.0, pos_unc / max(1e-6, self.pos_unc_ref)))  # 0..1
        # eff = base * (1 + beta*scale).  (불확실도 높으면 임계 ↑)
        eff = self.chi2_thr * (1.0 + self.adaptive_chi2_beta * scale)
        return float(eff)

    def gate_and_update(self, box_xyxy: np.ndarray, frame_idx: int):
        """
        반환:
          d2 (float), accept (bool)
        accept=True면 내부 상태 업데이트까지 수행.
        """
        if self.x is None:
            self.init(box_xyxy, frame_idx)
            return 0.0, True

        dt = max(1, int(frame_idx - (self.last_f if self.last_f is not None else frame_idx)))
        self._predict(float(dt))

        # ----- Hard gates -----
        z = self._bbox_to_z(box_xyxy)
        pred_z = np.array([self.x[0], self.x[1], self.x[4], self.x[5]], dtype=np.float64)
        pred_box = self._z_to_bbox_xyxy(pred_z)
        meas_box = box_xyxy.astype(np.float64)

        # 1) IoU gate
        iou = self._iou_xyxy(pred_box, meas_box)
        if iou < self.min_iou:
            # hard reject (업데이트 없이 예측만 반영)
            d2 = float("inf")
            return d2, False

        # 2) size ratio gate
        pred_w = max(1.0, np.exp(pred_z[2]))
        pred_h = max(1.0, np.exp(pred_z[3]))
        meas_w = max(1.0, np.exp(z[2]))
        meas_h = max(1.0, np.exp(z[3]))
        ratio_w = max(meas_w / pred_w, pred_w / meas_w)
        ratio_h = max(meas_h / pred_h, pred_h / meas_h)
        if (ratio_w > self.max_size_ratio) or (ratio_h > self.max_size_ratio):
            d2 = float("inf")
            return d2, False

        # 3) center displacement (speed) gate
        disp = np.hypot(z[0] - self.x[0], z[1] - self.x[1])  # px
        if disp > (self.max_center_speed_px * float(dt)):
            d2 = float("inf")
            return d2, False

        # ----- Soft (Mahalanobis) gate -----
        H, R, y, S = self._innovation(z)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S = S + 1e-6 * np.eye(4, dtype=np.float64)
            S_inv = np.linalg.inv(S)

        d2 = float(y.T @ S_inv @ y)
        eff_thr = self._adaptive_chi2_thr()
        accept = (d2 < eff_thr)

        if accept:
            K = self.P @ H.T @ S_inv
            self.x = self.x + K @ y
            self.P = (np.eye(6, dtype=np.float64) - K @ H) @ self.P
            self.last_f = frame_idx

        return d2, accept


class MotionGateRegistry:
    """
    per-task 칼만 게이트 레지스트리.
    """
    def __init__(self, **kf_kwargs):
        self._gates: dict = {}
        self._kf_kwargs = kf_kwargs

    def ensure(self, tid):
        if tid not in self._gates:
            self._gates[tid] = _BBoxKF(**self._kf_kwargs)
        return self._gates[tid]

    def init_with_gt(self, tid, gt_box_xyxy: np.ndarray, frame_idx: int):
        kf = self.ensure(tid)
        kf.init(gt_box_xyxy, frame_idx)

    def gate(self, tid, meas_box_xyxy: np.ndarray, frame_idx: int):
        kf = self.ensure(tid)
        return kf.gate_and_update(meas_box_xyxy, frame_idx)

    def clear(self, tid=None):
        if tid is None:
            self._gates.clear()
        else:
            self._gates.pop(tid, None)
