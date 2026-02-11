# # trackit/datasets/SOT/datasets/Anti_UAV_300_VIS.py
# import os
# import json
# from PIL import Image

# from trackit.datasets.common.seed import BaseSeed
# from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor

# class Anti_UAV_300_VIS_Seed(BaseSeed):
#     def __init__(self, root_path=None, data_split=('train','val','test')):
#         if root_path is None:
#             root_path = self.get_path_from_config('Anti_UAV_300_VIS_PATH')
#         super().__init__('Anti_UAV_300_VIS', root_path, data_split, data_split, version=1)

#     def construct(self, constructor: SingleObjectTrackingDatasetConstructor):
#         constructor.set_bounding_box_format('XYXY')
#         constructor.set_category_id_name_map({0: 'UAV'})

#         total = sum(len(os.listdir(os.path.join(self.root_path, split)))
#                     for split in self.data_split)
#         constructor.set_total_number_of_sequences(total)

#         for split in self.data_split:
#             split_dir = os.path.join(self.root_path, split)
#             if not os.path.isdir(split_dir):
#                 continue
#             for seq_name in sorted(os.listdir(split_dir)):
#                 seq_path = os.path.join(split_dir, seq_name)
#                 if not os.path.isdir(seq_path):
#                     continue

#                 ann_path = os.path.join(seq_path, 'visible.json')
#                 if not os.path.isfile(ann_path):
#                     continue
#                 with open(ann_path, 'r') as f:
#                     ann = json.load(f)
#                 exist_list = ann.get('exist', [])
#                 gt_rects   = ann.get('gt_rect', [])

#                 valid_items = []
#                 for idx, (rect, ex_flag) in enumerate(zip(gt_rects, exist_list), start=0):
#                     if not (ex_flag and isinstance(rect, (list, tuple)) and len(rect) == 4):
#                         continue
#                     x, y, w, h = rect
#                     if w <= 0 or h <= 0:
#                         continue
#                     w, h = max(w,1), max(h,1)
#                     x1, y1 = x, y
#                     x2, y2 = x + w, y + h

#                     img_name = f'visibleI{idx:04d}.jpg'
#                     img_path = os.path.join(seq_path, 'visible', img_name)
#                     if not os.path.isfile(img_path):
#                         continue

#                     with Image.open(img_path) as img:
#                         frame_size = img.size
#                     valid_items.append((img_path, frame_size, [x1, y1, x2, y2]))

#                 if not valid_items:
#                     continue

#                 # Prefix sequence name to avoid duplication
#                 with constructor.new_sequence(category_id=0) as seq_ctor:
#                     seq_ctor.set_name(f"Anti_UAV_300_VIS_{seq_name}")
#                     for img_path, frame_size, bbox in valid_items:
#                         with seq_ctor.new_frame() as f_ctor:
#                             f_ctor.set_path(img_path, frame_size)
#                             f_ctor.set_bounding_box(bbox, validity=True)





# =========================
# SA 평가용: 모든 프레임 + exist/non-exist 모두 포함
# =========================
# trackit/datasets/SOT/datasets/Anti_UAV_300_VIS.py
import os
import json
from PIL import Image

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor


class Anti_UAV_300_VIS_Seed(BaseSeed):
    def __init__(self, root_path=None, data_split=('train', 'val', 'test')):
        if root_path is None:
            root_path = self.get_path_from_config('Anti_UAV_300_VIS_PATH')
        # ★ 캐시 무효화: 버전 반드시 올리기 (기존 1이었다면 2로)
        super().__init__('Anti_UAV_300_VIS', root_path, data_split, data_split, version=2)

    def _count_images(self, seq_path: str) -> int:
        """
        visible/visibleI0000.jpg, visibleI0001.jpg, ... 개수를 센다.
        """
        n = 0
        while os.path.isfile(os.path.join(seq_path, 'visible', f'visibleI{n:04d}.jpg')):
            n += 1
        return n

    def construct(self, constructor: SingleObjectTrackingDatasetConstructor):
        constructor.set_bounding_box_format('XYXY')
        constructor.set_category_id_name_map({0: 'UAV'})

        # 전체 시퀀스 수 (디렉토리 기준)
        total = 0
        for split in self.data_split:
            split_dir = os.path.join(self.root_path, split)
            if os.path.isdir(split_dir):
                total += sum(
                    1 for s in os.listdir(split_dir)
                    if os.path.isdir(os.path.join(split_dir, s))
                )
        constructor.set_total_number_of_sequences(total)

        for split in self.data_split:
            split_dir = os.path.join(self.root_path, split)
            if not os.path.isdir(split_dir):
                continue

            for seq_name in sorted(os.listdir(split_dir)):
                seq_path = os.path.join(split_dir, seq_name)
                if not os.path.isdir(seq_path):
                    continue

                # anno 로드
                ann_path = os.path.join(seq_path, 'visible.json')
                if not os.path.isfile(ann_path):
                    continue

                with open(ann_path, 'r') as f:
                    ann = json.load(f)

                # 실제 이미지 개수로 전체 길이 T 확정
                T = self._count_images(seq_path)
                if T == 0:
                    continue

                # anno 길이를 T에 맞춰 패딩/절단
                exist_list = list(ann.get('exist', []))
                gt_rects = list(ann.get('gt_rect', []))

                if len(exist_list) < T:
                    exist_list.extend([0] * (T - len(exist_list)))
                else:
                    exist_list = exist_list[:T]

                if len(gt_rects) < T:
                    gt_rects.extend([[0, 0, 0, 0]] * (T - len(gt_rects)))
                else:
                    gt_rects = gt_rects[:T]

                # 모든 프레임을 ‘반드시’ 생성
                with constructor.new_sequence(category_id=0) as seq_ctor:
                    seq_ctor.set_name(f"Anti_UAV_300_VIS_{seq_name}")

                    for idx in range(T):
                        img_name = f'visibleI{idx:04d}.jpg'
                        img_path = os.path.join(seq_path, 'visible', img_name)
                        if not os.path.isfile(img_path):
                            # 드물게 이미지가 비면 그 프레임만 스킵
                            continue

                        with Image.open(img_path) as img:
                            W, H = img.size

                        ex_flag = bool(exist_list[idx])
                        rect = gt_rects[idx]

                        # 기본: non-exist → dummy + validity=False
                        bbox_xyxy = [0.0, 0.0, 0.0, 0.0]
                        validity = False

                        # exist=True & rect 유효 → XYXY
                        if ex_flag and isinstance(rect, (list, tuple)) and len(rect) == 4:
                            x, y, w, h = rect
                            if w and h and w > 0 and h > 0:
                                x1, y1 = float(x), float(y)
                                x2, y2 = x1 + float(w), y1 + float(h)
                                bbox_xyxy = [x1, y1, x2, y2]
                                validity = True

                        with seq_ctor.new_frame() as f_ctor:
                            f_ctor.set_path(img_path, (W, H))
                            # validity=True → 존재 프레임, False → non-exist 프레임(더미 bbox)
                            f_ctor.set_bounding_box(bbox_xyxy, validity=validity)
