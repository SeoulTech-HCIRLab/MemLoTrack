# import os
# import json
# from PIL import Image

# from trackit.datasets.common.seed import BaseSeed
# from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor

# class Anti_UAV_300_IR_Seed(BaseSeed):
#     def __init__(self, root_path=None, data_split=('train','val','test')):
#         if root_path is None:
#             root_path = self.get_path_from_config('Anti_UAV_300_IR_PATH')
#         super().__init__('Anti_UAV_300_IR', root_path, data_split, data_split, version=1)

#     def construct(self, constructor: SingleObjectTrackingDatasetConstructor):
#         # ① XYXY 로 처리
#         constructor.set_bounding_box_format('XYXY')
#         constructor.set_category_id_name_map({0: 'UAV'})

#         # ② 전체 시퀀스 수
#         total = sum(len(os.listdir(os.path.join(self.root_path, split)))
#                     for split in self.data_split)
#         constructor.set_total_number_of_sequences(total)

#         # ③ split별 시퀀스 순회
#         for split in self.data_split:
#             split_dir = os.path.join(self.root_path, split)
#             for seq_name in os.listdir(split_dir):
#                 seq_path = os.path.join(split_dir, seq_name)
#                 if not os.path.isdir(seq_path):
#                     continue

#                 # infrared.json에서 exist, gt_rect 불러오기
#                 ann_path = os.path.join(seq_path, 'infrared.json')
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

#                     # XYWH → XYXY
#                     x1, y1 = x, y
#                     x2, y2 = x + w, y + h

#                     img_name = f'infraredI{idx:04d}.jpg'
#                     img_path = os.path.join(seq_path, 'infrared', img_name)

#                     # 이미지 크기 읽기
#                     with Image.open(img_path) as img:
#                         frame_size = img.size  # (width, height)

#                     valid_items.append((img_path, frame_size, [x1, y1, x2, y2]))

#                 if not valid_items:
#                     continue

#                 # ④ 시퀀스/프레임 등록
#                 with constructor.new_sequence(category_id=0) as seq_ctor:
#                     seq_ctor.set_name(seq_name)
#                     for img_path, frame_size, bbox in valid_items:
#                         with seq_ctor.new_frame() as f_ctor:
#                             f_ctor.set_path(img_path, frame_size)
#                             f_ctor.set_bounding_box(bbox, validity=True)



"""
SA 평가용 
"""
# trackit/datasets/SOT/datasets/Anti_UAV_300_IR.py
import os
import json
from PIL import Image

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor


class Anti_UAV_300_IR_Seed(BaseSeed):
    def __init__(self, root_path=None, data_split=('train', 'val', 'test')):
        if root_path is None:
            root_path = self.get_path_from_config('Anti_UAV_300_IR_PATH')
        # ★ 캐시 무효화: 기존 1이었다면 반드시 버전 올리기
        super().__init__('Anti_UAV_300_IR', root_path, data_split, data_split, version=2)

    def _count_images(self, seq_path: str) -> int:
        """
        infrared/infraredI0000.jpg, infraredI0001.jpg, ... 개수를 센다.
        """
        n = 0
        while os.path.isfile(os.path.join(seq_path, 'infrared', f'infraredI{n:04d}.jpg')):
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

        # split별 시퀀스 순회
        for split in self.data_split:
            split_dir = os.path.join(self.root_path, split)
            if not os.path.isdir(split_dir):
                continue

            for seq_name in sorted(os.listdir(split_dir)):
                seq_path = os.path.join(split_dir, seq_name)
                if not os.path.isdir(seq_path):
                    continue

                # anno 로드 (infrared.json)
                ann_path = os.path.join(seq_path, 'infrared.json')
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
                gt_rects   = list(ann.get('gt_rect', []))

                if len(exist_list) < T:
                    exist_list.extend([0] * (T - len(exist_list)))
                else:
                    exist_list = exist_list[:T]

                if len(gt_rects) < T:
                    gt_rects.extend([[0, 0, 0, 0]] * (T - len(gt_rects)))
                else:
                    gt_rects = gt_rects[:T]

                # 모든 프레임을 ‘반드시’ 생성 (non-exist는 validity=False + 더미 bbox)
                with constructor.new_sequence(category_id=0) as seq_ctor:
                    seq_ctor.set_name(f"Anti_UAV_300_IR_{seq_name}")

                    for idx in range(T):
                        img_name = f'infraredI{idx:04d}.jpg'
                        img_path = os.path.join(seq_path, 'infrared', img_name)
                        if not os.path.isfile(img_path):
                            # 드물게 이미지가 비는 경우만 스킵
                            continue

                        with Image.open(img_path) as img:
                            W, H = img.size

                        ex_flag = bool(exist_list[idx])
                        rect    = gt_rects[idx]

                        # 기본: non-exist → dummy + validity=False
                        bbox_xyxy = [0.0, 0.0, 0.0, 0.0]
                        validity  = False

                        # exist=True & rect 유효 → XYXY 변환
                        if ex_flag and isinstance(rect, (list, tuple)) and len(rect) == 4:
                            x, y, w, h = rect
                            if w and h and w > 0 and h > 0:
                                x1, y1 = float(x), float(y)
                                x2, y2 = x1 + float(w), y1 + float(h)
                                bbox_xyxy = [x1, y1, x2, y2]
                                validity  = True

                        with seq_ctor.new_frame() as f_ctor:
                            f_ctor.set_path(img_path, (W, H))
                            # validity=True → 존재 프레임, False → non-exist 프레임(더미)
                            f_ctor.set_bounding_box(bbox_xyxy, validity=validity)
