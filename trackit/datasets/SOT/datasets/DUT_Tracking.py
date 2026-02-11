# trackit/datasets/SOT/datasets/DUT_Tracking.py

import os
import re
from PIL import Image

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor

class DUT_Tracking_Seed(BaseSeed):
    """
    DUT Tracking (Anti-UAV-Tracking-V0 & V0GT):
    - No train/val/test folders; treat entire V0 directory as single test split.
    - Use full GT if available, else fallback to first-frame GT.
    - Convert 1-based XYWH → XYXY.
    """
    def __init__(self, root_path=None):
        if root_path is None:
            root_path = self.get_path_from_config('DUT_Tracking_PATH')
        # Pass empty tuples so BaseSeed does not look for split subfolders
        super().__init__('DUT_Tracking', root_path, (), (), version=1)

    def _load_gt(self, seq_name: str):
        v0    = os.path.join(self.root_path, 'Anti-UAV-Tracking-V0')
        v0gt  = os.path.join(self.root_path, 'Anti-UAV-Tracking-V0GT')
        full  = os.path.join(v0gt,  f'{seq_name}_gt.txt')
        first = os.path.join(v0,   seq_name, f'{seq_name}_gt_first.txt')
        path  = full if os.path.isfile(full) else first

        gts = []
        with open(path) as f:
            for line in f:
                vals = [float(x) for x in re.split(r'\s+', line.strip()) if x]
                if len(vals) != 4:
                    continue
                x, y, w, h = vals
                if x < 0 or w <= 0 or h <= 0:
                    gts.append((False, None))
                else:
                    gts.append((True, [x, y, x + w, y + h]))
        return gts

    def construct(self, constructor: SingleObjectTrackingDatasetConstructor):
        # 1) format & category
        constructor.set_bounding_box_format('XYXY')
        constructor.set_category_id_name_map({0: 'UAV'})

        # 2) gather sequence names
        seq_root  = os.path.join(self.root_path, 'Anti-UAV-Tracking-V0')
        seq_names = sorted(
            d for d in os.listdir(seq_root)
            if os.path.isdir(os.path.join(seq_root, d))
        )
        constructor.set_total_number_of_sequences(len(seq_names))

        # 3) register each sequence
        for seq_name in seq_names:
            gt_info = self._load_gt(seq_name)
            if not gt_info:
                continue
            with constructor.new_sequence(category_id=0) as seq_ctor:
                seq_ctor.set_name(seq_name)
                img_dir = os.path.join(seq_root, seq_name)
                for idx, (valid, bbox) in enumerate(gt_info, start=1):
                    img_path = os.path.join(img_dir, f'{idx:05d}.jpg')
                    if not os.path.isfile(img_path):
                        continue
                    with Image.open(img_path) as img:
                        size = img.size
                    with seq_ctor.new_frame() as f_ctor:
                        f_ctor.set_path(img_path, size)
                        f_ctor.set_bounding_box(bbox if valid else [0,0,0,0],
                                                validity=valid)
