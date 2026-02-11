import os
import xml.etree.ElementTree as ET
from PIL import Image

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.DET.constructor import DetectionDatasetConstructor

class DUT_Detection_Seed(BaseSeed):
    """
    Pascal-VOC 스타일 DET 데이터셋 (1 class: UAV)
    ─────────────────────────────────────────────────────────────
    경로 구조:
      DUT_Detection_PATH/
        train/train/{img,xml}/...
        val  /val  /{img,xml}/...
        test /test /{img,xml}/...
    기본적으로 train split만 사용합니다.
    """
    def __init__(self, root_path=None, data_split=('train',)):
        if root_path is None:
            root_path = self.get_path_from_config('DUT_Detection_PATH')
        # data_split=('train',) 으로 학습 전용
        super().__init__('DUT_Detection', root_path, data_split, data_split, version=1)

    def construct(self, constructor: DetectionDatasetConstructor):
        # 1) 박스 포맷 설정 및 카테고리 맵핑
        constructor.set_bounding_box_format('XYXY')
        constructor.set_category_id_name_map({0: 'UAV'})

        # 2) 전체 이미지 개수 계산 (for progress reporting)
        total_imgs = 0
        for split in self.data_split:
            xml_dir = os.path.join(self.root_path, split, split, 'xml')
            if os.path.isdir(xml_dir):
                total_imgs += len([f for f in os.listdir(xml_dir) if f.endswith('.xml')])
        constructor.set_total_number_of_images(total_imgs)

        # 3) 실제 이미지+주석 등록
        for split in self.data_split:
            split_dir = os.path.join(self.root_path, split, split)
            img_dir   = os.path.join(split_dir, 'img')
            xml_dir   = os.path.join(split_dir, 'xml')
            if not os.path.isdir(img_dir) or not os.path.isdir(xml_dir):
                continue

            for xml_file in sorted(os.listdir(xml_dir)):
                if not xml_file.endswith('.xml'):
                    continue
                xml_path = os.path.join(xml_dir, xml_file)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # 이미지 파일 경로
                file_name = root.findtext('filename')
                img_path = os.path.join(img_dir, file_name)
                if not os.path.isfile(img_path):
                    continue

                # 이미지 크기 읽기
                with Image.open(img_path) as img:
                    size = img.size  # (width, height)

                # 객체 바운딩박스 파싱
                bboxes = []
                for obj in root.findall('object'):
                    if obj.findtext('name').lower() != 'uav':
                        continue
                    bb = obj.find('bndbox')
                    xmin = float(bb.findtext('xmin'))
                    ymin = float(bb.findtext('ymin'))
                    xmax = float(bb.findtext('xmax'))
                    ymax = float(bb.findtext('ymax'))
                    # 유효한 박스인지 체크
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    bboxes.append([xmin, ymin, xmax, ymax])

                if not bboxes:
                    continue

                # 이미지 하나에 대해 새 객체 등록
                with constructor.new_image() as ic:
                    ic.set_path(img_path, size)
                    for bb in bboxes:
                        with ic.new_object() as oc:
                            oc.set_bounding_box(bb)
                            oc.set_category_id(0)