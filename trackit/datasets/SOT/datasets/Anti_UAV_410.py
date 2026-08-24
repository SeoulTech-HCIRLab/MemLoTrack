import json
import os
import re
from numbers import Real
from typing import Iterator, Optional, Sequence, Tuple

from PIL import Image

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor


_STANDARD_EVALUATION_MODE = 'standard'
_STATE_ACCURACY_EVALUATION_MODE = 'state_accuracy'
_SUPPORTED_EVALUATION_MODES = (
    _STANDARD_EVALUATION_MODE,
    _STATE_ACCURACY_EVALUATION_MODE,
)
_SUPPORTED_DATA_SPLITS = ('train', 'val', 'test')
_FRAME_FILE_PATTERN = re.compile(r'^(\d{6})\.jpg$', re.IGNORECASE)


class Anti_UAV_410_Seed(BaseSeed):
    """Build Anti-UAV410 for standard OPE or State Accuracy evaluation.

    ``standard`` keeps only frames with a valid target bounding box, matching
    the historical AUC/precision evaluation setup in this repository.
    ``state_accuracy`` keeps every frame and represents target absence with an
    invalid dummy bounding box so the evaluation pipeline retains the state
    labels and original frame alignment.
    """

    def __init__(
        self,
        root_path: Optional[str] = None,
        data_split: Sequence[str] = _SUPPORTED_DATA_SPLITS,
        evaluation_mode: str = _STANDARD_EVALUATION_MODE,
    ):
        if root_path is None:
            root_path = self.get_path_from_config('Anti_UAV_410_PATH')
        if evaluation_mode not in _SUPPORTED_EVALUATION_MODES:
            supported = ', '.join(_SUPPORTED_EVALUATION_MODES)
            raise ValueError(
                f'Unsupported Anti-UAV410 evaluation_mode {evaluation_mode!r}; '
                f'expected one of: {supported}'
            )

        self.evaluation_mode = evaluation_mode

        # The cache version is mode-specific. Version 4 also invalidates the
        # old, manually toggled SA cache whose sequence names were prefixed.
        version = 1 if evaluation_mode == _STANDARD_EVALUATION_MODE else 4
        super().__init__(
            'Anti_UAV_410',
            root_path,
            data_split,
            _SUPPORTED_DATA_SPLITS,
            version=version,
        )

    def construct(self, constructor: SingleObjectTrackingDatasetConstructor):
        constructor.set_bounding_box_format('XYXY')
        constructor.set_bounding_box_coordinate_system('Continuous')
        constructor.set_category_id_name_map({0: 'UAV'})

        sequence_directories = tuple(self._iter_sequence_directories())
        constructor.set_total_number_of_sequences(len(sequence_directories))

        for _, sequence_name, sequence_path in sequence_directories:
            annotations = self._load_annotations(sequence_path)
            if annotations is None:
                continue

            if self.evaluation_mode == _STATE_ACCURACY_EVALUATION_MODE:
                self._construct_state_accuracy_sequence(
                    constructor, sequence_name, sequence_path, annotations
                )
            else:
                self._construct_standard_sequence(
                    constructor, sequence_name, sequence_path, annotations
                )

    def _iter_sequence_directories(self) -> Iterator[Tuple[str, str, str]]:
        for split in self.data_split:
            split_path = os.path.join(self.root_path, split)
            if not os.path.isdir(split_path):
                continue
            for sequence_name in sorted(os.listdir(split_path)):
                sequence_path = os.path.join(split_path, sequence_name)
                if os.path.isdir(sequence_path):
                    yield split, sequence_name, sequence_path

    @staticmethod
    def _load_annotations(sequence_path: str) -> Optional[dict]:
        annotation_path = os.path.join(sequence_path, 'IR_label.json')
        if not os.path.isfile(annotation_path):
            return None
        with open(annotation_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    @staticmethod
    def _xywh_to_xyxy(rect) -> Optional[list]:
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            return None
        if not all(isinstance(value, Real) for value in rect):
            return None
        x, y, width, height = rect
        if width <= 0 or height <= 0:
            return None
        return [float(x), float(y), float(x + width), float(y + height)]

    @staticmethod
    def _read_frame_size(image_path: str) -> Tuple[int, int]:
        with Image.open(image_path) as image:
            return image.size

    @staticmethod
    def _list_state_accuracy_frames(sequence_path: str) -> Tuple[Tuple[int, str], ...]:
        indexed_paths = []
        for file_name in os.listdir(sequence_path):
            match = _FRAME_FILE_PATTERN.fullmatch(file_name)
            if match is not None:
                indexed_paths.append((int(match.group(1)), os.path.join(sequence_path, file_name)))
        indexed_paths.sort(key=lambda item: item[0])

        if not indexed_paths:
            return ()

        indices = [index for index, _ in indexed_paths]
        expected_indices = list(range(1, indices[-1] + 1))
        if indices != expected_indices:
            raise ValueError(
                f'Anti-UAV410 frames must be contiguous from 000001.jpg in {sequence_path!r}; '
                f'found indices {indices[:3]}...{indices[-3:]}'
            )
        return tuple(indexed_paths)

    def _construct_standard_sequence(
        self,
        constructor: SingleObjectTrackingDatasetConstructor,
        sequence_name: str,
        sequence_path: str,
        annotations: dict,
    ) -> None:
        valid_frames = []
        groundtruth_rects = annotations.get('gt_rect', ())
        existence_flags = annotations.get('exist', ())

        for frame_index, (rect, exists) in enumerate(
            zip(groundtruth_rects, existence_flags), start=1
        ):
            bounding_box = self._xywh_to_xyxy(rect) if bool(exists) else None
            if bounding_box is None:
                continue

            image_path = os.path.join(sequence_path, f'{frame_index:06d}.jpg')
            if not os.path.isfile(image_path):
                continue
            valid_frames.append(
                (image_path, self._read_frame_size(image_path), bounding_box)
            )

        if not valid_frames:
            return

        with constructor.new_sequence(category_id=0) as sequence_constructor:
            # Preserve the historical name in standard mode.
            sequence_constructor.set_name(f'Anti_UAV_410_{sequence_name}')
            for image_path, frame_size, bounding_box in valid_frames:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path, frame_size)
                    frame_constructor.set_bounding_box(bounding_box, validity=True)

    def _construct_state_accuracy_sequence(
        self,
        constructor: SingleObjectTrackingDatasetConstructor,
        sequence_name: str,
        sequence_path: str,
        annotations: dict,
    ) -> None:
        indexed_frames = self._list_state_accuracy_frames(sequence_path)
        if not indexed_frames:
            return

        total_frames = len(indexed_frames)
        existence_flags = list(annotations.get('exist', ()))
        groundtruth_rects = list(annotations.get('gt_rect', ()))
        if len(existence_flags) != total_frames or len(groundtruth_rects) != total_frames:
            raise ValueError(
                f'Anti-UAV410 frame/annotation length mismatch in {sequence_path!r}: '
                f'images={total_frames}, exist={len(existence_flags)}, '
                f'gt_rect={len(groundtruth_rects)}'
            )

        with constructor.new_sequence(category_id=0) as sequence_constructor:
            # The official toolkit expects the raw Anti-UAV410 sequence name.
            sequence_constructor.set_name(sequence_name)

            for (frame_index, image_path), exists, rect in zip(
                indexed_frames, existence_flags, groundtruth_rects
            ):
                if frame_index < 1:
                    raise ValueError(f'Invalid frame index {frame_index} in {sequence_path!r}')

                bounding_box = self._xywh_to_xyxy(rect) if bool(exists) else None
                is_valid = bounding_box is not None
                if bounding_box is None:
                    bounding_box = [0.0, 0.0, 0.0, 0.0]

                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(
                        image_path, self._read_frame_size(image_path)
                    )
                    frame_constructor.set_bounding_box(
                        bounding_box, validity=is_valid
                    )
