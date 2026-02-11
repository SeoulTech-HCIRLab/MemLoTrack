# trackit/datasets/SOT/datasets/MyDataset.py

import os
import numpy as np

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor

class MyDataset_Seed(BaseSeed):
    def __init__(self, root_path: str = None):
        if root_path is None:
            # get the path from `consts.yaml` file
            root_path = self.get_path_from_config('MyDataset_PATH') 
        super(MyDataset_Seed, self).__init__(
            'MyDataset', # dataset name
            root_path,   # dataset root path
        )

    def construct(self, constructor: SingleObjectTrackingDatasetConstructor):
        # Implement the dataset construction logic here
        
        sequence_names = ['seq1','seq2','seq3','seq4','seq5']
        
        # Set the total number of sequences (Optional, for progress bar)
        constructor.set_total_number_of_sequences(len(sequence_names))
        
        # Set the bounding box format (Optional, 'XYXY' or 'XYWH', default for XYWH)
        constructor.set_bounding_box_format('XYWH')
        
        # get root_path
        root_path = self.root_path
        
        for sequence_name in sequence_names:
            '''
            The following is an example of the dataset structure:
            root_path
            ├── seq1
            │   ├── frames
            │   │   ├── 0001.jpg
            │   │   ├── 0002.jpg
            │   │   └── ...
            │   └── groundtruth.txt
            ├── seq2
            ...            
            '''
            with constructor.new_sequence() as sequence_constructor:
                sequence_constructor.set_name(sequence_name)
                
                sequence_path = os.path.join(root_path, sequence_name)
                # groundtruth.txt: the path of the bounding boxes file
                boxes_path = os.path.join(sequence_path, 'groundtruth.txt')
                frames_path = os.path.join(sequence_path, 'frames')
                
                # load bounding boxes using numpy
                boxes = np.loadtxt(boxes_path, delimiter=',')

                for frame_id, box in enumerate(boxes):
                    frame_path = os.path.join(frames_path, f'{frame_id + 1:06d}.jpg')
                    
                    with sequence_constructor.new_frame() as frame_constructor:
                        frame_constructor.set_path(frame_path, image_size=(1920, 1080))
                        frame_constructor.set_bounding_box(box, validity=True)