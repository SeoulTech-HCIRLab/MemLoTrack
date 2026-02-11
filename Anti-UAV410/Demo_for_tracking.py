from __future__ import absolute_import

import unittest
import os

from trackers.SiamFC.siamfc import TrackerSiamFC
from experiments import ExperimentAntiUAV410


"""
Experiments Setup
"""

dataset_path='path/to/Anti-UAV410'

# test or val
subset='test'

net_path = './Trackers/SiamFC/model.pth'

#LoRAT-B-224 - Trained Model path : /home/gpuadmin/jk/lorat_b_410_results/LoRAT-dinov2-mixin-anti_uav_train-mixin-anti_uav_test-2025.08.12-17.02.46-900241/checkpoint/epoch_49/model.safetensors 


tracker = TrackerSiamFC(net_path=net_path)

# run experiment
experiment = ExperimentAntiUAV410(root_dir=dataset_path, subset=subset)

experiment.run(tracker, visualize=True)
# report performance
experiment.report([tracker.name])