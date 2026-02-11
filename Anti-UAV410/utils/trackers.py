# utils/trackers.py
import os

"""
mode 1: (x,y,w,h)
mode 2: (x1,y1,x2,y2)

Trained_Trackers: AntiUAV410로 fine-tune/학습된 결과만 평가
- 아래 base_dir만 변경하면 전체 경로가 자동으로 맞춰짐.
"""

# >>> 필요한 경우 여기 한 줄만 환경에 맞게 바꾸세요 <<<
base_dir = "/home/gpuadmin/jk/410_results_evaluation/Anti-UAV410/Tracking_results/Trained_with_antiuav410"

def P(*parts):  # path join helper
    return os.path.join(base_dir, *parts)

Trained_Trackers = [
    {'name': 'AiATrack',       'path': P('AiATrack', 'baseline'),                                              'mode': 1},
    {'name': 'ATOM',           'path': P('ATOM', 'default'),                                                   'mode': 1},
    {'name': 'DiMP50',         'path': P('DiMP', 'dimp50'),                                                    'mode': 1},
    {'name': 'Super_DiMP',     'path': P('DiMP', 'super_dimp'),                                                'mode': 1},
    {'name': 'KeepTrack',      'path': P('KeepTrack'),                                                         'mode': 1},
    {'name': 'Stark-ST101',    'path': P('Stark-ST101'),                                                       'mode': 1},
    {'name': 'SwinTrack-Tiny', 'path': P('SwinTrack-Tiny'),                                                    'mode': 2},
    {'name': 'SwinTrack-Base', 'path': P('SwinTrack-Base'),                                                    'mode': 2},
    {'name': 'TCTrack',        'path': P('TCTrack'),                                                           'mode': 1},
    {'name': 'ToMP50',         'path': P('ToMP', 'tomp50'),                                                    'mode': 1},
    {'name': 'ToMP101',        'path': P('ToMP', 'tomp101'),                                                   'mode': 1},
    {'name': 'DropTrack',      'path': P('DropTrack', 'ostrack', 'vitb_384_mae_ce_32x4_got10k_ep100'),         'mode': 1},
    {'name': 'MixformerV2-B',  'path': P('MixFormerV2', 'mixformer2_vit_online', '288_depth8_score'),          'mode': 1},
    {'name': 'MemLoTrack(MB=1)',     'path': P('MemLoTrack', 'MB_size_1'),                                                        'mode': 1},
    {'name': 'MemLoTrack(MB=3)',     'path': P('MemLoTrack', 'MB_size_3'),                                                        'mode': 1},
    {'name': 'MemLoTrack(MB=7)',     'path': P('MemLoTrack', 'MB_size_7'),                                                        'mode': 1},
    {'name': 'MemLoTrack(MB=11)',     'path': P('MemLoTrack', 'MB_size_11'),                                                        'mode': 1},
    {'name': 'FocusTrack',     'path': P('FocusTrack',  'FocusTrack'),                                         'mode': 1}, 
    {'name': 'ROMTrack',       'path': P('ROMTrack'),                                         'mode': 1}, 
    {'name': 'ZoomTrack',       'path': P('ZoomTrack'),                                         'mode': 1},
    {'name': 'LoRAT-Base',       'path': P('LoRAT-Base'),                                         'mode': 1}, 
 
]

