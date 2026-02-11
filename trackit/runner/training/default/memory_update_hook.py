# import torch
# import numpy as np
# from trackit.core.memory.memory_bank import MemoryBank

# def update_memory_bank_training(batch_output: dict, batch_input: dict,
#                                 memory_banks: dict, search_region_image_size: tuple,
#                                 patch_size: int, device: torch.device):
    
    
#     # print(f"[MEMUP_DBG] Entered memory_update_hook; batch_output keys = {list(batch_output.keys())}")

#     # 배치 출력에서 memory_frames 키를 가져옴
#     memory_features = batch_output.get('memory_frames', None)
    
#     # print(f"[MEMUP_DBG] memory_features = {type(memory_features)}; value = {memory_features}")

#     if memory_features is None:
#         # print("[MemoryUpdateHook] Warning: 'memory_frames' not found in batch_output, skipping memory update.")
#         return

#     # memory_features는 배치의 각 샘플마다 여러 메모리 프레임 임베딩(리스트 또는 tensor)로 구성됨
#     B = len(memory_features)

#     if 'task_ids' not in batch_input:
#         batch_input['task_ids'] = torch.zeros(B, dtype=torch.int, device=device)
#         # print("[MemoryUpdateHook] 'task_ids' not found; defaulting to zeros.")
#     if 'frame_indices' not in batch_input:
#         batch_input['frame_indices'] = torch.zeros(B, dtype=torch.int, device=device)
#         # print("[MemoryUpdateHook] 'frame_indices' not found; defaulting to zeros.")

#     task_ids = batch_input['task_ids']
#     base_frame_indices = batch_input['frame_indices']

#     for i in range(B):
#         key = int(task_ids[i].item() if hasattr(task_ids[i], 'item') else task_ids[i])
#         base_frame_idx = int(base_frame_indices[i].item() if hasattr(base_frame_indices[i], 'item') else base_frame_indices[i])
#         sample_memory = memory_features[i]
#         # 만약 sample_memory가 tensor이면 (예: (L, C) shape)
#         if isinstance(sample_memory, torch.Tensor):
#             # ── MEMUP: sample_memory 텐서 통계 ────────────────────────
#             # print(f"[MEMUP] sample_memory[{i}] shape = {sample_memory.shape}")
#             # print(f"[MEMUP] sample_memory min/max/mean/std = "
#             #       f"{sample_memory.min().item():.4f}/"
#             #       f"{sample_memory.max().item():.4f}/"
#             #       f"{sample_memory.mean().item():.4f}/"
#             #       f"{sample_memory.std().item():.4f}")
#             if torch.isnan(sample_memory).any():  print("[MEMUP] sample_memory has NaN")
#             if torch.isinf(sample_memory).any(): print("[MEMUP] sample_memory has Inf")

#             num_frames = sample_memory.shape[0]
#             for j in range(num_frames):
#                 mem_feat = sample_memory[j].unsqueeze(0)  # shape: (1, C)
#                 # ── MEMUP: mem_feat 통계 ─────────────────────────────────
#                 # print(f"[MEMUP] mem_feat[{i},{j}] min/max/mean/std = "
#                 #       f"{mem_feat.min().item():.4f}/"
#                 #       f"{mem_feat.max().item():.4f}/"
#                 #       f"{mem_feat.mean().item():.4f}/"
#                 #       f"{mem_feat.std().item():.4f}")
#                 if torch.isnan(mem_feat).any():  print("[MEMUP] mem_feat has NaN")
#                 if torch.isinf(mem_feat).any(): print("[MEMUP] mem_feat has Inf")

#                 conf = 0.9  # 필요에 따라 confidence 계산
#                 if key not in memory_banks:
#                     memory_banks[key] = MemoryBank(max_size=7, conf_thresh=0.9)
#                 memory_banks[key].add(mem_feat.to(device), conf, base_frame_idx + j)
#                 # print(f"[MemoryUpdateHook] Task {key} MemoryBank size: {memory_banks[key].size()}")
#         # sample_memory가 리스트이면
#         elif isinstance(sample_memory, list):
#             for j, mem in enumerate(sample_memory):
#                 mem_feat = mem.unsqueeze(0)
#                 conf = 0.9
#                 if key not in memory_banks:
#                     memory_banks[key] = MemoryBank(max_size=7, conf_thresh=0.9)
#                 memory_banks[key].add(mem_feat.to(device), conf, base_frame_idx + j)
#                 # print(f"[MemoryUpdateHook] Task {key} MemoryBank size: {memory_banks[key].size()}")
#         else:
#             # print(f"[MemoryUpdateHook] Unexpected type for memory_features[{i}]: {type(sample_memory)}")
#             pass
