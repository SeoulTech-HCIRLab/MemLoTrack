from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from .modules.lora.apply import find_all_frozen_nn_linear_names, apply_lora
from timm.models.layers import trunc_normal_
from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.head.mlp import MlpAnchorFreeHead
from collections import OrderedDict

from trackit.core.memory.memory_attention import MemoryAttention

#==== DEBUG helper functions ====
def debug_memory_frames_structure(memory_frames):
    # print("=== Debug: Memory Frames Structure ===")
    if isinstance(memory_frames, list):
        # print(f"Total samples in batch: {len(memory_frames)}")
        for b, sample in enumerate(memory_frames):
            # print(f" Sample {b}: Type = {type(sample)}")
            if isinstance(sample, list):
                # print(f"  Sample {b} has {len(sample)} memory frames:")
                for i, frame in enumerate(sample):
                    # frame_type = type(frame)
                    # frame_shape = getattr(frame, "shape", None)
                    # print(f"    Memory frame {i}: Type = {frame_type}, Shape = {frame_shape}")
                    pass
            else:
                # frame_shape = getattr(sample, "shape", None)
                # print(f"  Memory frame: Type = {type(sample)}, Shape = {frame_shape}")
                pass
    else:
        # print(f"Memory frames is not a list; type: {type(memory_frames)}")
        pass
    # print("======================================")


# 기존 import 등은 그대로 유지 (예: PatchEmbed, positional embedding, token_type_embed, self.blocks 등)
class MemLoTrack_DINOv2(nn.Module):
    def __init__(self, 
                 vit: DinoVisionTransformer,
                 template_feat_size: tuple,
                 search_region_feat_size: tuple,
                 lora_r: int, 
                 lora_alpha: float, 
                 lora_dropout: float, 
                 use_rslora: bool = False,
                 use_memory_attention: bool = True):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        # 1) Backbone 세팅
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks      = vit.blocks
        self.norm        = vit.norm
        self.embed_dim   = vit.embed_dim
     
     # 2) Positional Embedding (MemLoTrack BASE 방식)
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim)
        )
        self.pos_embed.data.copy_(
            interpolate_pos_encoding(
                vit.pos_embed.data[:, 1:, :],
                self.x_size,
                vit.patch_embed.patches_resolution,
                num_prefix_tokens=0,
                interpolate_offset=0
            )
        )
        print("=> interpolated pos_embed.shape:", self.pos_embed.shape)

        # ── 여기까지 backbone, norm, pos_embed 파라미터 freeze ──
        # (이후 생성될 token_type_embed, LoRA adapter, head 파라미터만 trainable)
        for param in self.parameters():
            param.requires_grad = False

        # LoRA 하이퍼파라미터 메타데이터 저장
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora


        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=0.02)

        # 4) Freeze된 Linear들에 LoRA 어댑터 적용 (LoRA 파라미터는 기본적으로 trainable)
        for block in self.blocks:
            linear_names = find_all_frozen_nn_linear_names(block)
            apply_lora(block, linear_names, lora_r, lora_alpha, lora_dropout, use_rslora)

        # 5) Head 생성 (head 파라미터는 기본적으로 trainable)
        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)
        
        
        self.use_memory_attention = use_memory_attention


        if self.use_memory_attention:
            self.memory_attention = MemoryAttention(
                d_model=self.embed_dim,
                nhead=8,
                num_layers=4,
                dim_feedforward=3072,
                dropout=0.1,
                batch_first=True
            )
        else:
            self.memory_attention = None


        # ─── 초기화 확인용 디버그 ─────────────────────────────
        # 회귀 헤드 마지막 레이어(weight, bias)의 분포를 찍어봅니다.
        w = self.head.reg_mlp.layers[-1].weight
        b = self.head.reg_mlp.layers[-1].bias
        print(f"[Init Debug] reg_mlp last weight mean/var = {w.mean().item():.6f}/{w.var().item():.6f}")
        print(f"[Init Debug] reg_mlp last bias       = {b.detach().cpu().numpy()}")
        # ──────────────────────────────────────────────────────

        self.patch_size = 14

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor) -> torch.Tensor:
        z = self.patch_embed(z)
        z_W, z_H = self.z_size
        slice_len = z_H * z_W
        pos_z = self.pos_embed[:, :slice_len, :]
        z = z + pos_z
        if z_feat_mask is not None:
            if z_feat_mask.ndim > 2:
                z_feat_mask = z_feat_mask.view(z_feat_mask.shape[0], -1)
            B, L = z_feat_mask.shape
            if L != z_H * z_W:
                raise ValueError(f"z_feat_mask has {L} tokens, expected {z_H*z_W}")
            token_type_emb = self.token_type_embed[z_feat_mask.view(-1)]
            token_type_emb = token_type_emb.view(B, L, self.embed_dim)
            z = z + token_type_emb
        # print("[Z_FEAT Debug] z_feat shape:", z.shape)
        return z

    def _x_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = x + self.token_type_embed[2].view(1, 1, self.embed_dim)
        # print("[X_FEAT Debug] x_feat shape:", x.shape)
        return x

    def _mem_feat(self, memory_frames: List[List[torch.Tensor]]) -> torch.Tensor:
        # 1) flatten frames
        flat: List[torch.Tensor] = []
        for sample_mem in memory_frames:
            for img in sample_mem:
                flat.append(img)

        # 2) empty case
        if len(flat) == 0:
            N, D = self.pos_embed.shape[1], self.pos_embed.shape[2]
            device = next(self.patch_embed.parameters()).device
            return torch.empty(0, N, D, device=device, dtype=self.pos_embed.dtype)

        # 3) stack & to device
        device = next(self.patch_embed.parameters()).device
        imgs = torch.stack(flat, dim=0).to(device, non_blocking=True)  # [total, C, H, W]

        # 4) patch‑embed + pos‑embed 
        emb = self.patch_embed(imgs)      # [total, N_tokens, D]
        tokens = emb + self.pos_embed     # broadcast over batch
        tokens = tokens + self.token_type_embed[1].view(1,1,-1)
        return tokens


    def _fusion(self, z_feat: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        if z_feat.shape[0] != x_feat.shape[0]:
            if z_feat.shape[0] == 1 and x_feat.shape[0] > 1:
                z_feat = z_feat.expand(x_feat.shape[0], -1, -1)
            elif x_feat.shape[0] == 1 and z_feat.shape[0] > 1:
                x_feat = x_feat.expand(z_feat.shape[0], -1, -1)
            else:
                raise RuntimeError(f"Batch size mismatch: z_feat={z_feat.shape[0]}, x_feat={x_feat.shape[0]}")
        if not hasattr(self, '_template_token_length'):
            self._template_token_length = z_feat.shape[1]
            # print("[FUSION Debug] Set template token length:", self._template_token_length)
        fusion_feat = torch.cat((z_feat, x_feat), dim=1)
        # print("[FUSION Debug] contains NaN?", fusion_feat.isnan().any().item())
        # print("[FUSION Debug] Before transformer blocks, fusion_feat shape:", fusion_feat.shape)

        for i,block in enumerate(self.blocks):
            fusion_feat = block(fusion_feat)
            # print(f"[FUSION Debug] after block {i}, NaN?", fusion_feat.isnan().any().item())
        fusion_feat = self.norm(fusion_feat)

        # print("[FUSION Debug] After norm, fusion_feat shape:", fusion_feat.shape)
        fused = fusion_feat[:, self._template_token_length:, :]
        # print("[FUSION Debug] Final fused feature shape (search region part):", fused.shape)
        return fused
    
    def forward(self,
                z: torch.Tensor,
                x: torch.Tensor,
                z_feat_mask: torch.Tensor,
                use_memory_mask: torch.Tensor = None,
                memory_frames: list = None,
                **kwargs) -> dict:
        # print(f"[FWD Debug] use_memory_mask received: {use_memory_mask} (shape={None if use_memory_mask is None else use_memory_mask.shape})")
        # print(f"[FWD Debug] memory_frames received, batch size = {None if memory_frames is None else len(memory_frames)}")
        # print("[FORWARD FLAG_DBG] use_memory_mask =", 
        #       use_memory_mask.tolist() if isinstance(use_memory_mask, torch.Tensor) else use_memory_mask)
        # print("[FORWARD MEM_DBG] memory_frames lengths =", 
        #       [len(mf) for mf in (memory_frames or [])])

        # 1) debug: memory_frames 구조
        # print("[FORWARD Debug] Memory frames structure:")
        debug_memory_frames_structure(memory_frames)

        # 2) 기본 z/x 처리
        # print("[FORWARD Debug] --- Start forward pass ---")
        # print("[FORWARD Debug] Template (z) input shape:", z.shape)
        # print("[FORWARD Debug] Search   (x) input shape:", x.shape)
        z_feat = self._z_feat(z, z_feat_mask)
        
        # print("[FWD Debug] z_feat contains NaN?", z_feat.isnan().any().item())
        x_feat = self._x_feat(x)

        # print("[FWD Debug] x_feat contains NaN?", x_feat.isnan().any().item())
        fusion_feat = self._fusion(z_feat, x_feat)

        B = z.shape[0]
        if use_memory_mask is None:
             use_memory_mask = torch.zeros(B, dtype=torch.bool, device=fusion_feat.device)
        if memory_frames is None:
            memory_frames = [[] for _ in range(B)]

        # print(f"[MemFlow] use_memory_mask = {use_memory_mask.tolist()}")

        fusion_list = []
        # print("[FORWARD Debug] Processing memory frames for batch size:", B)
        for b in range(B):
            # mode = "SOT" if use_memory_mask[b] else "DET"
            # print(f"[MemFlow] Sample {b}: use_memory_mask={use_memory_mask[b].item()}")
            # DET 샘플: 원본 fusion_feat만
            if not use_memory_mask[b] or len(memory_frames[b]) == 0:
                fusion_list.append(fusion_feat[b:b+1])
            else:
                # SOT 샘플: memory_attention 결과만 사용
                emb = self._mem_feat([memory_frames[b]])
                sample_mem = emb.reshape(1, -1, emb.size(-1))
                mem_out = self.memory_attention(fusion_feat[b:b+1], sample_mem)
                fusion_list.append(mem_out)
            # ────────────────────────────────────────────────────────
            
        
        fusion_feat = torch.cat(fusion_list, dim=0)
        # print("[FORWARD Debug] Fusion feature shape after all samples:", fusion_feat.shape)

        out = self.head(fusion_feat)


        # ── 디버그: head 출력 dict 키와, bbox 값 범위 찍기 ──
        # print("[HEAD Debug] out.keys():", list(out.keys()))
        # 실제 head가 어떤 키로 박스를 내놓는지 확인한 뒤, 예를 들어 'bbox_map' 이라고 가정하면:
        if 'bbox_map' in out:
            bmap = out['bbox_map']           # torch.Tensor, shape=(B, Nq, 4) 등
            print("[HEAD Debug] bbox_map min/max:", bmap.min().item(), bmap.max().item())
        elif 'pred_boxes' in out:
            pb = out['pred_boxes']
            print("[HEAD Debug] pred_boxes min/max:", pb.min().item(), pb.max().item())
        # ────────────────────────────────


        if kwargs.get('return_x_feat', True):
            out['x_feat'] = fusion_feat
        # print("[FORWARD Debug] Forward complete. Output keys:", list(out.keys()))
        return out
    # def state_dict(self, **kwargs):
    #     """
    #     학습 가능한 파라미터만 남기고, LoRA 메타정보(lora_alpha, use_rslora)를 저장합니다.
    #     """
    #     # 기본 state_dict 확보
    #     state = super().state_dict(**kwargs)
    #     prefix = kwargs.get('prefix', '')
    #     # frozen 파라미터들 제거
    #     for key in list(state.keys()):
    #         if not self.get_parameter(key[len(prefix):]).requires_grad:
    #             state.pop(key)
    #     # LoRA 메타데이터 추가
    #     state[prefix + 'lora_alpha'] = torch.as_tensor(self.lora_alpha)
    #     state[prefix + 'use_rslora'] = torch.as_tensor(self.use_rslora)
    #     return state

    # def load_state_dict(self, state_dict, **kwargs):
    #     """
    #     저장된 LoRA 메타정보를 로드한 후, 나머지는 기본 로딩합니다.
    #     """
    #     prefix = kwargs.get('prefix', '')
    #     # 메타정보가 존재하면 꺼내서 설정
    #     meta_alpha = prefix + 'lora_alpha'
    #     meta_rslora = prefix + 'use_rslora'
    #     if meta_alpha in state_dict:
    #         self.lora_alpha = state_dict[meta_alpha].item()
    #         self.use_rslora = state_dict[meta_rslora].item()
    #         # 메타키 제거
    #         del state_dict[meta_alpha]
    #         del state_dict[meta_rslora]
    #     # 나머지는 기본 로딩
    #     return super().load_state_dict(state_dict, **kwargs)


    def state_dict(self, **kwargs):
        """
        super().state_dict 로 모든 파라미터를 가져온 뒤,
        LoRA 메타(lora_alpha, use_rslora)만 추가합니다.
        """
        state = super().state_dict(**kwargs)
        prefix = kwargs.get('prefix', '')

        # lora_alpha가 기본(1.0)이 아니면 메타정보 추가
        state[prefix + 'lora_alpha']  = torch.as_tensor(self.lora_alpha)
        state[prefix + 'use_rslora'] = torch.as_tensor(self.use_rslora)

        return state

    def load_state_dict(self, state_dict, **kwargs):
        """
        저장된 LoRA 메타정보를 꺼내서 self에 복원한 뒤,
        나머지 키들은 super()에 넘겨서 통상 로딩합니다.
        """
        prefix      = kwargs.get('prefix', '')
        meta_alpha  = prefix + 'lora_alpha'
        meta_rslora = prefix + 'use_rslora'

        if meta_alpha in state_dict:
            # dict 순서를 보장하기 위해 OrderedDict로 감싸고
            state_dict = OrderedDict(**state_dict)

            # 메타정보 복원
            self.lora_alpha  = state_dict[meta_alpha].item()
            self.use_rslora  = state_dict[meta_rslora].item()

            # 메타키 삭제
            del state_dict[meta_alpha]
            del state_dict[meta_rslora]

        # strict 기본값(True)로 두면, 모든 키가 있어야 에러가 나므로
        # base 코드는 strict=True여도 잘 통과됩니다.
        return super().load_state_dict(state_dict, **kwargs)