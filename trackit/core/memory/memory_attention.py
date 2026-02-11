"""
Attention map vis 이전 / 이후 
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):

        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first

        # Multi-Head Cross-Attention
        #   query = curr_feat, key=value=memory_feat
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )

        # LayerNorms for Pre-Norm approach
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward block
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # self.ffn = nn.Identity()

        # 새로운: gradient norm 기록을 위한 dictionary
        self.grad_norms = {}  # 각 파라미터 이름별로 gradient norm 기록 리스트

        # Hook 등록: cross_attn의 모든 파라미터에 대해 hook을 등록합니다.
        self._register_grad_hooks()

    def _register_grad_hooks(self):
        # 각 파라미터의 gradient norm을 기록하는 hook을 등록합니다.
        for name, param in self.cross_attn.named_parameters():
            def hook_factory(n):
                # hook 함수는 n (파라미터 이름)을 캡처합니다.
                def hook(grad):
                    norm = grad.norm().item()
                    if n not in self.grad_norms:
                        self.grad_norms[n] = []
                    self.grad_norms[n].append(norm)
                return hook
            param.register_hook(hook_factory(name))

    def forward(
        self,
        curr_feat: torch.Tensor,
        memory_feat: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # [MemoryAttentionLayer Debug] Forward called.
        # print("[MemoryAttentionLayer Debug] Forward called.")
        # print(f"[MemoryAttentionLayer Debug] Input curr_feat shape: {curr_feat.shape}")
        # print(f"[MemoryAttentionLayer Debug] Input memory_feat shape: {memory_feat.shape}")

        # 1) Cross-Attention
        # Pre-Norm
        q = self.norm1(curr_feat)
        k = self.norm1(memory_feat)  

        # print(f"[MemoryAttentionLayer Debug] After norm1: q.shape = {q.shape}, k.shape = {k.shape}")

        # CrossAttn: Query=curr_feat, Key=memory_feat, Value=memory_feat
        # shape of out_x: (B, Nq, d_model)

        import torch.utils.checkpoint as cp

        # ── 디버그: cross-attention 출력을 out_x, attn_weights로 받기 ──
        # (checkpoint 대신 직접 호출해도 좋습니다)
        out_x, attn_weights = self.cross_attn(q, k, k, need_weights=True, attn_mask=attn_mask)
        # 또한 checkpoint된 버전이 필요하면 아래처럼:
        # out_x, _ = cp.checkpoint(self.cross_attn, q, k, k, attn_mask)+
        # ── MEMDBG: out_x 통계 ─────────────────────────────────────
        # print(f"[MEMDBG] out_x min/max/mean/std = "
        #       f"{out_x.min().item():.4f}/"
        #       f"{out_x.max().item():.4f}/"
        #       f"{out_x.mean().item():.4f}/"
        #       f"{out_x.std().item():.4f}")
        if torch.isnan(out_x).any():  print("[MEMDBG] out_x has NaN")
        if torch.isinf(out_x).any(): print("[MEMDBG] out_x has Inf")

        # ── MEMDBG: attn_weights 통계 ─────────────────────────────
        # print(f"[MEMDBG] attn_weights min/max/mean/std = "
        #       f"{attn_weights.min().item():.4f}/"
        #       f"{attn_weights.max().item():.4f}/"
        #       f"{attn_weights.mean().item():.4f}/"
        #       f"{attn_weights.std().item():.4f}")
        if torch.isnan(attn_weights).any():  print("[MEMDBG] attn_weights has NaN")
        if torch.isinf(attn_weights).any(): print("[MEMDBG] attn_weights has Inf")
        
        # Residual
        # alpha = 0.1
        alpha = 1.0
        x = curr_feat + alpha * out_x        
        x_norm = self.norm2(x)

        
        x_ffn = self.ffn(x_norm)
        # 5) Residual connection after FFN
        x = x + x_ffn

        return x

    # 새로 추가: gradient 기록 데이터를 파일로 저장하는 메서드
    def save_gradients(self, file_path: str):
        import csv
        with open(file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['parameter', 'grad_norms'])
            for param_name, norms in self.grad_norms.items():
                writer.writerow([param_name, norms])
        print(f"[MemoryAttentionLayer] Gradient norms saved to {file_path}")

class MemoryAttention(nn.Module):
    """
    여러 MemoryAttentionLayer를 쌓아 순차적으로 적용하는 모듈.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int = 4,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
       
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryAttentionLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=batch_first
            ) for _ in range(num_layers)
        ])
        # print(f"[MemoryAttention Debug] Initialized with {num_layers} layers, d_model={d_model}, nhead={nhead}")

    def forward(
        self,
        curr_feat: torch.Tensor,
        memory_feat: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
                
        # print(f"[MemoryAttention Debug] Called. curr_feat.shape={curr_feat.shape}, memory_feat.shape={memory_feat.shape}")

        if memory_feat is None or memory_feat.size(1) == 0:
            return curr_feat

        for i, layer in enumerate(self.layers):
            # print(f"[MemoryAttention Debug] Before layer {i}: curr_feat.shape={curr_feat.shape}")
            curr_feat = layer(curr_feat, memory_feat)
            # print(f"[MemoryAttention Debug] After  layer {i}: curr_feat.shape={curr_feat.shape}")

        # print("[MemoryAttention Debug] after all layers contains NaN?", curr_feat.isnan().any().item())
        return curr_feat



