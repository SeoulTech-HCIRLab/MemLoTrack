from typing import Tuple, Mapping, Any
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.head.mlp import MlpAnchorFreeHead
from .modules.lora.merge import lora_merge_state_dict


class MemLoTrackBaseline_DINOv2(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int]):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        assert isinstance(vit, DinoVisionTransformer)
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=.02)

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)


    def forward(self, 
                z: torch.Tensor, 
                x: torch.Tensor, 
                z_feat_mask: torch.Tensor, 
                return_x_feat = True,
                ) -> dict :
        """
        Forward pass for MemLoTrack.

        Args:
          z: Template image (B, 3, Hz_in, Wz_in).
          x: Search region image (B, 3, Hx_in, Wx_in).
          z_feat_mask: A mask for z tokens? (B, Hz*Wz)
          return_x_feat: if True, also return the final search feature
            (fusion_feat) in the output dict as 'x_feat'.

        Returns:
          out_dict = {
            'score_map': Tensor shape (B, 1, H, W),
            'boxes': Tensor shape (B, H, W, 4)
            # if return_x_feat=True, then 'x_feat': Tensor shape (B, H*W, C)
          }
        """

        # print("[MemLoTrack Debug] We are in My Custom forward in `memlotrack.py` with ID=ABC123.")
        # return_x_feat = True
        
        # print("[MemLoTrack Debug] forward() called with return_x_feat =", return_x_feat)

        # 1) template feature
        z_feat = self._z_feat(z, z_feat_mask)
        # 2) search feature
        x_feat = self._x_feat(x)
        # 3) fuse
        fusion_feat = self._fusion(z_feat, x_feat)
            # fusion_feat shape: (B, (Wx*Hx), C)

        #MLP head => boxes, score_map
        out_dict = self.head(fusion_feat)
            # out_dict =  {'score_map' : (B,1,H,W), 'boxes' : (B,H,W,4)} 

        if return_x_feat:
            out_dict['x_feat'] = fusion_feat
            # print("[MemLoTrack Debug] out_dict keys after adding x_feat =", list(out_dict.keys()))
            # print("[MemLoTrack Debug] x_feat shape =", fusion_feat.shape)

        # print("[MemLoTrack Debug] out_dict keys = ", list(out_dict.keys()))
        return out_dict

        # return self.head(x_feat)

        
    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z = self.patch_embed(z)
        z_W, z_H = self.z_size
        # reshape pos_embed to match z size
        pos_reshaped = (
            self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)
                          [:, :z_H, :z_W, :]
                          .reshape(1, z_H * z_W, self.embed_dim)
        )
        z = z + pos_reshaped

        # token type embed
        # (z_feat_mask가 [B, z_H*z_W] 형태라면) 
        z = z + self.token_type_embed[z_feat_mask.flatten(1)]
        return z


    def _x_feat(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = x + self.token_type_embed[2].view(1, 1, self.embed_dim)
        return x

    def _fusion(self, z_feat: torch.Tensor, x_feat: torch.Tensor):
        fusion_feat = torch.cat((z_feat, x_feat), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)
        return fusion_feat[:, z_feat.shape[1]:, :]

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        state_dict = lora_merge_state_dict(self, state_dict)
        return super().load_state_dict(state_dict, **kwargs)
