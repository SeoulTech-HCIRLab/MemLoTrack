# scale_and_translate.py
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from trackit.core.operator.numpy.bbox.scale_and_translate import bbox_scale_and_translate
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize_

def reverse_scale_and_translation_parameters(scale: np.ndarray, translation: np.ndarray):
    return 1./scale, -translation/scale

def scale_and_translate(
    img: torch.Tensor,
    output_size: np.ndarray,
    scale: np.ndarray,
    translation: np.ndarray,
    background_color: Optional[torch.Tensor]=None,
    mode: str='bilinear',
    align_corners: bool=False,
    output_img: Optional[torch.Tensor]=None,
    return_adjusted_params: bool=False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray, np.ndarray]]:
    # MemLoTrack BASE 흐름 그대로
    img_dtype = img.dtype; device = img.device
    bbox_dtype = scale.dtype if scale.dtype in (np.float32, np.float64) else np.float32
    assert img.ndim in (3,4)
    batch_mode = (img.ndim==4)
    if not batch_mode: img = img.unsqueeze(0)
    n,c,h,w = img.shape

    # output buffer
    if output_img is not None:
        if batch_mode:
            assert output_img.ndim==4
        else:
            assert output_img.ndim in (3,4)
            if output_img.ndim==4: assert output_img.shape[0]==1
            else: output_img = output_img.unsqueeze(0)
    else:
        output_img = torch.empty((n,c,output_size[1],output_size[0]), dtype=img_dtype, device=device)

    # background
    if background_color is not None:
        if background_color.ndim==1:
            output_img[:] = background_color.view(1,-1,1,1)
        elif background_color.ndim==2:
            bn,bc = background_color.shape; assert bn==n
            output_img[:] = background_color.view(bn,bc,1,1)
        else:
            raise RuntimeError("Incompatible background_color shape")
    else:
        output_img.zero_()

    # compute bounding boxes
    output_bbox = bbox_scale_and_translate(np.asarray((0,0,w,h),dtype=bbox_dtype), scale, translation)
    reverse_scale, reverse_translation = reverse_scale_and_translation_parameters(scale, translation)
    bbox_clip_to_image_boundary_(output_bbox, output_size)
    bbox_rasterize_(output_bbox)

    input_bbox = bbox_scale_and_translate(output_bbox, reverse_scale, reverse_translation)
    bbox_rasterize_(input_bbox)
    bbox_clip_to_image_boundary_(input_bbox, np.asarray((w,h),dtype=bbox_dtype))

    validity = bbox_is_valid(output_bbox)
    output_bbox = torch.from_numpy(output_bbox).to(torch.long)
    input_bbox  = torch.from_numpy(input_bbox).to(torch.long)

    assert output_bbox.ndim in (1,2)

    # unpack once per MemLoTrack BASE
    if output_bbox.ndim==2:
        # batch
        for i in range(n):
            if not validity[i]: continue
            y1,y2 = output_bbox[i,1], output_bbox[i,3]
            x1,x2 = output_bbox[i,0], output_bbox[i,2]
            yi1,yi2 = input_bbox[i,1], input_bbox[i,3]
            xi1,xi2 = input_bbox[i,0], input_bbox[i,2]
            output_img[i:i+1,:, y1:y2, x1:x2] = F.interpolate(
                img[i:i+1,:, yi1:yi2, xi1:xi2],
                size=(y2-y1, x2-x1),
                mode=mode, align_corners=align_corners
            )
    else:
        # single
        if validity:
            y1,y2 = output_bbox[1], output_bbox[3]
            x1,x2 = output_bbox[0], output_bbox[2]
            yi1,yi2 = input_bbox[1], input_bbox[3]
            xi1,xi2 = input_bbox[0], input_bbox[2]
            output_img[0:1,:, y1:y2, x1:x2] = F.interpolate(
                img[0:1,:, yi1:yi2, xi1:xi2],
                size=(y2-y1, x2-x1),
                mode=mode, align_corners=align_corners
            )

    if not batch_mode:
        output_img = output_img.squeeze(0)

    if return_adjusted_params:
        obf = output_bbox.float(); ibf = input_bbox.float()
        real_scale = (obf[...,2:]-obf[...,:2])/(ibf[...,2:]-ibf[...,:2])
        real_trans = obf[...,:2] - ibf[...,:2]*real_scale
        return output_img, real_scale.numpy(), real_trans.numpy()

    return output_img

def scale_and_translate_subpixel(
    image: torch.Tensor,
    size: Tuple[int,int],
    scale: Union[torch.Tensor,np.ndarray],
    translation: Union[torch.Tensor,np.ndarray],
    interpolation_mode: str='bilinear',
    align_corners: bool=False,
    padding_mode: str='zeros'
) -> torch.Tensor:
    if image.dim()==3:
        image = image.unsqueeze(0)
    if isinstance(scale, np.ndarray):
        scale = torch.from_numpy(scale).float().to(image.device)
    if isinstance(translation, np.ndarray):
        translation = torch.from_numpy(translation).float().to(image.device)

    bsz, ch, H, W = image.shape
    out_w, out_h = size
    scale = scale.view(-1,1,1,2)
    translation = translation.view(-1,1,1,2)

    x = torch.linspace(0, out_w/W,  out_w, dtype=scale.dtype, device=scale.device)
    y = torch.linspace(0, out_h/H,  out_h, dtype=scale.dtype, device=scale.device)
    xg, yg = torch.meshgrid(x,y, indexing='xy')
    grid = torch.stack((xg,yg),-1).unsqueeze(0).expand(bsz,-1,-1,-1)
    grid.div_(scale)
    translation.div_(scale)
    translation.select(-1,0).div_(W)
    translation.select(-1,1).div_(H)
    grid.sub_(translation).mul_(2).sub_(1)

    out = F.grid_sample(
        image, grid, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners
    )
    return out.squeeze(0)
