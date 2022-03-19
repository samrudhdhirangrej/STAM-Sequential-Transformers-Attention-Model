'''
based on: https://github.com/facebookresearch/deit
'''

import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model

from patch_embed import PatchEmbed

from paths import PRETRAINED
from finetune_setup import finetune_setup

import utils

from STAMVisionTransformer import STAMVisionTransformer
from DeiTViT import DeiTVisionTransformer

# teacher student pair
__all__ = [
    'deit_small_patch16_224_fMoW',    'STAM_deit_small_patch16_224_fMoW',
    'DeiT_distill_small_patch16_224', 'STAM_deit_small_patch16_224',
    ]

@register_model
def deit_small_patch16_224_fMoW(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(PRETRAINED+'/fmow_deit_small_patch16_224.pth', map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        print('done')
    return model

@register_model
def STAM_deit_small_patch16_224_fMoW(pretrained=False, **kwargs):
    model = STAMVisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = finetune_setup(model, PRETRAINED+'/deit_small_distilled_patch16_224-649709d9.pth')
    return model

@register_model
def DeiT_distill_small_patch16_224(pretrained=False, **kwargs):
    model = DeiTVisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(PRETRAINED+'/deit_small_distilled_patch16_224-649709d9.pth', map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def STAM_deit_small_patch16_224(pretrained=False, **kwargs):
    model = STAMVisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(PRETRAINED+'/deit_small_distilled_patch16_224-649709d9.pth', map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model


