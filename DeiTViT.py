'''
adapted (and modified) from: https://github.com/facebookresearch/deit
'''


import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_
from patch_embed import PatchEmbed

class DeiTVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.img_size = kwargs['img_size']
        if self.img_size==224:
            self.num_patch_per_dim_glimpse = 2
        elif self.img_size==240:
            self.num_patch_per_dim_glimpse = 3
        elif self.img_size==256:
            self.num_patch_per_dim_glimpse = 4
        else:
            self.num_patch_per_dim_glimpse = None
        self.patch_size = kwargs['patch_size'] #16
        self.in_chans = 3 #kwargs['in_chans'] #3
        self.num_patch_per_dim = self.img_size//self.patch_size
        self.num_glimpse_per_dim = self.num_patch_per_dim // self.num_patch_per_dim_glimpse

        num_patches = self.patch_embed.num_patches
        self.patch_embed=PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2+num_patches, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)

    def forward(self, x):
        B = x.size(0)

        x = x.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size) # (B, 3, 224, 224) -> (B,3,14,14,16,16)
        x = x.unfold(2,self.num_patch_per_dim_glimpse,self.num_patch_per_dim_glimpse).unfold(3,self.num_patch_per_dim_glimpse,self.num_patch_per_dim_glimpse)     # (B,3,14,14,16,16) -> (B,3,7,7,16,16,2,2)
        x = x.permute(0,2,3,6,7,1,4,5)        # (B,3,7,7,16,16,2,2) -> (B,7,7,2,2,3,16,16) = (B, num_glimpses_per_dim, num_glimpses_per_dim, num_patches_per_glimpse, num_patches_per_glimpse, 3, 16, 16)
        x = x.flatten(1,2)

        pos = self.pos_embed[:,2:,:].reshape(1,self.num_patch_per_dim,self.num_patch_per_dim,self.embed_dim)
        pos = pos.unfold(1,self.num_patch_per_dim_glimpse,self.num_patch_per_dim_glimpse).unfold(2,self.num_patch_per_dim_glimpse,self.num_patch_per_dim_glimpse) # (1,14,14,C) -> (1,7,7,C,2,2)
        pos = pos.permute(0,1,2,4,5,3).repeat(B,1,1,1,1,1) # (1,7,7,C,2,2) -> (1,7,7,2,2,C) -> (B,7,7,2,2,C)
        pos = pos.flatten(1,2)

        x_pos = self.patch_embed(x.flatten(1,3)) + pos.flatten(1,3)
        cls_pos = self.cls_token.expand(B, -1, -1) + self.pos_embed[:,:1,:]
        dist_pos = self.dist_token.expand(B, -1, -1) + self.pos_embed[:,1:2,:]
        x = torch.cat((cls_pos, dist_pos, x_pos), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        gt_label = self.head(x[:,0])
        dist_label = self.head_dist(x[:,1])

        return gt_label, dist_label
