""" Image to Patch Embedding using Conv2d
A convolution based approach to patchifying a 2D image w/ embedding projection.
Based on the impl in https://github.com/rwightman/pytorch-image-models/
"""

from torch import nn as nn

from timm.models.layers.helpers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.reshape(B*N, C, H, W)
        x = self.proj(x)   # B*N, C, 1, 1
        assert x.size(2)==1 and x.size(3)==1, f"Glimpse size is incorrect."
        if self.flatten:
            x = x[...,0,0]
        x = x.reshape(B,N,self.embed_dim)
        x = self.norm(x)
        return x
