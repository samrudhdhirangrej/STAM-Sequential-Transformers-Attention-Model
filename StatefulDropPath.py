'''
implementation borrowed from timm library
Satetful DropPath
'''

import timm
import torch
from torch import nn
from functools import partial

class StatefulDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(StatefulDropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.
        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        output = x.div(keep_prob) * self.random_tensor.view((x.shape[0],) + (1,) * (x.ndim - 1))
        return output

    def reset(self, B, xdtype, xdevice):
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(B, dtype=xdtype, device=xdevice)
        random_tensor.floor_()  # binarize
        self.random_tensor = random_tensor

def reset_StatefulDropPath(model, B, xdtype, xdevice):
    for block in model.blocks:
        for module in block.children():
            if isinstance(module, StatefulDropPath):
                module.reset(B, xdtype, xdevice)

def make_DropPath_stateful(m):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == timm.models.layers.DropPath:
            print('replaced: ', attr_str)
            setattr(m, attr_str, StatefulDropPath(target_attr.drop_prob))
    for n, ch in m.named_children():
        make_DropPath_stateful(ch)

