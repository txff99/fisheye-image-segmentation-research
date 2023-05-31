# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
# from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
# from collections import OrderedDict

# from mmseg.ops import resize
# from ..builder import HEADS
# from .decode_head import BaseDecodeHead
# from mmseg.models.utils import *
# import attr

# from IPython import embed

class combined_head(nn.Module):
    def __init__(self,embedding_dim=256,num_classes=21):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
    def forward(self,x):
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x
        

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ConvModule(nn.Module):
    def __init__(self, embedding_dim=2048):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(self.bn(x))
        return x


# @HEADS.register_module()
class SegFormerHeadwithdecoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self,in_channels=[32, 64, 160, 256],embed_dim=256,num_class=21):
        super().__init__()
        # assert len(feature_strides) == len(self.in_channels)
        # assert min(feature_strides) == feature_strides[0]
        # self.feature_strides = feature_strides
        # self.cfg=cfg
        self.num_classes = num_class
        self.in_channels = in_channels
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # decoder_params = kwargs['decoder_params']
        embedding_dim = embed_dim#decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse =ConvModule(
            embedding_dim=embedding_dim
        )
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
        # self.batch_norm = nn.BatchNorm2d(embedding_dim)
        # self.activation = nn.ReLU()

        # self.dropout = nn.Dropout(0.1)
        # self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # self.linear_pred2 = nn.Conv2d(embedding_dim, 10, kernel_size=1)

        
    def forward(self, x):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = nn.functional.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = nn.functional.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = nn.functional.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        # x = self.combined_head(_c)
        x = self.dropout(_c)
        x = self.linear_pred(x)
        # if self.cfg=='woodscape':
        #     x = self.linear_pred2(x)
        # else:

        return x,_c
