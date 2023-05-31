import torch 
from .segformer_head import SegFormerHead
from .segformer_head_decoder import SegFormerHeadwithdecoder
from .mix_transformer import MixVisionTransformer
from torch import nn

class combined_head(nn.Module):
    def __init__(self,embedding_dim=256,num_classes=21):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
    def forward(self,x):
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x

class Segformer(nn.Module):
    def __init__(self,combine_head=False):
        super().__init__()
        self.backbone = MixVisionTransformer()
        self.decode_head = SegFormerHead()
        # self.head_flag = combine_head
        # # self.combined_head=combined_head()
    def forward(self,x):
        x=self.backbone(x)
        x=self.decode_head(x)
        # if self.head_flag==True:
        #     x=self.combined_head(x)
        return x

class vanilla_segformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MixVisionTransformer()
        self.decode_head = SegFormerHeadwithdecoder()
        # self.head_flag = combine_head
        # self.combined_head=combined_head()
    def forward(self,x):
        x=self.backbone(x)
        x=self.decode_head(x)
        # if self.head_flag==True:
        #     x=self.combined_head(x)
        return x
