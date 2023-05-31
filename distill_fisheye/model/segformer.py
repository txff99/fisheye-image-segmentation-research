import torch 
from .segformer_head import SegFormerHead
from .mix_transformer import MixVisionTransformer
from torch import nn

class Segformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MixVisionTransformer()
        self.decode_head = SegFormerHead()
        self.softmax = nn.Softmax(dim=1)
        # self.softmax_flag=softmax
    def forward(self,x):
        x=self.backbone(x)
        x=self.decode_head(x)
        # if self.softmax_flag==True:
        #     x=self.softmax(x)
        return x