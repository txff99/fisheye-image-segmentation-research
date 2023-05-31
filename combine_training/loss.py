from torch import nn
import torch
import torch.nn.functional as F
import sys

class MyLoss_correction(nn.Module):
    def __init__(self,weight=None,att_depth=3,out_channels=10,patch_size=16):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.mseloss=nn.MSELoss()

        self.att_depth=att_depth
        self.patch_size=patch_size
        self.out_channels=out_channels

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                    stride=(self.patch_size, self.patch_size))

    def forward(self, y_pr, y_gt, attentions):
        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(2 ** self.att_depth, 2 ** self.att_depth),
                             stride=(2 ** self.att_depth, 2 ** self.att_depth), groups=self.out_channels, bias=False)
        conv_feamap_size.weight = nn.Parameter(torch.ones((self.out_channels, 1, 2 ** self.att_depth, 2 ** self.att_depth)))
        conv_feamap_size.to(y_pr.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False
        
        y_gt = F.one_hot(y_gt,num_classes=10).permute(0,3,1,2)
        y_gt = torch.tensor(y_gt,dtype=torch.float)
        
        y_gt_conv=conv_feamap_size(y_gt)/(2 ** self.att_depth*2 ** self.att_depth)
        attentions_gt=[]

        for i in range(y_gt_conv.size()[1]):
            unfold_y_gt = self.unfold(y_gt[:, i:i + 1, :, :]).transpose(-1, -2)
            unfold_y_gt_conv = self.unfold(y_gt_conv[:, i:i + 1, :, :])
            att=torch.matmul(unfold_y_gt,unfold_y_gt_conv)/(self.patch_size*self.patch_size)
            att=torch.unsqueeze(att,dim=1)
            attentions_gt.append(att)

        attentions_gt=torch.cat(attentions_gt,dim=1)
        y_gt=torch.argmax(y_gt,dim=-3)

        loss_entropy=self.ce(y_pr,y_gt)
        #loss_mse=self.mseloss(attentions,attentions_gt)/torch.numel(attentions)
        loss_mse = self.mseloss(attentions, attentions_gt)
        
        loss=5*loss_entropy+loss_mse

        return loss

class loss_correction(object):
    def __init__(self,cfg=''):
        # super().__init__()
        if cfg=='woodscape':
            self.void_classes = [1,2,3,4,5,6,8,9,10,14,15,16]
        else:
            self.void_classes = [19,20]

        self.ce = nn.CrossEntropyLoss(ignore_index=21)

    def target_init(self,inputs,target,ignored_index=21):
        im = inputs.argmax(dim=1)
        for void_class in self.void_classes:
            im[ im == void_class ]=ignored_index
        # sys.exit()
        mask = im == ignored_index
        # print(mask.shape)
        # print(target.shape)
        target[mask]=21
        return target

    def mse_loss(input1, target, ignored_index=21, reduction='mean'):
        mask = target == ignored_index
        # print(mask)
        out = (input1[~mask]-target[~mask])**2
        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out

    def ce_loss(self,inputs, target, ignored_index=21):
        target=self.target_init(inputs=inputs,target=target, ignored_index=ignored_index)
        # print(torch.unique(inputs))
        # print(torch.unique(target))
        loss = self.ce(inputs,target)
        return loss


