import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation
import sys

class encoder(nn.Module):
    def __init__(self, patch_size=16, out_channels=[32, 64, 128, 256], attention_on_depth=1):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            num_labels=10,
            output_hidden_states=True,
            ignore_mismatched_sizes=True,
        )
        self._in_channels  = 3
        self._out_channels = out_channels
        self._attention_on_depth = attention_on_depth
        self.patch_size = patch_size
        self.conv_img=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7,7),padding=3),
            nn.Conv2d(64, 1, kernel_size=(3,3), padding=1)
        )
        self.conv_feamap=nn.Sequential(
            nn.Conv2d(self._out_channels[self._attention_on_depth], 10, kernel_size=(1, 1), stride=1)
        )
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        self.resolution_trans=nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2*self.patch_size * self.patch_size, bias=False),
            nn.Linear(2*self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )

    def forward(self,x):
        attentions=[]
        
        # images, masks = batch['pixel_values'], batch['labels']
        ini_img=self.conv_img(x)
        outputs = self.model(pixel_values=x)
        
        x = outputs['hidden_states'][1]
        logits = outputs.logits
        # print(logits.shape)
        # sys.exit()
        # upsampled_logits = nn.functional.interpolate(
        #     logits, 
        #     size=masks.shape[-2:], 
        #     mode="bilinear", 
        #     align_corners=False
        # )
        # predicted = upsampled_logits.argmax(dim=1)
        
        if self._attention_on_depth == 1:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)
            
            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att=torch.unsqueeze(att,1)

                attentions.append(att)

            attentions = torch.cat((attentions), dim=1)

        return attentions, logits

class decoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=10, kernel_size=3, patch_size=16, activation=None, upsampling=1, att_depth=1):
        super().__init__()
        self.patch_size=patch_size
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        self.out_channels=out_channels
        self.activation = nn.Softmax(dim=1)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.activation = nn.Softmax(dim=1)
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        # self.activation = Activation(activation)
        if att_depth < 4:
            self.att_depth = att_depth+2
        else:
            self.att_depth = 3

    def forward(self, attentions,x):
        
        conv_feamap_size = nn.Conv2d(self.out_channels,self.out_channels, kernel_size=(2**self.att_depth, 2**self.att_depth),stride=(2**self.att_depth, 2**self.att_depth),groups=self.out_channels,bias=False)
        conv_feamap_size.weight=nn.Parameter(torch.ones((self.out_channels, 1, 2**self.att_depth, 2**self.att_depth)))
        conv_feamap_size.to(x.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False

        # x = self.conv_x(x)
        # x = self.upsampling(x)
        x = nn.functional.interpolate(
            x, 
            size=(x.size()[-2]*4,x.size()[-1]*4), 
            mode="bilinear", 
            align_corners=False
        )

        fold_layer = torch.nn.Fold(output_size=(x.size()[-2], x.size()[-1]), kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        correction=[]
        # x = F.one_hot(x)
        # x_argmax = x.argmax(dim=1)
        x_softmax = self.activation(x)
        # x_argmax=torch.argmax(x, dim=1)
        # pr_temp = torch.zeros(x.size()).to(x.device)
        # src = torch.ones(x.size()).to(x.device)
        # x_softmax = pr_temp.scatter(dim=1, index=x_argmax.unsqueeze(1), src=src)
        argx_feamap = conv_feamap_size(x_softmax) / (2 ** self.att_depth * 2 ** self.att_depth)
        for i in range(x.size()[1]):
            non_zeros = torch.unsqueeze(torch.count_nonzero(attentions[:, i:i + 1, :, :], dim=-1) + 0.00001,dim=-1)
            att = torch.matmul(attentions[:,i:i + 1, :, :]/non_zeros, torch.unsqueeze(self.unfold(argx_feamap[:, i:i + 1, :, :]), dim=1).transpose(-1, -2))
            att=torch.squeeze(att, dim=1)
            att = fold_layer(att.transpose(-1,-2))

            correction.append(att)

        correction=torch.cat(correction, dim=1)

        x = correction * x + x
        # x = self.activation(x)

        return x, attentions

class model(nn.Module):
    def __init__(self,patch_size=8):
        super().__init__()
        self.encoder = encoder(patch_size=patch_size)
        self.decoder = decoder(patch_size=patch_size)
    
    def forward(self,x):
        attentions,logits = self.encoder(x)
        output, attentions= self.decoder(attentions,logits)

        return output, attentions