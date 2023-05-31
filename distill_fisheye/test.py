import cv2 
from torch import nn
from norm2fisheye import fisheye
import numpy as np
import torch
from transformers import SegformerModel
from transformers import SegformerFeatureExtractor
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from model.segformer import Segformer
from visualize import Colorize

def convert2fisheye(imgs,grid,label=False):
    if label==False:
        outp = F.grid_sample(imgs.permute(0,3,1,2),grid=grid,mode='bilinear')
    else:
        print(imgs.unsqueeze(dim=1).shape)
        outp = F.grid_sample(imgs.unsqueeze(dim=1).float(),grid=grid,mode='nearest')
    return outp


def kl_divergence(inputs, target,ignore_index=1):
    #p is the label
    print(inputs)
    print(target)
    mask = target == ignore_index

    return (target[~mask]*(target[~mask].log()-inputs[~mask].log())).mean()
klloss = nn.KLDivLoss()
#     summ = 0
#     def kl_divergence(p, q):
#  return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
    # for i in len(p.shape[dim]):
    #     summ += 
    #     return sum(p[:,i,:,:] * log2(p[:,i,:,:]/q[:,i,:,:]))
softmax = nn.Softmax(dim=1)
x = torch.randn(1,2,10,10)
y = torch.randn(1,2,10,10)
x = softmax(x)
log_x = x.log()
y = softmax(y)
p = x[0,:,0,0]
q = y[0,:,0,0]
print(x[0,:,0,0])
print(y[0,:,0,0])
loss1 = kl_divergence(x,y)
loss2 = klloss(log_x,y)
print(loss2)
print(loss1)
# print(q*torch.log(q/p))
sys.exit()
# img = cv2.imread("/mnt/hdd/dataset/woodscape/rgb_images/00000_FV.png")
target = cv2.imread("/mnt/hdd/dataset/cityscapes/extracted/gtFine/train/weimar/weimar_000141_000019_gtFine_labelIds.png")
# target = cv2.imread("/mnt/hdd/dataset/woodscape/semantic_annotations/gtLabels/00000_FV.png")
ori_img = cv2.imread("/mnt/hdd/dataset/cityscapes/extracted/leftImg8bit/train/weimar/weimar_000141_000019_leftImg8bit.png")
# model = SegformerModel.from_pretrained(
#             "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
#             output_hidden_states=True
#         )
model = Segformer()
for param in model.decode_head.linear_pred.named_parameters():
    print(param)
    param[1].requires_grad=False
# model.decode_head.linear_pred.requires_grad=False
class_head = Segformer().decode_head.linear_pred
print(list(model.named_parameters()))
sys.exit()
pretrained_dict = torch.load("./model/segformer.b0.1024x1024.city.160k.pth")['state_dict']
# pretrained_dict = {k.replace('backbone','encoder'):v for k,v in pretrained_dict.items()}
# print(pretrained_dict.keys())
class_head.weight = nn.Parameter(pretrained_dict['decode_head.linear_pred.weight'])
class_head.bias = nn.Parameter(pretrained_dict['decode_head.linear_pred.bias'])

# sys.exit()
model.load_state_dict(pretrained_dict,strict=False)
# class_head.load_state_dict(pretrained_dict,strict=False)
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
# output,hidden = model(inputs1['pixel_values'])
def mse_loss(inputs, target, ignored_index=255, reduction='mean'):
    mask = target == ignored_index
    out = (inputs[~mask]-target[~mask])**2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out
# sys.exit()

# print(torch.unique(inputs['labels']))
ori_img = cv2.resize(ori_img,dsize=(512,512))
target = cv2.resize(target,dsize=(512,512))

target = target[:,:,0]
fi = fisheye()
fi.f0 = np.random.randint(200,600)
fi.pitch = np.random.uniform(-0.8,0)
fi.trans_x = np.random.uniform(0.5,1.5)
map_x,map_y = fi.norm2fisheye(ori_img)
# B,w,h,C = ori_inputs['pixel_values'].shape
w,h = 512,512        
# ori_img = torch.tensor(ori_img).unsqueeze(dim=0).float()#.permute(0,3,1,2)
# f_inputs = feature_extractor(fimages,return_tensors='pt')
ori_inputs = feature_extractor(ori_img,target,return_tensors='pt')
# print(ori_inputs['pixel_values'].shape)
# target = torch.tensor(target).unsqueeze(dim=0).float()
map_x = [torch.tensor(i).float() for i in map_x]
map_y = [torch.tensor(i).float() for i in map_y]
map_x = torch.stack(map_x,dim=0).unsqueeze(dim=0)
map_y = torch.stack(map_y,dim=0).unsqueeze(dim=0)
grid = torch.stack((map_x/((w)/2)-1,map_y/((h)/2)-1),dim=3)#.unsqueeze(0)

# print(ori_inputs['pixel_values'].shape)
# print(f"hahaha{ori_inputs['labels'].shape}")
# # print(ori_inputs['pixel_values'].shape)
# print(torch.unique(ori_inputs['labels']))
# fimages = convert2fisheye(imgs=ori_inputs['pixel_values'].permute(0,2,3,1),grid=grid)
# masks = convert2fisheye(imgs=ori_inputs['labels']-255,grid=grid,label=True)
# # masks = masks+255
# print(f"mask:{masks.shape}")
# print(torch.unique(fimages))
# fimages = torch.nan_to_num(fimages,nan=0)
# masks = torch.nan_to_num(masks+255,nan=255)
# print(torch.unique(masks))
# label_color = Colorize()(masks.squeeze(0))
# img=label_color.numpy()
# img=img.transpose(1,2,0)
# cv2.imwrite('./demo/mask.png',img)

# fimages = fimages.permute(0,2,3,1)
# print(ori_img.shape)
# print(fimages.shape)
# masks = masks[:,0,:,:]+255
# fimages = np.array(fimages.squeeze(dim=0).permute(1,2,0))
# print(fimages.shape)
# cv2.imwrite('./demo/fimg.png',fimages)
# sys.exit()


# ori_img = torch.tensor(ori_img).unsqueeze(dim=0).permute(0,3,1,2).float()
# print(ori_img.shape)
# output = F.normalize(ori_img)
# print(output[0,0,:,0])
# sys.exit()
# for k,v in ori_inputs.items():
#     ori_inputs[k].squeeze_()
# for k,v in finputs.items():
#     finputs[k].squeeze_()   
# img = torch.tensor(img).unsqueeze(0).to(torch.float32)
# fimg = fi.norm2fisheye(imgs=img).squeeze(0).permute(1,2,0)
# fimg = np.array(fimg,dtype=np.float32)
# print(fimg.shape)
# sys.exit()
# target = fi.norm2fisheye(target,label=True)

# pretrained_dict = torch.load("/mnt/ssd/home/tianxiaofeng/fisheye_aug/fisheye_pretrain/lightning_logs/version_0/checkpoints/epoch=19-step=1500.ckpt")['state_dict']
# pretrained_dict = {key.replace('model.',''): value for key,value in pretrained_dict.items()}
# pretrained_dict = {key.replace('segformer.',''): value for key,value in pretrained_dict.items()}
# model.load_state_dict(pretrained_dict,strict=False)

# del pretrained_dict['decode_head.classifier.bias']
# del pretrained_dict['decode_head.classifier.weight']
# print(inputs['pixel_values'].shape)
# print(inputs['pixel_values'].shape)
model.eval()
with torch.no_grad():
    ori_output,ori_hidden = model(ori_inputs['pixel_values'])
    # foutput,f_hidden = model(fimages)#f_inputs['pixel_values'])
# loss = nn.CrossEntropyLoss()
# masks = masks.squeeze(dim=1)
# print(masks.shape)
# print(foutput.shape)
# print(loss(foutput,masks))
# ori_hidden = ori_output.hidden_states[0]#.squeeze(dim=0)#.detach().numpy()
# f_hidden = foutput.hidden_states[0].squeeze(dim=0)
# print(ori_output.shape)
# ori_output = ori_output.argmax(dim=1)
# print(fi.map_x.shape)
# map_x = [torch.tensor(fi.map_x)]
# map_y = [torch.tensor(fi.map_y)]
# # print(time.time()-start)
# # sys.exit()
# map_x = torch.stack(map_x,dim=0)
# # print(map_x.shape)
# map_y = torch.stack(map_y,dim=0)
# # print(map_x)
# grid = torch.stack((map_x/(512/2)-1,map_y/(512/2)-1),dim=3)
# print(grid.shape)

# if label==False:
t_hidden = nn.functional.interpolate(
    ori_hidden, 
    size=(512,512), 
    mode="bilinear", 
    align_corners=False
)
t_hidden = F.grid_sample(t_hidden,grid=grid,mode='bilinear')#,padding_mode="zeros")
# print(grid)
# print(t_hidden)
t_hidden = nn.functional.interpolate(
    t_hidden, 
    size=ori_hidden.shape[-2:], 
    mode="bilinear", 
    align_corners=False
)
print(t_hidden.shape)
output = class_head(t_hidden)
print(output.shape)
output = nn.functional.interpolate(
    output, 
    size=(512,512),
    mode="bilinear", 
    align_corners=False
)
label_color = Colorize()(output.argmax(dim=1))
img=label_color.numpy()
img=img.transpose(1,2,0)
cv2.imwrite('./demo/saveme.png',img)
sys.exit()
# print(t_hidden.shape)
# t_hidden = t_hidden.squeeze(dim=0)
# t_hidden = torch.nan_to_num(t_hidden,nan=0)
# sys.exit()
fig,ax = plt.subplots(4,4,figsize=(10,10))       
for i in range(4):
    for j in range(4):
        # print(t_hidden[i*4+j,:,:].shape)
        ax[i,j].imshow(t_hidden[i*4+j,:,:].detach().numpy(),cmap='gray')
        ax[i,j].axis('off')
plt.savefig('./demo/fconvert.png')   
ori_hidden = ori_hidden.squeeze(dim=0)
for i in range(4):
    for j in range(4):
        # print(t_hidden[i*4+j,:,:].shape)
        ax[i,j].imshow(ori_hidden[i*4+j,:,:].detach().numpy(),cmap='gray')
        ax[i,j].axis('off')
plt.savefig('./demo/ori_f.png') 

f_hidden = f_hidden.squeeze(dim=0)
print(f_hidden.shape)
for i in range(4):
    for j in range(4):
        # print(t_hidden[i*4+j,:,:].shape)
        ax[i,j].imshow(f_hidden[i*4+j,:,:].detach().numpy(),cmap='gray')
        ax[i,j].axis('off')
plt.savefig('./demo/fish_f.png') 
# print(f_hidden.shape)
# print(t_hidden)
# print(t_hidden.shape)
loss1 = mse_loss(f_hidden,t_hidden)
print(f"upsampledloss:{loss1}")
# cv2.imwrite('fimage.png',fimage)
# cv2.imwrite('target.png',target)
# "/mnt/ssd/home/tianxiaofeng/fisheye_aug/fisheye_pretrain/lightning_logs/version_0/checkpoints/epoch=19-step=1500.ckpt"

# fa = nn.functional.interpolate(
#     ori_hidden.unsqueeze(dim=0), 
#     size=ori_img.shape[:-1], 
#     mode="bilinear", 
#     align_corners=False
# )
sys.exit()
fa = ori_hidden.permute(1,2,0).detach().numpy()
# print(fa.shape)
# fa = fa.squeeze(dim=0).permute(1,2,0).detach().numpy()
fa = fi.norm2fisheye(fa)
for i in range(4):
    for j in range(4):
        # print(t_hidden[i*4+j,:,:].shape)
        ax[i,j].imshow(fa[:,:,i*4+j],cmap='gray')
        ax[i,j].axis('off')
plt.savefig('./demo/new_fconvert.png') 
# f_hidden = f_hidden.detach().numpy()
# print(f_hidden.shape)

fa = torch.tensor(fa.transpose(2,0,1))
# fa = nn.functional.interpolate(
#     fa.unsqueeze(dim=0), 
#     size=ori_hidden.shape[-2:], 
#     mode="bilinear", 
#     align_corners=False
# )
# print(fa)
# print(fa.shape)
loss2 = mse_loss(f_hidden,fa)#.squeeze(dim=0))
print(f"directly_convert_loss:{loss2}")