import numpy as np
import torch
from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation,SegformerConfig,SegformerModel,SegformerDecodeHead
from transformers import SegformerFeatureExtractor
from PIL import Image
# import requests
from torch import nn
import cv2
import time
import sys
from model import model

def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap

def colormap_woodscape(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([ 0,0,0])
    cmap[1,:] = np.array([255,
            0,
            255])
    cmap[2,:] = np.array([255,
            0,
            0])
    cmap[3,:] = np.array([0,
            255,
            0])
    cmap[4,:] = np.array([ 0,
            0,
            255])
    cmap[5,:] = np.array([  255,
            255,
            255])
    cmap[6,:] = np.array([  255,
            255,
            0])

    cmap[7,:] = np.array([0,
            255,
            255])
    cmap[8,:] = np.array([ 128,
            128,
            255])
    cmap[9,:] = np.array([  0,
            128,
            128])
    # cmap[10,:] = np.array([ 70,130,180])

    # cmap[11,:] = np.array([ 220, 20, 60])
    # cmap[12,:] = np.array([ 255,  0,  0])
    # cmap[13,:] = np.array([ 0,  0,142])
    # cmap[14,:] = np.array([  0,  0, 70])
    # cmap[15,:] = np.array([  0, 60,100])

    # cmap[16,:] = np.array([  0, 80,100])
    # cmap[17,:] = np.array([  0,  0,230])
    # cmap[18,:] = np.array([ 119, 11, 32])
    # cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap

class Colorize:

    def __init__(self, n=20):
        #self.cmap = colormap(256)
        self.cmap = colormap_woodscape(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

# image_ = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
# image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024",num_labels=10,ignore_mismatched_sizes=True)
# model2 = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
# model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024",output_hidden_states=True)
# pretrained_dict = torch.load('/content/drive/MyDrive/Colab_Notebooks/fisheye_aug/classifier_trained/new/fisheye_init.ckpt',map_location=torch.device('cpu'))['state_dict']
# model = model()
# pretrained_dict = torch.load("/mnt/ssd/home/tianxiaofeng/AFMA/lightning_logs/version_8/checkpoints/epoch=467-step=96408.ckpt",map_location=torch.device('cpu'))['state_dict']
pretrained_dict = torch.load("/mnt/ssd/home/tianxiaofeng/segformer_train/lightning_logs/version_2/checkpoints/epoch=495-step=93976.ckpt",map_location=torch.device('cpu'))['state_dict']
# print(dir(pretrained_dict))
# print(pretrained_dict.keys())
pretrained_dict={keys.replace('model.','',1): value for keys,value in pretrained_dict.items()}
# pretrained_dict={keys.replace('segformer.','',1): value for keys,value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)

# print(dir(model))
# print(model.tie_weights)
# print(model.state_dict)
# for i in model.state_dict:
#   print(i)
# print(dir(SegformerForSemanticSegmentation))
# model = SegformerForSemanticSegmentation(num_class=9)
feature_extractor=SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('/mnt/hdd/dataset/woodscape/rgb_images/00025_FV.png')
masks = Image.open('/mnt/hdd/dataset/woodscape/semantic_annotations/gtLabels/00000_FV.png')
# print(image.size)
encoded_inputs = feature_extractor(image, masks,return_tensors="pt",size = {"height" : 1024, "width": 1024})
for k,v in encoded_inputs.items():
  encoded_inputs[k].squeeze_()
# start = time.time()
# inputs = feature_extractor(images=image, return_tensors="pt",size = {"height" : 512, "width": 1024})
# inputs = image_processor(images=image, return_tensors="pt",size = {"height" : 1024, "width": 1024})

# print(inputs.items)
# print(model.state)
images, masks = encoded_inputs['pixel_values'], encoded_inputs['labels']
images=images.unsqueeze(dim=0)
masks=masks.unsqueeze(dim=0)
# output,attention = model(images)
# output = model(images).logits
model.eval()
with torch.no_grad():
        output = model(images)[0]
# print(output)
# print(images.shape)
# outputs = model(images, masks)
# print(model.state_dict())
# # print(model)
# print(outputs.hidden_states[-1].shape)
# logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
# list(logits.shape)
# loss, logits = outputs[0], outputs[1]  
# print(logits.shape)

output = nn.functional.interpolate(
    output, 
    size=(1024,1024), 
    mode="bilinear", 
    align_corners=False
)
# print(upsampled_logits.shape)
# print(model)
label = output[0].max(0)[1].byte().cpu().data
label_color = Colorize()(label.unsqueeze(0))
img=label_color.numpy()
img=img.transpose(1,2,0)
cv2.imwrite('./demo/00055_fv1_baseline.png',img)