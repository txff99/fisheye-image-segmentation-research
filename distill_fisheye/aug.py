from torchvision.datasets import Cityscapes
import numpy as np
from norm2fisheye import fisheye
import cv2
import time
import random
from PIL import Image
import evaluate
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
from torch.utils.data import DataLoader
from test import Colorize

def encode_segmap(mask):
    ignore_index=255
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
                'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
                'train', 'motorcycle', 'bicycle']
    #why i choose 20 classes
    #https://stackoverflow.com/a/64242989

    class_map = dict(zip(valid_classes, range(len(valid_classes))))
    #remove unwanted classes and recitify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

class mydataset(Cityscapes):
    def __init__(self,root,split,mode,target_type,test_mode=False):
        super().__init__(root,split,mode,target_type)
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.test_mode = test_mode
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        # class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
        #             'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
        #             'train', 'motorcycle', 'bicycle']
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

    def encode_segmap(self,mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = 255
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def __getitem__(self,idx):
        image = Image.open(self.images[idx])
        target = Image.open(self.targets[idx][0])
        image = np.array(image)
        target = np.array(target)
        # image = cv2.imread(self.images[idx])
        # target=cv2.imread(self.targets[idx][0])
        start = time.time()
        target = self.encode_segmap(target)
        fi = fisheye()
        if self.test_mode == False:
            fi.f0 = np.random.randint(500,1000)
            fi.pitch = np.random.uniform(-0.6,0)
        image = fi.norm2fisheye(image)
        target = fi.norm2fisheye(target,label=True)
        inputs = self.feature_extractor(image,target,return_tensors='pt')
        for k,v in inputs.items():
            inputs[k].squeeze_()
        return inputs

model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        )
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")

train_dataset = mydataset(
    '/mnt/hdd/dataset/cityscapes/extracted/',
                split='train',
                mode='fine',
                target_type='semantic'                
)

val_dataset = mydataset(
    '/mnt/hdd/dataset/cityscapes/extracted/',
                split='val',
                mode='fine',
                target_type='semantic' ,
                test_mode=True               
)

train_dataloader = DataLoader(val_dataset,batch_size=1)
inputs = next(iter(train_dataloader))
# print(output)

img = inputs['pixel_values']
res = inputs['labels']
# print(torch.unique(img))
outputs = model(img,res)
loss,prediction = outputs[0],outputs[1]
# prediction = outputs.logits
# criterion = nn.CrossEntropyLoss(ignore_index=255)
# loss = criterion(prediction,res)
print(loss)
upsampled_logits = nn.functional.interpolate(
            prediction, 
            size=res.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
predicted = upsampled_logits.argmax(dim=1)
# label_color = Colorize()(res)
# img=label_color.numpy()
# img=img.transpose(1,2,0)
# cv2.imwrite('test03.png',res)
val_mean_iou = evaluate.load('mean_iou')
val_mean_iou.add_batch(
    predictions=predicted,
    references=res
)
metrics = val_mean_iou.compute(
              num_labels=19,
              ignore_index=255,
              reduce_labels=False,
          )
# print(torch.unique(predicted))
# print(torch.unique(res))
print(metrics)
# target =np.array(target)
# img = np.array(img)
# start = time.time()
# target = fisheye(test_mode=True)
# # target.norm2fisheye(target)
# print(time.time()-start)
# print(target.fimg)
# cv2.imwrite('1.png',img)
