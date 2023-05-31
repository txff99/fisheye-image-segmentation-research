from torch.utils.data import Dataset
import os
from transformers import SegformerFeatureExtractor
import numpy as np
from PIL import Image
import PIL
import cv2
class woodscape_dataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, 
                root_dir, 
                test=False):
        self.img_dir='/mnt/hdd/dataset/woodscape/rgb_images'
        self.label_dir='/mnt/hdd/dataset/woodscape/semantic_annotations/gtLabels'
        self.root_dir = root_dir
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    
        self.test = test

        self.valid_classes = range(10)
        self.class_map = dict(zip(self.valid_classes, [19,0,19,19,11,12,13,18,17,7]))
        # 21 is the ignored class

        image_file_names = []
        with open(self.root_dir) as f:
            lines = f.readlines()
            for i in lines:
            # if 'MVR' in i or 'MVL' in i:
            #     pass
            # else:
                image_file_names.append(i.replace('\n',''))
        
        if '00034_FV.png' in image_file_names:
            image_file_names.remove('00034_FV.png')

        self.images = sorted(image_file_names)
        self.images=np.array(self.images)
        # self.labels = sorted(label_file_names)

    def __len__(self):
        return len(self.images)

    def encode_segmap(self,mask):
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = cv2.imread(os.path.join(self.label_dir, self.images[idx]))
        segmentation_map = self.encode_segmap(segmentation_map)
        #data augmentation
        # if not self.test:
        #     width = np.random.randint(1000,1280)
        #     height = int(width/1280*966)
        #     left = np.random.randint(0,1280-width)
        #     top = np.random.randint(0,966-height)
        #     # box = (left,top,left+width,top+height)
        #     # image = image.crop(box)
        #     image = image[left:left+width,top:top+height]
        #     segmentation_map=segmentation_map[left:left+width,top:top+height]
        #     if np.random.random(1)[0] >0.5:
        #         image = cv2.flip(image,1)
        #         segmentation_map = cv2.flip(segmentation_map,1)

        encoded_inputs = self.feature_extractor(image, segmentation_map.transpose(2,0,1), return_tensors="pt",size = {"height" : 512, "width": 512})
        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        return encoded_inputs
