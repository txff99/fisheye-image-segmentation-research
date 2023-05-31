from torch.utils.data import Dataset
import os
from transformers import SegformerFeatureExtractor
import numpy as np
from PIL import Image
import PIL
import cv2
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from norm2fisheye import fisheye

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
        self.class_map = dict(zip(self.valid_classes, [21,0,19,20,11,12,13,18,17,7]))
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
        if not self.test:
            width = np.random.randint(1000,1280)
            height = int(width/1280*966)
            left = np.random.randint(0,1280-width)
            top = np.random.randint(0,966-height)
            # box = (left,top,left+width,top+height)
            # image = image.crop(box)
            image = image[left:left+width,top:top+height]
            segmentation_map=segmentation_map[left:left+width,top:top+height]
            if np.random.random(1)[0] >0.5:
                image = cv2.flip(image,1)
                segmentation_map = cv2.flip(segmentation_map,1)

        encoded_inputs = self.feature_extractor(image, segmentation_map.transpose(2,0,1), return_tensors="pt",size = {"height" : 512, "width": 512})
        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        return encoded_inputs

class cityscape_dataset(Cityscapes):
    def __init__(self,root,split,mode,target_type,test_mode=False):
        super().__init__(root,split,mode,target_type)
        # self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.test_mode = test_mode
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    

    def encode_segmap(self,mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = 21
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def __getitem__(self,idx):
        image = cv2.imread(self.images[idx])
        target = cv2.imread(self.targets[idx][0])
        image = cv2.resize(image,dsize=(512,512))
        target = cv2.resize(target,dsize=(512,512))
        target = self.encode_segmap(target)[:,:,0]
        # fi = fisheye()
        # if self.test_mode == False:
        #     fi.f0 = np.random.randint(200,600)
        #     fi.pitch = np.random.uniform(-0.8,0)
        #     fi.trans_x = np.random.uniform(0.5,1.5)
        
        # map_x,map_y = fi.norm2fisheye(image)
        # target = fi.norm2fisheye(target,label=True)
        inputs = self.feature_extractor(image,target,return_tensors='pt',size = {"height" : 1024, "width": 2048})
        # ori_inputs =self.feature_extractor(image,return_tensors='pt')
        for k,v in inputs.items():
            inputs[k].squeeze_()
        # ori_inputs['pixel_values'].squeeze_()

        return inputs#,map_x,map_y



class cy_dataset(Cityscapes):
    def __init__(self,root,split,mode,target_type,test_mode=False,distill=False):
        super().__init__(root,split,mode,target_type)
        # self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.test_mode = test_mode
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.distill_mode = distill

    def encode_segmap(self,mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = 255
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def __getitem__(self,idx):
        image = cv2.imread(self.images[idx])
        image = cv2.resize(image,dsize=(512,512))
        fi = fisheye()
        if self.test_mode == False:
            fi.f0 = np.random.randint(200,600)
            fi.pitch = np.random.uniform(-0.8,0)
            fi.trans_x = np.random.uniform(0.5,1.5)
        map_x,map_y = fi.norm2fisheye(image)
        # target = fi.norm2fisheye(target,label=True)
        if self.distill_mode == False:
            target = cv2.imread(self.targets[idx][0])
            target = cv2.resize(target,dsize=(512,512))
            target = self.encode_segmap(target)[:,:,0]
            inputs = self.feature_extractor(image,target,return_tensors='pt',size = {"height" : 512, "width": 512})
        else:
            inputs = self.feature_extractor(image,return_tensors='pt',size = {"height" : 512, "width": 512})

        # ori_inputs =self.feature_extractor(image,return_tensors='pt')
        for k,v in inputs.items():
            inputs[k].squeeze_()
        # ori_inputs['pixel_values'].squeeze_()

        return inputs,map_x,map_y


class wd_dataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir,test=False):
        self.img_dir='/mnt/hdd/dataset/woodscape/rgb_images'
        self.label_dir='/mnt/hdd/dataset/woodscape/semantic_annotations/gtLabels'
        self.root_dir = root_dir
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.test = test
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

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.label_dir, self.images[idx]))
        
        
        #data augmentation
        if not self.test:
            width = np.random.randint(1000,1280)
            height = int(width/1280*966)
            left = np.random.randint(0,1280-width)
            top = np.random.randint(0,966-height)
            box = (left,top,left+width,top+height)
            image = image.crop(box)
            segmentation_map=segmentation_map.crop(box)
            if np.random.random(1)[0] >0.5:
                image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                segmentation_map = segmentation_map.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()

        return encoded_inputs



# class combined_dataloader(DataLoader):
#     def __init__(self,cityspapes,woodscape,)


