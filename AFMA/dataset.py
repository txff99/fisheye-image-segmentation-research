from torch.utils.data import Dataset
import os
from PIL import Image
from transformers import SegformerFeatureExtractor
import numpy as np
import PIL


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor,test=False):
        self.img_dir='/mnt/hdd/dataset/woodscape/rgb_images'
        self.label_dir='/mnt/hdd/dataset/woodscape/semantic_annotations/gtLabels'
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
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

        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt",size = {"height" : 512, "width": 512})
        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        return encoded_inputs