from torch.utils.data import Dataset
import os
from PIL import Image
from transformers import SegformerFeatureExtractor
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from transformers import (SegformerForSemanticSegmentation, 
        get_constant_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup)
import evaluate
import torch
from torch import nn
import time
import numpy as np
import sys
import PIL
import torch.nn.functional as F
# from meaniou import meanIOU
from model import model
from dataset import SemanticSegmentationDataset
from loss import MyLoss_correction
from metrics import compute_metrics
from meaniou import meanIOU
from metrics import calculate_metrics

if __name__ == '__main__':
        root_dir='/mnt/hdd/dataset/woodscape/rgb_images'
        label_dir='/mnt/hdd/dataset/woodscape/semantic_annotations'

        train_dir='/mnt/hdd/dataset/woodscape/lists/train.txt'
        test_dir='/mnt/hdd/dataset/woodscape/lists/val.txt'
        # torch.multiprocessing.set_start_method('spawn')
        feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        # feature_extractor.reduce_labels = False
        # feature_extractor.size = 128

        train_dataset = SemanticSegmentationDataset(train_dir, feature_extractor)
        # val_dataset = SemanticSegmentationDataset("./roboflow/valid/", feature_extractor)
        test_dataset = SemanticSegmentationDataset(test_dir, feature_extractor,test=True)

        batch_size = 1
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=4)#pin_memory=True, shuffle=True)
        val_dataloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=4)#pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=4)#,pin_memory=True)#,num_workers=3,prefetch_factor=8

        batch = next(iter(train_dataloader))
        images, masks = batch['pixel_values'], batch['labels']
        # model.eval()
        mymodel = model(patch_size=16)
        pretrained_dict = torch.load("/mnt/ssd/home/tianxiaofeng/AFMA/lightning_logs/patch16/version_2/checkpoints/epoch=299-step=49500.ckpt",map_location='cpu')['state_dict']
        pretrained_dict = {key.replace('model.','',1): value for key,value in pretrained_dict.items()}
        mymodel.load_state_dict(pretrained_dict)
        output,attention = mymodel(images)
        # print(output.shape)
        # loss = MyLoss_correction()
        predicted = output.argmax(dim=1)

        # test_mean_iou=evaluate.load('mean_iou')
        # test_mean_iou.add_batch(
        #     predictions=predicted.detach().cpu().numpy(), 
        #     references=masks.detach().cpu().numpy()
        # )
        # metrics = test_mean_iou.compute(
        #       num_labels=10, 
        #       ignore_index=255, 
        #       reduce_labels=False,
        #   )
        # print(metrics)
        masks = F.one_hot(masks,num_classes=10).permute(0,3,1,2)
        # print(masks.shape)
        predicted = F.one_hot(predicted,num_classes=10).permute(0,3,1,2)
        # cm = compute_metrics(predicted,masks)
        # print(cm)
        
        metrics = calculate_metrics(predicted,masks)
        # print(metrics['per_iou'])
        # print()
        piou = [metrics['per_iou']]
        # print(piou)
        category = {}
        li = []
        for i in range(1,10):
            for j in piou:
                if i in j:
                    print(j[i])
                    li.append(j[i])
            category[i] = sum(li)/len(li)
        print(category)


                


        # meaniou = torch.from_numpy(meanIOU(masks,predicted))
        # print(meaniou)
        # # print(loss(output,masks,attention))