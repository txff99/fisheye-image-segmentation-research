from torch.utils.data import Dataset
import os
# from PIL import Image
from transformers import SegformerFeatureExtractor
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from transformers import (
        get_constant_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup)
import torch
from torch import nn
import time
import numpy as np
import sys
import cv2
# import PIL
from torchvision.datasets import Cityscapes
from norm2fisheye import fisheye
import torch.nn.functional as F
# from meaniou import meanIOU
from metrics import calculate_metrics
from model.segformer import Segformer
import matplotlib.pyplot as plt
from visualize import Colorize
from wd_dataset import woodscape_dataset

class mydataset(Cityscapes):
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
            mask[mask == _voidc] = 19
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def __getitem__(self,idx):
        image = cv2.imread(self.images[idx])
        target = cv2.imread(self.targets[idx][0])
        image = cv2.resize(image,dsize=(512,512))
        target = cv2.resize(target,dsize=(512,512))
        target = self.encode_segmap(target)[:,:,0]
        fi = fisheye()
        if self.test_mode == False:
            fi.f0 = np.random.randint(200,600)
            fi.pitch = np.random.uniform(-0.8,0)
            fi.trans_x = np.random.uniform(0.5,1.5)
        map_x,map_y = fi.norm2fisheye(image)
        # target = fi.norm2fisheye(target,label=True)
        inputs = self.feature_extractor(image,target,return_tensors='pt',size = {"height" : 512, "width": 512})
        # ori_inputs =self.feature_extractor(image,return_tensors='pt')
        for k,v in inputs.items():
            inputs[k].squeeze_()
        # ori_inputs['pixel_values'].squeeze_()

        return inputs,map_x,map_y

def convert2fisheye(imgs,grid,label=False):
    if label==False:
        outp = F.grid_sample(imgs,grid=grid,mode='bilinear')
    else:
        outp = F.grid_sample(imgs.unsqueeze(dim=1).float(),grid=grid,mode='nearest')
    return outp

def mse_loss(input1, target, ignored_index=255, reduction='mean'):
    mask = target == ignored_index
    out = (input1[~mask]-target[~mask])**2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out

class SegformerFinetuner(pl.LightningModule):
    
    def __init__(self, learning_rate= 2e-4,train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        # self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.learning_rate=learning_rate
        self.num_classes = 19 #len(id2label.keys())
        # self.MSEloss = mse_loss#nn.MSELoss()
        self.CEloss = nn.CrossEntropyLoss(ignore_index=19)
        # self.t_model = Segformer()
        # for param in self.t_model.named_parameters():
        #     param[1].requires_grad=False
        self.s_model = Segformer()
            # return_dict=False,
        pretrained_dict = torch.load('/mnt/ssd/home/tianxiaofeng/distill_fisheye/model/segformer.b0.1024x1024.city.160k.pth')['state_dict']
        # pretrained_dict = {key.replace('model.',''): value for key,value in pretrained_dict.items()}
        # self.model.load_state_dict(pretrained_dict)
        # self.t_model.load_state_dict(pretrained_dict,strict=False)
        self.s_model.load_state_dict(pretrained_dict,strict=False)

    def training_step(self, batch, batch_nb):
        # images,masks,map_x,map_y = batch[0]['pixel_values'],batch[1]['pixel_values'], batch[1]['labels'],batch[2],batch[3]
        images,masks,map_x,map_y = batch[0]['pixel_values'],batch[0]['labels'],batch[1],batch[2]
        B,C,w,h = images.shape
        map_x = [torch.tensor(i).float().to(images.device) for i in map_x]
        map_y = [torch.tensor(i).float().to(images.device) for i in map_y]
        map_x = torch.stack(map_x,dim=0)
        map_y = torch.stack(map_y,dim=0)
        grid = torch.stack((map_x/((w)/2)-1,map_y/((h)/2)-1),dim=3)#.unsqueeze(0)

        fimages = convert2fisheye(imgs=images,grid=grid)
        masks = convert2fisheye(imgs=masks-19,grid=grid,label=True)
        masks = masks.squeeze(dim=1)+19 #padding with 255

        s_outputs = self.s_model(fimages)
        s_outputs = nn.functional.interpolate(
            s_outputs, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        masks = torch.tensor(masks, dtype=torch.long)
        loss = self.CEloss(s_outputs,masks)
        # loss= 0.7*CEloss + 0.3*mse_loss(s_hidden,t_hidden,ignored_index=0)
        
        
        return({'loss': loss})
    
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('loss',avg_train_loss,on_epoch=True,on_step=False)

    def validation_step(self, batch, batch_nb):
        images,masks = batch['pixel_values'],batch['labels']
        # print(masks.shape)
        # f_inputs = feature_extractor(fimages,return_tensors='pt')
        # ori_inputs = feature_extractor(images,return_tensors='pt')
        
        # s_outputs = self.s_model(pixel_values=fimages,labels=masks)
        outputs = self.s_model(images)
        # outputs = self.wd_head(outputs)
        
        outputs = nn.functional.interpolate(
            outputs, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        masks = torch.tensor(masks[:,0,:,:], dtype=torch.long)
        # CEloss = self.CEloss(outputs,masks)
        # loss = self.CEloss(outputs,masks)
        
        # loss= 0.7*CEloss + 0.3*mse_loss(s_hidden,t_hidden,ignored_index=0)
        
        # print(masks.shape)
        masks = F.one_hot(masks,num_classes=20).permute(0,3,1,2)
        predicted = outputs.argmax(dim=1)
        predicted = F.one_hot(predicted,num_classes=19).permute(0,3,1,2)
        cm = {}
        cm = calculate_metrics(predicted,masks)
        return({'val_miou':cm['miou'],'mean_acc':cm['macc']})
    
    
    def validation_epoch_end(self, outputs):
        # avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_miou = torch.stack([x["val_miou"] for x in outputs]).mean()
        avg_val_macc =  torch.stack([x["mean_acc"] for x in outputs]).mean()
        metrics = {"val_mean_iou":avg_val_miou,"val_mean_acc":avg_val_macc}
        
        for k,v in metrics.items():
            self.log(k,v)
        # return ({'avg_val_loss':avg_val_loss})
    
    def test_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
            
        return({'test_loss': loss})
    
    def test_epoch_end(self, outputs):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=0, 
              reduce_labels=False,
          )
       
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}
        
        for k,v in metrics.items():
            self.log(k,v)
        
        return metrics
    
    # def configure_optimizers(self):
    #     return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,self.parameters()),lr=self.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=8000
        # )
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=200,
        )
        scheduler = {"scheduler":scheduler, "interval":"step","frequency":1}
        
        return [optimizer],[scheduler]

if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    # print(dir(feature_extractor))
    # feature_extractor.reduce_labels = False
    # feature_extractor.size = 128
    train_dataset = mydataset(
    '/mnt/hdd/dataset/cityscapes/extracted/',
                split='train',
                mode='fine',
                target_type='semantic'                
    )

    # val_dataset = mydataset(
    #     '/mnt/hdd/dataset/cityscapes/extracted/',
    #                 split='val',
    #                 mode='fine',
    #                 target_type='semantic' ,
    #                 test_mode=True               
    # )
    # test_dataset = mydataset(
    #     '/mnt/hdd/dataset/cityscapes/extracted/',
    #                 split='test',
    #                 mode='fine',
    #                 target_type='semantic' ,
    #                 test_mode=True               
    # )

    root_dir='/mnt/hdd/dataset/woodscape/rgb_images'
    label_dir='/mnt/hdd/dataset/woodscape/semantic_annotations'

    train_dir='/mnt/hdd/dataset/woodscape/lists/train.txt'
    test_dir='/mnt/hdd/dataset/woodscape/lists/val.txt'

    # trainwd_dataset = wd_dataset(train_dir)
    # val_dataset = SemanticSegmentationDataset("./roboflow/valid/", feature_extractor)
    valwd_dataset = woodscape_dataset(test_dir,test=True)

    batch_size = 40
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=4)#pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(valwd_dataset, batch_size=batch_size,num_workers=4)#pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size)#,pin_memory=True)#,num_workers=3,prefetch_factor=8

    # segformer_finetuner=SegformerFinetuner.load_from_checkpoint(
    #     "/mnt/ssd/home/tianxiaofeng/segformer_train/lightning_logs/version_24/checkpoints/epoch=6-step=1250.ckpt"
    # )
    segformer_finetuner = SegformerFinetuner(
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
    )
        # test_dataloader=test_dataloader

    lr_monitor = LearningRateMonitor()

    early_stop_callback = EarlyStopping(
        monitor="avg_val_loss", 
        min_delta=0.00, 
        patience=7, 
        verbose=False, 
        mode="min",
    )   

    checkpoint_callback = ModelCheckpoint(
                        save_top_k=-1,
                        save_weights_only=True,
                        every_n_train_steps=750,
                        )
                        # save_on_train_epoch_end=True,
                        # save_last=True,
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[3],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=200,
        check_val_every_n_epoch=10)
        # log_every_n_steps=20,
    #     resume_from_checkpoint="/mnt/ssd/home/tianxiaofeng/segformer_train/lightning_logs/version_24/checkpoints/epoch=6-step=1250.ckpt"
    # )
    # trainer.tune(segformer_finetuner)
    # print(segformer.learning_rate)
    trainer.fit(segformer_finetuner)
            #   ckpt_path="/mnt/ssd/home/tianxiaofeng/distill_fisheye/lightning_logs/version_0/checkpoints/epoch=99-step=18600.ckpt")
# function ConnectButton(){
#     console.log("Connect pushed"); 
#     document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
# }

# Interval(ConnectButton,60000);
