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
from dataset import cityscape_dataset,woodscape_dataset,cy_dataset,wd_dataset
from loss import loss_correction
# import lightning.pytorch.utilities as ul

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

class head(nn.Module):
    def __init__(self,embedding_dim=256,num_classes=19):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
    def forward(self,x):
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x
        

class SegformerFinetuner(pl.LightningModule):
    
    def __init__(self, learning_rate= 2e-4,train_dataloader=None,val_dataloader=None ,metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        # self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        # self.test_dl = test_dataloader
        self.learning_rate=learning_rate
        # self.num_classes = 21 #len(id2label.keys())
        self.CEloss = nn.CrossEntropyLoss(ignore_index=255)
        self.model = Segformer()
        # self.wd_loss = loss_correction(cfg='woodscape')
        # self.cy_loss = loss_correction(cfg='')
            # return_dict=False,
        pretrained_dict = torch.load('/mnt/ssd/home/tianxiaofeng/distill_fisheye/model/segformer.b0.1024x1024.city.160k.pth')['state_dict']
        pretrained_dict = {key.replace('linear_pred.',''): value for key,value in pretrained_dict.items()}
        # self.model.load_state_dict(pretrained_dict)
        self.model.load_state_dict(pretrained_dict,strict=False)
        self.wd_head = head(num_classes=10)
        self.cy_head = head(num_classes=19)
        # self.s_model.load_state_dict(pretrained_dict,strict=False)

    def training_step(self, batch, batch_nb):
        # images,masks,map_x,map_y = batch[0]['pixel_values'],batch[1]['pixel_values'], batch[1]['labels'],batch[2],batch[3]
        cy,wd = batch[1],batch[0]
        images,masks,map_x,map_y = cy[0]['pixel_values'],cy[0]['labels'],cy[1],cy[2]
        B,C,w,h = images.shape
        map_x = [torch.tensor(i).float().to(images.device) for i in map_x]
        map_y = [torch.tensor(i).float().to(images.device) for i in map_y]
        map_x = torch.stack(map_x,dim=0)
        map_y = torch.stack(map_y,dim=0)
        grid = torch.stack((map_x/((w)/2)-1,map_y/((h)/2)-1),dim=3)#.unsqueeze(0)

        fimages = convert2fisheye(imgs=images,grid=grid)
        masks = convert2fisheye(imgs=masks-255,grid=grid,label=True)
        masks = masks.squeeze(dim=1)+255 #padding with 255
        # print(masks.shape)
        # print(masks)
        # label_color = Colorize()(masks)
        # img=label_color.numpy()
        # img=img.transpose(1,2,0)
        # cv2.imwrite('./demo/debug2.png',img)
        # sys.exit()
        # print(masks.shape)
        # f_inputs = feature_extractor(fimages,return_tensors='pt')
        # ori_inputs = feature_extractor(images,return_tensors='pt')
        
        # _,t_hidden = self.t_model(images)
        # s_outputs = self.s_model(pixel_values=fimages,labels=masks)
        outputs = self.model(fimages)
        outputs = self.cy_head(outputs)
    
        outputs = nn.functional.interpolate(
            outputs, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        masks = torch.tensor(masks, dtype=torch.long)
        # CEloss = self.CEloss(outputs,masks)
        cy_loss = self.CEloss(outputs,masks)
    
        images,masks = wd['pixel_values'],wd['labels']
        outputs = self.model(images)
        outputs = self.wd_head(outputs)
    
        outputs = nn.functional.interpolate(
            outputs, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        masks = torch.tensor(masks, dtype=torch.long)
        # CEloss = self.CEloss(outputs,masks)
        wd_loss = self.CEloss(outputs,masks)
        loss = 5*wd_loss+cy_loss
            # loss= 0.7*CEloss + 0.3*mse_loss(s_hidden,t_hidden,ignored_index=0)
        return({'loss': loss})
    
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('loss',avg_train_loss,on_epoch=True,on_step=False)

    def validation_step(self, batch, batch_nb):
        # print(batch2)
        # sys.exit()
        images,masks = batch['pixel_values'],batch['labels']
        # print(masks.shape)
        # f_inputs = feature_extractor(fimages,return_tensors='pt')
        # ori_inputs = feature_extractor(images,return_tensors='pt')
        
        # s_outputs = self.s_model(pixel_values=fimages,labels=masks)
        outputs = self.model(images)
        outputs = self.wd_head(outputs)
        
        outputs = nn.functional.interpolate(
            outputs, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        masks = torch.tensor(masks, dtype=torch.long)
        # CEloss = self.CEloss(outputs,masks)
        loss = self.CEloss(outputs,masks)
        
        # loss= 0.7*CEloss + 0.3*mse_loss(s_hidden,t_hidden,ignored_index=0)
        
        
        masks = F.one_hot(masks,num_classes=10).permute(0,3,1,2)
        predicted = outputs.argmax(dim=1)
        predicted = F.one_hot(predicted,num_classes=10).permute(0,3,1,2)
        cm = {}
        cm = calculate_metrics(predicted,masks)
        return({'val_loss': loss,'val_miou':cm['miou'],'mean_acc':cm['macc']})
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_miou = torch.stack([x["val_miou"] for x in outputs]).mean()
        avg_val_macc =  torch.stack([x["mean_acc"] for x in outputs]).mean()
        metrics = {"val_loss": avg_val_loss, "val_mean_iou":avg_val_miou,"val_mean_acc":avg_val_macc}
        
        for k,v in metrics.items():
            self.log(k,v)
        return metrics
    
    # def test_step(self, batch, batch_nb):
        
    #     images, masks = batch['pixel_values'], batch['labels']
        
    #     outputs = self(images, masks)
        
    #     loss, logits = outputs[0], outputs[1]
        
    #     upsampled_logits = nn.functional.interpolate(
    #         logits, 
    #         size=masks.shape[-2:], 
    #         mode="bilinear", 
    #         align_corners=False
    #     )
        
    #     predicted = upsampled_logits.argmax(dim=1)
        
    #     self.test_mean_iou.add_batch(
    #         predictions=predicted.detach().cpu().numpy(), 
    #         references=masks.detach().cpu().numpy()
    #     )
            
    #     return({'test_loss': loss})
    
    # def test_epoch_end(self, outputs):
    #     metrics = self.test_mean_iou.compute(
    #           num_labels=self.num_classes, 
    #           ignore_index=0, 
    #           reduce_labels=False,
    #       )
       
    #     avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     test_mean_iou = metrics["mean_iou"]
    #     test_mean_accuracy = metrics["mean_accuracy"]

    #     metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}
        
    #     for k,v in metrics.items():
    #         self.log(k,v)
        
    #     return metrics
    
    # def configure_optimizers(self):
    #     return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    # def test_dataloader(self):
    #     return self.test_dl

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,self.parameters()),lr=self.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=8000
        # )
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
        )
        scheduler = {"scheduler":scheduler, "interval":"step","frequency":1}
        
        return [optimizer],[scheduler]

# class datamodule(pl.LightningDataModule):
#     def __init__(self,train_dataloader,val_dataloader):
#         super().__init__()
#         self.train_dataloader=train_dataloader
#         self.val_dataloader=val_dataloader
#     def train_dataloader(self):
#         return self.train_dataloader
#     def val_dataloader(self):
#         return self.val_dataloader



if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')

    traincy_dataset = cy_dataset(
    '/mnt/hdd/dataset/cityscapes/extracted/',
                split='train',
                mode='fine',
                target_type='semantic'                
    )

    valcy_dataset = cy_dataset(
        '/mnt/hdd/dataset/cityscapes/extracted/',
                    split='val',
                    mode='fine',
                    target_type='semantic' ,
                    test_mode=True               
    )

    root_dir='/mnt/hdd/dataset/woodscape/rgb_images'
    label_dir='/mnt/hdd/dataset/woodscape/semantic_annotations'

    train_dir='/mnt/hdd/dataset/woodscape/lists/train.txt'
    test_dir='/mnt/hdd/dataset/woodscape/lists/val.txt'

    trainwd_dataset = wd_dataset(train_dir)
    # val_dataset = SemanticSegmentationDataset("./roboflow/valid/", feature_extractor)
    valwd_dataset = wd_dataset(test_dir,test=True)


    batch_size = 24
    traincy_dataloader = DataLoader(traincy_dataset, batch_size=batch_size,shuffle=True,num_workers=4)#pin_memory=True, shuffle=True)
    trainwd_dataloader = DataLoader(trainwd_dataset, batch_size=batch_size,shuffle=True,num_workers=4)#pin_memory=True, shuffle=True)
    train_dataloader = [trainwd_dataloader,traincy_dataloader]
    # valcy_dataloader = DataLoader(valcy_dataset, batch_size=batch_size,shuffle=False,num_workers=4)#pin_memory=True, shuffle=True)
    valwd_dataloader = DataLoader(valwd_dataset, batch_size=batch_size,shuffle=False,num_workers=4)#pin_memory=True, shuffle=True)
    val_dataloader = valwd_dataloader
    # dm = datamodule(train_dataloader,val_dataloader)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=4)#pin_memory=True)
    # test_dataloader = DataLoader(testwd_dataset, batch_size=batch_size)#,pin_memory=True)#,num_workers=3,prefetch_factor=8

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
                        save_top_k=3,
                        monitor="val_mean_iou",
                        mode='max',
                        save_weights_only=True,
                        filename="{epoch}-{step}-{val_mean_iou:.4f}"
                        )
                        # every_n_train_steps=2750,
                        # save_last=True,
                        # save_on_train_epoch_end=True,
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[3],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=600,
        check_val_every_n_epoch=1,
        multiple_trainloader_mode="max_size_cycle"
        )
        # multiple_trainloader_mode='max_size_cycle'
        # log_every_n_steps=20,
    #     resume_from_checkpoint="/mnt/ssd/home/tianxiaofeng/segformer_train/lightning_logs/version_24/checkpoints/epoch=6-step=1250.ckpt"
    # )
    # trainer.tune(segformer_finetuner)
    # print(segformer.learning_rate)
    trainer.fit(segformer_finetuner,
                ckpt_path="/mnt/ssd/home/tianxiaofeng/combine_training/lightning_logs/multi_decoder_5.1_loss2/checkpoints/epoch=493-step=135850-val_mean_iou=0.5634.ckpt")
# function ConnectButton(){
#     console.log("Connect pushed"); 
#     document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
# }

# Interval(ConnectButton,60000);
