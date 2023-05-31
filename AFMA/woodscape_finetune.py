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
from metrics import calculate_metrics
from model import model
from dataset import SemanticSegmentationDataset
from loss import MyLoss_correction

class SegformerFinetuner(pl.LightningModule):
    
    def __init__(self, patch_size=16,learning_rate= 2e-4,train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        # self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.learning_rate=learning_rate
        self.num_classes = 10 #len(id2label.keys())
        # self.label2id = {v:k for k,v in self.id2label.items()}
        self.model = model(patch_size=patch_size)
        self.Loss = MyLoss_correction(patch_size=patch_size)
        # self.model = SegformerForSemanticSegmentation.from_pretrained(
        #     "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        #     num_labels=10,
        #     return_dict=False,
        #     ignore_mismatched_sizes=True
        # )
        # pretrained_dict = torch.load("/mnt/ssd/home/tianxiaofeng/distill_fisheye/pretrain/lightning_logs/mlp99/checkpoints/new_epoch=99-step=7500.ckpt")['state_dict']
        # pretrained_dict = {key.replace('model.',''): value for key,value in pretrained_dict.items()}
        # del pretrained_dict['decode_head.classifier.bias']
        # del pretrained_dict['decode_head.classifier.weight']
        # self.model.load_state_dict(pretrained_dict,strict=False)
        # for i in self.model.named_parameters():
        #     print(i[0])
        # for param in self.named_parameters():
        #     if 'decode_head' in param[0]:
        #         param[1].requires_grad=True
        #     else: param[1].requires_grad=False
        # self.model = SegformerForSemanticSegmentation.from_pretrained(
        #     "nvidia/segformer-b0-finetuned-ade-512-512", 
        #     return_dict=False, 
        #     num_labels=self.num_classes,
        #     id2label=self.id2label,
        #     label2id=self.label2id,
        #     ignore_mismatched_sizes=True,
        # )
        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")
    

    # def forward(self, images, masks):
    #     outputs = self.model(pixel_values=images, labels=masks)
    #     return(outputs)
    
    def training_step(self, batch, batch_nb):

        images, masks = batch['pixel_values'], batch['labels']
    # model.eval()
        
        output,attention = self.model(images)
        
        loss = self.Loss(output,masks,attention)

        return({'loss': loss})
    
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('loss',avg_train_loss,on_epoch=True,on_step=False)

    def validation_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
    # model.eval()
        output,attention = self.model(images)
        
        loss = self.Loss(output,masks,attention)
        masks = F.one_hot(masks,num_classes=10).permute(0,3,1,2)
        # print(masks.shape)
        predicted = output.argmax(dim=1)
        predicted = F.one_hot(predicted,num_classes=10).permute(0,3,1,2)
        cm = {}
        cm = calculate_metrics(predicted,masks)
        # self.val_mean_iou.add_batch(
        #     predictions=predicted.detach().cpu().numpy(), 
        #     references=masks.detach().cpu().numpy()
        # )
        # log = {"val_loss:":loss}
        # self.log('val_loss',loss)
        return({'val_loss': loss,'val_miou':cm['miou'],
        'mean_acc':cm['macc'],'per_iou':cm['per_iou'],'per_acc':cm['per_acc']})
    
    def validation_epoch_end(self, outputs):
        # metrics = self.val_mean_iou.compute(
        #       num_labels=self.num_classes, 
        #       ignore_index=255, 
        #       reduce_labels=False,
        #   )
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_miou = torch.stack([x["val_miou"] for x in outputs]).mean()
        avg_val_macc =  torch.stack([x["mean_acc"] for x in outputs]).mean()
        piou = [x["per_iou"] for x in outputs]
        pacc = [x["per_acc"] for x in outputs]
        category_iou = {}
        category_acc = {}
        for i in range(1,10):
            li = []
            for j in piou:
                if i in j:
                    li.append(j[i])
            category_iou[i] = sum(li)/len(li)
            li = []
            for j in pacc:
                if i in j:
                    li.append(j[i])
            category_acc[i] = sum(li)/len(li)
            
        print(f"per_iou:{category_iou}")
        print(f"per_acc:{category_acc}")
        # avg_per_miou =  torch.stack([x["per_iou"] for x in outputs]).mean()
        # avg_per_macc =  torch.stack([x["per_acc"] for x in outputs]).mean()
        # val_mean_iou = metrics["mean_iou"]
        # val_mean_accuracy = metrics["mean_accuracy"]
        # log = {"val_loss":avg_val_loss}
        metrics = {"val_loss": avg_val_loss, "val_mean_iou":avg_val_miou,"val_mean_acc":avg_val_macc}
        for k,v in metrics.items():
            self.log(k,v)
        # self.log("avg_val_loss",avg_val_loss,on_epoch=True,on_step=False)
        return ({'avg_val_loss':avg_val_loss,"val_mean_iou":avg_val_miou,"val_mean_acc":avg_val_macc})
    
    def test_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
    # model.eval()
        output,attention = self.model(images)
        loss = self.Loss(output,masks,attention)
        # loss, logits = outputs[0], outputs[1]
        # upsampled_logits = nn.functional.interpolate(
        #     logits, 
        #     size=masks.shape[-2:], 
        #     mode="bilinear", 
        #     align_corners=False
        # )
        
        predicted = output.argmax(dim=1)
        
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
            
        return({'test_loss': loss})
    
    def test_epoch_end(self, outputs):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
       
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}
        for k,v in metrics.items():
            self.log(k,v)
        print(f"test_loss,{avg_test_loss},test_mean_iou,{test_mean_iou},test_mean_accuracy,{test_mean_accuracy}")
        
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
        # filter(lambda p: p.requires_grad,self.parameters
        # for i in self.named_parameters():
        #     print(i[0])
        #     if 
        # sys.exit()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=self.learning_rate)
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
        )
        scheduler = {"scheduler":scheduler, "interval":"step","frequency":1}
        
        return [optimizer],[scheduler]

if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')

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

    batch_size = 40
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=4)#pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=4)#pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=4)#,pin_memory=True)#,num_workers=3,prefetch_factor=8

    # segformer_finetuner=SegformerFinetuner.load_from_checkpoint(
    #     "/mnt/ssd/home/tianxiaofeng/segformer_train/lightning_logs/version_24/checkpoints/epoch=6-step=1250.ckpt"
    # )
    
    segformer_finetuner = SegformerFinetuner(
        learning_rate=2e-4,
        patch_size=8,
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )

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
                        monitor='val_mean_iou',
                        mode='max',
                        )
                        # every_n_train_steps=206,
                        # save_weights_only=True,
                        # save_on_train_epoch_end=True,
                        # save_last=True,
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[3],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=500,
        check_val_every_n_epoch=1)
        # log_every_n_steps=20,
    #     resume_from_checkpoint="/mnt/ssd/home/tianxiaofeng/segformer_train/lightning_logs/version_24/checkpoints/epoch=6-step=1250.ckpt"
    # )
    # trainer.tune(segformer_finetuner)
    trainer.validate(segformer_finetuner
    ,  ckpt_path="/mnt/ssd/home/tianxiaofeng/AFMA/lightning_logs/version_8/checkpoints/epoch=467-step=96408.ckpt")
                #   ckpt_path="/mnt/ssd/home/tianxiaofeng/fisheye_aug/lightning_logs/version_1/checkpoints/epoch=99-step=16500.ckpt")
# function ConnectButton(){
#     console.log("Connect pushed"); 
#     document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
# }

# Interval(ConnectButton,60000);
