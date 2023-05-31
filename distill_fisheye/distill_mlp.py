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
        SegformerModel,
        SegformerDecodeHead,
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
from torchvision.datasets import Cityscapes
from norm2fisheye import fisheye


class mydataset(Cityscapes):
    def __init__(self,root,split,mode,target_type,test_mode=False):
        super().__init__(root,split,mode,target_type)
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.test_mode = test_mode
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
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
        target = self.encode_segmap(target)
        fi = fisheye()
        if self.test_mode == False:
            fi.f0 = np.random.randint(500,1000)
            fi.pitch = np.random.uniform(-0.6,0)
        fimage = fi.norm2fisheye(image)
        target = fi.norm2fisheye(target,label=True)
        inputs = self.feature_extractor(fimage,target,return_tensors='pt')
        ori_inputs =self.feature_extractor(image,return_tensors='pt')
        for k,v in inputs.items():
            inputs[k].squeeze_()
        ori_inputs['pixel_values'].squeeze_()
        return ori_inputs,inputs

class mlp(nn.Module):
    def __init__(self,
                infeatures,
                hidden_features,
                out_features
                ):
        super().__init__()
        self.infeatures = infeatures
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.layers = nn.Sequential(
          nn.Linear(self.infeatures,self.hidden_features,bias=True),
          nn.ReLU(),
          nn.Linear(self.hidden_features,self.out_features,bias=False)  
        ) 
    def forward(self,x):
        x = self.layers(x)
        return x
# class t_model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = SegformerModel.from_pretrained(
#             "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
#         )
    
#     def forward(self,x):
#         x = self.model(pixel_values=x)[0]
#         x = x.transpose(1,3).mean


class SegformerFinetuner(pl.LightningModule):
    
    def __init__(self, learning_rate= 2e-5,train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        # self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.learning_rate=learning_rate
        self.num_classes = 19 #len(id2label.keys())
        self.MSEloss = nn.MSELoss()
        # self.label2id = {v:k for k,v in self.id2label.items()}
        
        self.t_model =  SegformerModel.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        )
            # return_dict=False,
            # output_hidden_states=True
        for param in self.t_model.named_parameters():
            param[1].requires_grad=False
        self.s_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            output_hidden_states=True
        )
        self.mlp = mlp(256,2048,256)
            # return_dict=False,
        # pretrained_dict = torch.load("/mnt/ssd/home/tianxiaofeng/segformer_train/lightning_logs/version_26/checkpoints/epoch=9-step=2050.ckpt")['state_dict']
        # pretrained_dict = {key.replace('model.',''): value for key,value in pretrained_dict.items()}
        # self.model.load_state_dict(pretrained_dict)

        # self.model = SegformerForSemanticSegmentation.from_pretrained(
        #     "nvidia/segformer-b0-finetuned-ade-512-512", 
        #     return_dict=False, 
        #     num_labels=self.num_classes,
        #     id2label=self.id2label,
        #     label2id=self.label2id,
        #     ignore_mismatched_sizes=True,
        # )
        # self.train_mean_iou = evaluate.load("mean_iou")
        # self.val_mean_iou = evaluate.load("mean_iou")
        # self.test_mean_iou = evaluate.load("mean_iou")
    
    def training_step(self, batch, batch_nb):
        images, fimages,masks = batch[0]['pixel_values'],batch[1]['pixel_values'], batch[1]['labels']
        # print(self.model)
        # print(images)
        t_outputs = self.t_model(pixel_values=images)
        s_outputs = self.s_model(pixel_values=fimages,labels=masks)
        t_hidden = t_outputs[0]
        s_hidden = s_outputs['hidden_states'][-1].flatten(2)
        s_hidden = self.mlp(s_hidden).unflatten(2,(16,16))
        CEloss = s_outputs[0]
        MSEloss = self.MSEloss(s_hidden,t_hidden)

        
        loss= 0.7*CEloss + 0.3*MSEloss
        
        
        #     # log ={"train_loss":loss}
        #     metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            
        #     for k,v in metrics.items():
        #         self.log(k,v)
            
        #     return metrics
        # else:
        # self.log('loss',loss)
        return({'loss': loss,'MSEloss':MSEloss,'CEloss':CEloss})
    

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        avg_train_celoss = torch.stack([x["CEloss"] for x in outputs]).mean()
        avg_train_mseloss = torch.stack([x["MSEloss"] for x in outputs]).mean()
        # val_mean_iou = metrics["mean_iou"]
        # val_mean_accuracy = metrics["mean_accuracy"]
        # log = {"val_loss":avg_val_loss}
        # metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
        # for k,v in metrics.items():
        #     self.log(k,v)
        self.log("avg_loss",avg_train_loss,on_epoch=True,on_step=False)
        self.log("avg_celoss",avg_train_celoss)
        self.log("avg_mseloss",avg_train_mseloss)
        # self.log('loss',avg_train_loss,on_epoch=True,on_step=False)

    def validation_step(self, batch, batch_nb):
        images, fimages,masks = batch[0]['pixel_values'],batch[1]['pixel_values'], batch[1]['labels']
        # print(self.model)
        # print(images)
        t_outputs = self.t_model(pixel_values=images)
        s_outputs = self.s_model(pixel_values=fimages,labels = masks)
        t_hidden = t_outputs[0]
        # s_hidden = s_outputs['hidden_states'][-1].transpose(1,3)
        # s_hidden = self.mlp(s_hidden).transpose(1,3)
        s_hidden = s_outputs['hidden_states'][-1].flatten(2)
        s_hidden = self.mlp(s_hidden).unflatten(2,(16,16))
        CEloss = s_outputs[0]

        MSEloss = self.MSEloss(s_hidden,t_hidden)
        loss= 0.7*CEloss + 0.3*MSEloss
        # upsampled_logits = nn.functional.interpolate(
        #     logits, 
        #     size=masks.shape[-2:], 
        #     mode="bilinear", 
        #     align_corners=False
        # )
        
        # predicted = upsampled_logits.argmax(dim=1)
        # self.val_mean_iou.add_batch(
        #     predictions=predicted.detach().cpu().numpy(), 
        #     references=masks.detach().cpu().numpy()
        # )
        # # log = {"val_loss:":loss}
        # self.log('val_loss',loss)
        return({'val_loss': loss,'MSEloss':MSEloss,'CEloss':CEloss})
    
    def validation_epoch_end(self, outputs):
        # metrics = self.val_mean_iou.compute(
        #       num_labels=self.num_classes, 
        #       ignore_index=255, 
        #       reduce_labels=False,
        #   )
        
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_celoss = torch.stack([x["CEloss"] for x in outputs]).mean()
        avg_val_mseloss = torch.stack([x["MSEloss"] for x in outputs]).mean()
        # val_mean_iou = metrics["mean_iou"]
        # val_mean_accuracy = metrics["mean_accuracy"]
        # log = {"val_loss":avg_val_loss}
        # metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
        # for k,v in metrics.items():
        #     self.log(k,v)
        self.log("avg_val_loss",avg_val_loss,on_epoch=True,on_step=False)
        self.log("avg_val_celoss",avg_val_celoss)
        self.log("avg_val_mseloss",avg_val_mseloss)
        # print(f'val_loss:{avg_val_loss},val_mean_iou:{val_mean_iou},val_mean_acc{val_mean_accuracy}')
        return ({'avg_val_loss':avg_val_loss})
    
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
            num_warmup_steps=0,
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

    val_dataset = mydataset(
        '/mnt/hdd/dataset/cityscapes/extracted/',
                    split='val',
                    mode='fine',
                    target_type='semantic' ,
                    test_mode=True               
    )
    test_dataset = mydataset(
        '/mnt/hdd/dataset/cityscapes/extracted/',
                    split='test',
                    mode='fine',
                    target_type='semantic' ,
                    test_mode=True               
    )

    batch_size = 40
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=4)#pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=4)#pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=4)#,pin_memory=True)#,num_workers=3,prefetch_factor=8

    # segformer_finetuner=SegformerFinetuner.load_from_checkpoint(
    #     "/mnt/ssd/home/tianxiaofeng/segformer_train/lightning_logs/version_24/checkpoints/epoch=6-step=1250.ckpt"
    # )
    segformer_finetuner = SegformerFinetuner(
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
                        save_weights_only=True,
                        every_n_train_steps=3750,
                        save_last=True,
                        )
                        # save_top_k=-1,
                        # save_on_train_epoch_end=True,
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[1],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=600,
        check_val_every_n_epoch=20)
        # log_every_n_steps=20,
    #     resume_from_checkpoint="/mnt/ssd/home/tianxiaofeng/segformer_train/lightning_logs/version_24/checkpoints/epoch=6-step=1250.ckpt"
    # )
    # trainer.tune(segformer_finetuner)
    # print(segformer.learning_rate)
    trainer.fit(segformer_finetuner,
              ckpt_path="/mnt/ssd/home/tianxiaofeng/distill_fisheye/lightning_logs/version_4/checkpoints/epoch=449-step=33750.ckpt")
# function ConnectButton(){
#     console.log("Connect pushed"); 
#     document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
# }

# Interval(ConnectButton,60000);
