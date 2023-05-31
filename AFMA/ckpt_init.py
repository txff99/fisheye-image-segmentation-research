import torch

# ckpt=torch.load("/mnt/ssd/home/tianxiaofeng/segformer_train/lightning_logs/version_33/checkpoints/new_epoch=75-step=13000.ckpt")
ckpt=torch.load("/mnt/ssd/home/tianxiaofeng/AFMA/lightning_logs/version_7/checkpoints/epoch=338-step=69834.ckpt")
ckpt['callbacks']={}
# ckpt['optimizer_states']=[]
# ckpt['lr_schedulers']=[]
# ckpt['state_dict'] = {k.replace('t_model.','t_model.model.'):v for k,v in ckpt['state_dict'].items()}
# print(ckpt['callbacks'])
# print(ckpt['lr_schedulers'])
# print(ckpt['optimizer_states'])
torch.save(ckpt,"/mnt/ssd/home/tianxiaofeng/AFMA/lightning_logs/version_7/checkpoints/epoch=338-step=69834.ckpt")


# upsampled_logits = nn.functional.interpolate(
#             logits, 
#             size=masks.shape[-2:], 
#             mode="bilinear", 
#             align_corners=False
#         )

#         predicted = upsampled_logits.argmax(dim=1)
#         # start=time.time()
#         # # print(dir(self.train_mean_iou))
#         # self.train_mean_iou.add_batch(
#         #     predictions=predicted.detach().cpu().numpy(), 
#         #     references=masks.detach().cpu().numpy()
#         # )
#         # print(time.time()-start)
#         predicted=predicted.detach().cpu().numpy(), 
#         masks=masks.detach().cpu().numpy()
#         # print(predicted[0])
#         self.save_predict=list(self.save_predict)
#         self.save_label=list(self.save_label)
#         for i in range(32):
#             self.save_predict.append(predicted[0][i])
#             self.save_label.append(masks[i])
#         self.save_predict=np.array(self.save_predict,dtype='uint16')
#         self.save_label=np.array(self.save_label,dtype='uint16')
#         # self.train_mean_iou.add_batch(
#         #     predictions=predicted, 
#         #     references=masks
#         # )
#         # if batch_nb % self.metrics_interval == 0:
#         start=time.time()

#         metrics = self.train_mean_iou.compute(
#             predictions=self.save_predict,
#             references=self.save_label,
#             num_labels=self.num_classes, 
#             ignore_index=0, 
#             reduce_labels=False
#         )
#         print(time.time()-start)
#         print(metrics)
#         print(sys.getsizeof(self.save_predict))