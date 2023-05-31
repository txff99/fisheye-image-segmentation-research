import torch

def compute_metrics(pr, gt):
    cm={}
    # pr=torch.argmax(pr,dim=-3)

    # pr_temp = torch.zeros(gt.shape).to(pr.device)
    # src=torch.ones(gt.shape).to(pr.device)

    # pr=pr_temp.scatter(dim=1, index=pr.unsqueeze(1), src=src)

    for i in range(1,pr.size()[1]):
        if i not in cm:
            #cm[i]={"TP":0,"FP":0,"TN":0,"FN":0}
            pred=pr[:, i, :, :]
            target=gt[:, i, :, :]
            a,b,c = target.shape
            tmp = torch.zeros(a,b,c)
            tmp = tmp.to(pr.device)
            if torch.equal(target,tmp.long()):
                pass
            else:
                cm[i] = {}

                # print(target)
                cm[i]["TP"] = torch.sum(pred * target).item()
                cm[i]["FP"] = torch.sum(pred * (1 - target)).item()
                cm[i]["FN"] = torch.sum((1 - pred) * target).item()
                cm[i]["TN"] = torch.sum((1 - pred) * (1 - target)).item()
                # else:
                #         for tp_tn_fp_fn in cm[category]:
                #                 cm[category][tp_tn_fp_fn] += cm[category][tp_tn_fp_fn]

    IoUs=[]
    for category in cm:
            TP = cm[category]["TP"]
            TN = cm[category]["TN"]
            FP = cm[category]["FP"]
            FN = cm[category]["FN"]
            IoUs.append( (TP) / (TP + FN + FP))
            # print((TP+0.001) / (TP + FN + FP+0.001))

    ACCs=[]
    for category in cm:
            TP = cm[category]["TP"]
            TN = cm[category]["TN"]
            FP = cm[category]["FP"]
            FN = cm[category]["FN"]
            ACCs.append( (TP+TN) / (TP + FN + FP+TN))
    
    TP=TN=FP=FN=0
    for category in cm:
        TP += cm[category]["TP"]
        TN += cm[category]["TN"]
        FP += cm[category]["FP"]
        FN += cm[category]["FN"]
    
    return {'mean_iou':torch.tensor(sum(IoUs)/len(IoUs)),
        "global_iou":torch.tensor((TP)/(TP+FN+FP)),
        "mean_acc":torch.tensor(sum(ACCs)/len(ACCs))}

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs

def calculate_metrics(pr,gt,eps=1e-7):
    IoUs = {}
    acc= {}
    sum_iou = 0
    sum_acc = 0
    for i in range(0,pr.size()[1]):
        if torch.sum(gt[:, i, :, :]) == 0 :
            # remove the unlabeled channel
            pass
        else: 
            intersection = torch.sum(gt[:,i,:,:] * pr[:,i,:,:])
            union = torch.sum(gt[:,i,:,:]) + torch.sum(pr[:,i,:,:]) - intersection + eps
            IoUs[i] = (intersection + eps) / union
            acc[i] =  (intersection + eps) / (torch.sum(gt[:,i,:,:])+eps)
            sum_iou += IoUs[i]
            sum_acc += acc[i]
            # IoUs.append((intersection + eps) / union)
            # acc.append((intersection + eps) / (torch.sum(gt[:,i,:,:])+eps))
    mIoUs = sum_iou/len(IoUs)
    macc = sum_acc/len(acc)
    
    return {"miou":mIoUs,"macc":macc,'per_iou':IoUs,'per_acc':acc}
