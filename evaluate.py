from utils import mask_labeling
import numpy as np
import torch
import torch.nn as nn


def miou(pred, target, num_classes, ignore_idx):
    '''
    pred: (N, C, H, W), ndarray
    target : (N, H, W), ndarray
    '''
    def bincount_fn(cats, num_classes):
        ''' when the number of category(cats)'s bins is not equal to num_classes^2 '''
        bincount = np.zeros((num_classes**2,))
        l = list(cats)
        for i in range(num_classes**2):
            bincount[i] = l.count(i)
        return bincount
        
    pred = pred.argmax(axis=1) # (N, H, W)
    assert pred.shape[0] == target.shape[0], \
        "pred and target's batch size (shape[0]) must have same value "
    
    batchsize = pred.shape[0]
    # reshape to (N, HxW)
    pred_1d, target_1d = np.reshape(pred, (batchsize, pred.shape[1]*pred.shape[2])), np.reshape(target, (batchsize, target.shape[1]*target.shape[2]))
    
    cats_cnt = []
    for i in range(batchsize):
        # target_1d[i] : (HxW, ), pred_1d[i] : (HxW, )
        cats = target_1d[i] * num_classes + pred_1d[i]
        
        bincount = np.bincount(cats)
        cats_cnt += [bincount if bincount.shape[0] == num_classes**2 else bincount_fn(cats, num_classes)]
        
    cats_cnt = np.array(cats_cnt) # (N, num_classes^2)
    
    # confusion matrix
    conf_mat = np.reshape(cats_cnt, (batchsize, num_classes, num_classes)) 
    
    miou_per_image = []
    for i in range(batchsize):
        iou_list = []
        sum_row = np.sum(conf_mat[i], 0)
        sum_col = np.sum(conf_mat[i], 1)
        for j in range(num_classes):
            if j==ignore_idx:
                continue
            iou_list += [conf_mat[i][j][j] / (sum_col[j]+sum_row[j]-conf_mat[i][j][j])]
        
        miou_per_image += [sum(iou_list)/len(iou_list)]
    
    miou = sum(miou_per_image) / len(miou_per_image)
    return miou
    

def accuracy_per_pixel(pred, target, ignore_idx):
    '''
    pred: (N, C, H, W), ndarray
    target : (N, H, W), ndarray
    '''
    batchsize = pred.shape[0]
    pred = pred.argmax(axis=1) # (N, H, W)
    pred = np.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2])) # (N, HxW)
    target = np.reshape(target, (target.shape[0], target.shape[1]*target.shape[2])) # (N, HxW)
    
    accuracy_per_image = []
    for b in range(batchsize):
        if ignore_idx is not None:
            not_ignore_idxs = np.where(target[b]!=ignore_idx) # where target is not equal to ignore_idx
            pred_temp = pred[b][not_ignore_idxs] # 1dim (HxW, )
            target_temp = target[b][not_ignore_idxs] # 1dim (HxW, )
            accuracy_per_image += [np.sum(pred_temp == target_temp) / target_temp.shape[0]]
        else : 
            accuracy_per_image += [np.sum(pred[b] == target[b]) / target[b].shape[0]]
    
    acc = sum(accuracy_per_image) / len(accuracy_per_image)
    return acc

def evaluate(model, valloader, device, num_classes, ignore_idx):
    model.eval()
    val_acc_pixel = 0
    val_loss = 0
    val_miou = 0
    iter = 0
    for x_batch, y_batch in valloader:
        iter += 1
        x_batch = x_batch.to(device, dtype=torch.float32)
        pred = model(x_batch)
        # mask labeling and flatten
        y_batch = mask_labeling(y_batch, num_classes)
        y_batch = y_batch.to(device, dtype=torch.long)
        
        loss = nn.CrossEntropyLoss()
        loss_output = loss(pred, y_batch).item()
        
        copied_pred = pred.data.cpu().numpy() 
        copied_y_batch = y_batch.data.cpu().numpy() 
        accuracy_px = accuracy_per_pixel(copied_pred, copied_y_batch, ignore_idx)
        miou_per_batch = miou(copied_pred, copied_y_batch, num_classes, ignore_idx) 
        # val loss / val accuracy
        val_acc_pixel += accuracy_px
        val_loss += loss_output
        if miou_per_batch == -1:
            val_miou += val_miou / iter

        else : val_miou += miou_per_batch
        
    
    val_acc_pixel = val_acc_pixel/ len(valloader)
    val_loss = val_loss / len(valloader)
    val_miou = val_miou / len(valloader)

    return val_loss, val_acc_pixel, val_miou
        
if __name__ == '__main__':
    np.random.seed(0)
    pred = np.random.randint(low=0, high=3, size=(2, 3, 24, 24))
    gt = np.random.randint(low=0, high=3, size=(2, 24, 24))
    miou = miou(pred, gt, 3, None)
    # accuracy_per_pixel(pred, gt, None)
    # print(miou)
    
