from utils import mask_labeling
import numpy as np
import torch
import torch.nn as nn


def miou(pred, target, num_classes, ignore_idx):
    '''
    pred: (N, C, H, W), ndarray
    target : (N, H, W), ndarray
    '''
    pred = pred.argmax(axis=1) # (N, H, W)
    assert pred.shape[0] == target.shape[0], \
        "pred and target's batch size (shape[0]) must have same value "
    
    batchsize = pred.shape[0]
    # reshape to (N, HxW)
    pred_1d, target_1d = np.reshape(pred, (batchsize, pred.shape[1]*pred.shape[2])), np.reshape(target, (batchsize, target.shape[1]*target.shape[2]))
    
    cats_cnt = []
    for i in range(batchsize):
        cats = target_1d[i] * num_classes + pred_1d[i]
        
        cats_cnt += [np.bincount(cats)]
        if i>0:
            if cats_cnt[i-1].shape != cats_cnt[i].shape:
                '''
                when calculating miou, the number of categories has to be num_classes ^ 2 but sometimes 
                '''
                return -1
    cats_cnt = np.array(cats_cnt) # (N, num_classes^2)
    if cats_cnt.shape[1] != (num_classes ** 2): return -1
    
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
        # print('iou_list:', iou_list, '\tmiou_per_image:', miou_per_image)
    
    miou = sum(miou_per_image) / len(miou_per_image)
    return miou
    

def accuracy_per_pixel(pred, target, ignore_idx):
    '''
    pred: (N, C, H, W), ndarray
    target : (N, H, W), ndarray
    '''
    pred = pred.argmax(axis=1) # (N, H, W)
    # if ignore_idx is not None:
    #     pred = np.where(pred==ignore_idx, -1, pred) # ignore_idx -> -1
    
    accuracy = np.sum(pred == target) / (target.shape[0] * target.shape[1] * target.shape[2] )
    return accuracy

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
    miou = miou(pred, gt, 3, 0)
    print(miou)
    
