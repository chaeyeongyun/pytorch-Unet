from utils.utils import mask_labeling
import numpy as np
import torch
import torch.nn as nn
from utils.utils import onehot_ndarray
from utils.dice_loss import DiceLoss
from utils.focal_loss import FocalLoss

def bincount_fn(cats, num_classes):
    '''
    this function is used to make confusion matrix for calculating miou(and iou)
    when the number of category(cats)'s bins is not equal to num_classes^2 
    '''
    bincount = np.zeros((num_classes**2,))
    l = list(cats)
    for i in range(num_classes**2):
        bincount[i] = l.count(i)
    return bincount

def miou(pred, target, num_classes, ignore_idx=None):
    '''
    calculate miou
    
    Args:
        pred: (N, C, H, W), ndarray
        target : (N, H, W), ndarray

    Return:
        miou(float), iou_per_class(ndarray)
    '''
       
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
    
    iou_per_class = np.array([0]*num_classes, dtype=np.float64)
    miou_per_image = []
    for i in range(batchsize):
        iou_list = []
        sum_row = np.sum(conf_mat[i], 0)
        sum_col = np.sum(conf_mat[i], 1)
        for j in range(num_classes):
            if j==ignore_idx:
                continue
            iou_list += [conf_mat[i][j][j] / (sum_col[j]+sum_row[j]-conf_mat[i][j][j])]
        iou_per_class += np.array(iou_list, dtype=np.float64)
        miou_per_image += [sum(iou_list)/len(iou_list)]
    
    iou_per_class = iou_per_class / batchsize 
    miou = sum(miou_per_image) / len(miou_per_image)
    return miou, iou_per_class
    
    
    
def confusion_matrix(pred, target, num_classes, ignore_idx=None):
    '''
    make confusion matrix for miou_modified in train.py
    
    modified miou in CED-Net
    https://www.mdpi.com/2079-9292/9/10/1602/htm
    Args:
        pred: (N, C, H, W), ndarray
        target : (N, H, W), ndarray

    Return:
        conf_mats : (num_classes, 3), ndarray, conf_mats[class] : (3, ) -> [TN, FN+FP, TP]
    '''
    def catcount(cats):
        catcount = np.zeros(4)
        l = list(cats)
        for i in range(4):
            catcount[i] = l.count(i)
        return catcount
    
    pred = pred.argmax(axis=1) # (N, H, W)
    assert pred.shape[0] == target.shape[0], \
        "pred and target's batch size (shape[0]) must have same value "
    
    batchsize = pred.shape[0]
    pred = onehot_ndarray(pred, num_classes) # (N, C, H, W)
    target = onehot_ndarray(target, num_classes) # (N, C, H, W)
    
    # reshape to (N, C, HxW)
    pred_1d, target_1d = np.reshape(pred, (batchsize, pred.shape[1], pred.shape[2]*pred.shape[3])), \
        np.reshape(target, (batchsize, target.shape[1], target.shape[2]*target.shape[3]))
    # reshape to (C, NxHxW)
    pred_1d = np.reshape(np.swapaxes(pred_1d, 0, 1), (num_classes, batchsize*pred_1d.shape[-1]))
    target_1d = np.reshape(np.swapaxes(target_1d, 0, 1), (num_classes, batchsize*target_1d.shape[-1]))
    
    cats = (pred_1d + target_1d)+1 # 3: TP, 2: FN and FP, 1:TN ( iou = TP/(TP+FN+FP) )
    cats = np.where((cats==2.0)&(target_1d==1.0), 0, cats) # 3:TP, 2:FP,  1:TN, 0:FN
    # confusion matrixes per class
    # conf_mats = np.zeros((num_classes,3), dtype=np.int64)
    conf_mats = []
    for i in range(num_classes):
        cat = cats[i, :]
        cat_bincount = np.bincount(cat) if np.bincount(cat).shape[0] == 4 else catcount(cat) # [FN, TN, FP, TP]
        conf_mats += [cat_bincount]
    
    return np.array(conf_mats) 
        
def conf_to_miou(conf_sum):
    '''
    convert confusion matrix to CED-Net miou
    '''
    m_ious = conf_sum[:, 3] / (conf_sum[:, 0]+conf_sum[:, 2]+conf_sum[:, 3]) # [class 0 iou, class 1 iou, class 2 iou, ...]
    m_miou = np.sum(conf_sum[1:, :], axis=0) # calculate except background ( class 0 )
    m_miou = m_miou[3] / (m_miou[0] + m_miou[2] + m_miou[3])
    return m_ious, m_miou    
        
def f1_score(conf_sum, ignore_idx=None):
    '''
    F1 score has a high value when the recall and precision are not biased to either side.
    F1 score = 2 / (1/recall + 1/precision) = 2 x (recall x precision) / (recall + precision)
    
    Args:
        conf_sum : ndarray, (num_classes, 4) - sum of confusion matrix per epoch, 4:[FN, TN, FP, TP]
    Returns:
        f1_score : F1 score(int)
    '''
    sum = np.sum(conf_sum[1:, :], axis=0)
    precision = sum[3] / (sum[2]+sum[3]) # TP/(FP+TP) : The actual P ratio among the total predicted P.
    recall = sum[3] / (sum[3]+sum[0])# TP/(TP+FN) : The ratio predicted by P among actual P.
    f1_score = 2 * (recall * precision) / (recall + precision)
    return f1_score
    
def accuracy_per_pixel(pred, target, ignore_idx=None):
    '''
    Args:
        pred: (N, C, H, W), ndarray
        target : (N, H, W), ndarray
    Returns:
        the accuracy per pixel : acc(int)
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


    


def evaluate(model, valloader, device, num_classes, loss_fn, ignore_idx=None):
    model.eval()
    val_acc_pixel = 0
    val_loss = 0
    val_miou = 0
    iou_per_class = np.array([0]*num_classes, dtype=np.float64)
    conf_sum = np.zeros((num_classes, 4))
    
    iter = 0
    for x_batch, y_batch in valloader:
        iter += 1
        x_batch = x_batch.to(device, dtype=torch.float32)
        pred = model(x_batch)
        # mask labeling and flatten
        y_batch = mask_labeling(y_batch, num_classes)
        y_batch = y_batch.to(device, dtype=torch.long)
        
        if loss_fn == 'ce':
            loss = nn.CrossEntropyLoss()
            
        elif loss_fn == 'dice':
            loss = DiceLoss(num_classes, ignore_idx)
        
        elif loss_fn == 'focal':
            loss = FocalLoss(num_classes)
        
        loss_output = loss(pred, y_batch).item()
        
        copied_pred = pred.data.cpu().numpy() 
        copied_y_batch = y_batch.data.cpu().numpy() 
        accuracy_px = accuracy_per_pixel(copied_pred, copied_y_batch, ignore_idx)
        miou_per_batch, iou_ndarray = miou(copied_pred, copied_y_batch, num_classes, ignore_idx) 
        # modified iou
        conf = confusion_matrix(copied_pred, copied_y_batch, num_classes, ignore_idx)
        
        # val loss / val accuracy
        val_acc_pixel += accuracy_px
        val_loss += loss_output
        val_miou += miou_per_batch
        iou_per_class += iou_ndarray
        conf_sum += conf
        
    
    val_acc_pixel = val_acc_pixel/ len(valloader)
    val_loss = val_loss / len(valloader)
    val_miou = val_miou / len(valloader)
    val_ious = np.round((iou_per_class / len(valloader)), 5).tolist()
    ## modified iou
    val_m_ious, val_m_miou = conf_to_miou(conf_sum)
    
    return val_loss, val_acc_pixel, val_miou, val_ious, val_m_miou, val_m_ious
        
if __name__ == '__main__':
    np.random.seed(0)
    pred = np.random.randint(low=0, high=3, size=(2, 3, 2, 2))
    gt = np.random.randint(low=0, high=3, size=(2, 2, 2))
     
    confmats =  confusion_matrix(pred, gt, 3)
    print(confmats)
    # print(miou, iou_list)
    # gt = np.random.randint(low=0, high=3, size=(2, 24, 24))
    # miou = miou(pred, gt, 3, None)
    # accuracy_per_pixel(pred, gt, None)
    # print(miou)
    # pred = torch.from_numpy(np.array([[[[0.8, 0.7],[0.3, 0.4]], [[0.1, 0.2], [0.5, 0.7]], [[0.5, 0.7],[0.8, 0.9]]], [[[0.8, 0.7],[0.3, 0.4]], [[0.1, 0.2], [0.5, 0.7]], [[0.5, 0.7],[0.8, 0.9]]]]))
    # target = np.array([[[0, 1],[2, 2]], [[0, 1],[2, 2]]])
  