from utils import mask_labeling
import numpy as np
import torch
import torch.nn as nn


def accuracy_per_pixel(pred, target, ignore_idx):
    '''
    pred: (N, C, H, W)
    target : (N, H, W)
    '''
    pred = pred.argmax(axis=1) # (N, H, W)
    accuracy = np.sum(pred == target) / (target.shape[0] * target.shape[1] * target.shape[2] )
    return accuracy

def evaluate(model, valloader, device, num_classes, ignore_idx):
    model.eval()
    val_acc_pixel = 0
    val_loss = 0
    for x_batch, y_batch in valloader:
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
            
        # val loss / val accuracy
        val_acc_pixel += accuracy_px
        val_loss += loss_output
    
    val_acc_pixel = val_acc_pixel/ len(valloader)
    val_loss = val_loss / len(valloader)

    return val_loss, val_acc_pixel
        