from utils import mask_labeling
import torch
import torch.nn as nn


def accuracy_per_pixel(pred, target):
    '''
    pred: (N, C, HxW)
    target : (N, HxW)
    '''
    pred = pred.argmax(dim=1) # (N, HxW)
    accuracy = torch.sum(pred == target) / (pred.shape[1] * pred.shape[0])
    return accuracy

def evaluate(model, valloader, device, num_classes):
    model.eval()
    val_acc_pixel = 0
    val_loss = 0
    for x_batch, y_batch in valloader:
        x_batch = x_batch.to(device)
        pred = model(x_batch)
        # mask labeling and flatten
        y_batch = mask_labeling(y_batch, num_classes)
        y_batch = torch.reshape(y_batch, (y_batch.shape[0], -1))
        y_batch = y_batch.type(torch.long)
        y_batch = y_batch.to(device)
        
        pred = torch.reshape(pred, (pred.shape[0], pred.shape[1], -1))# (N, C, HxW)
        pred = pred.type(torch.float64)
        pred = pred.softmax(dim=1)
        # prediction flatten
       
        loss = nn.CrossEntropyLoss(ignore_index=0)
        loss_output = loss(pred, y_batch).item()
        accuracy_per_px = accuracy_per_pixel(pred, y_batch)
        # val loss / val accuracy
        val_acc_pixel += accuracy_per_px
        val_loss += loss_output
    
    val_acc_pixel = val_acc_pixel/ len(valloader)
    val_loss = val_loss / len(valloader)

    return val_loss, val_acc_pixel
        