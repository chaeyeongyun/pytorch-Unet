from utils import match_pred_n_mask, mask_labeling_n_flatten
import torch
import torch.nn as nn


def accuracy_per_pixel(pred, target):
    '''
    pred and target is 2-dimensional flatten Tensor: shape (N, number of featuremap's pixels)
    both tensor have 0.0 or 1.0 value
    '''
    accuracy = torch.sum(pred == target) / (pred.shape[1] * pred.shape[0])
    return accuracy

def evaluate(model, valloader, device):
    model.eval()
    val_acc_pixel = 0
    val_loss = 0
    for x_batch, y_batch in valloader:
        x_batch = x_batch.to(device)
        pred = model(x_batch)
        # mask labeling and flatten
        y_batch = match_pred_n_mask(pred, y_batch)
        y_batch = mask_labeling_n_flatten(y_batch)
        
        y_batch = y_batch.to(device)
        sigmoid = nn.Sigmoid()
        pred = sigmoid(pred)
        pred = pred.type(torch.float64)
        # prediction flatten
        pred = torch.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2]*pred.shape[3]))
        loss = nn.BCELoss()
        loss_output = loss(pred, y_batch).item()
        accuracy_per_px = accuracy_per_pixel(torch.round(pred), y_batch)
        # val loss / val accuracy
        val_acc_pixel += accuracy_per_px
        val_loss += loss_output
    
    val_acc_pixel = val_acc_pixel/ len(valloader)
    val_loss = val_loss / len(valloader)

    return val_loss, val_acc_pixel
        