import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from .utils import label_to_onehot

def focal_loss(pred:Tensor, target:Tensor, alpha, gamma, num_classes, ignore_idx=None, reduction="sum"):
    assert pred.shape[0] == target.shape[0],\
        "pred tensor and target tensor must have same batch size"
           
    if num_classes == 1:
        pred = F.sigmoid(pred)
    
    else:
        pred = F.softmax(pred, dim=1).float()

    onehot = label_to_onehot(target, num_classes, ignore_idx)
    onehot = onehot.to(pred.device)
    focal_loss = 0

    focal = torch.pow((1-pred), gamma) # (B, C, H, W)
    ce = -torch.log(pred) # (B, C, H, W)
    focal_loss = alpha * focal * ce * onehot
    focal_loss = torch.sum(focal_loss, dim=1) # (B, H, W)
    
    if reduction == 'none':
        # loss : (B, H, W)
        loss = focal_loss
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(focal_loss)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(focal_loss)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    
    return loss
    

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, ignore_idx=None, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        
        if self.num_classes == 1:
            pred = F.sigmoid(pred)
        else:
            pred = F.softmax(pred, dim=1).float()
        
        return focal_loss(pred, target, self.alpha, self.gamma, self.num_classes, self.ignore_idx, self.reduction)