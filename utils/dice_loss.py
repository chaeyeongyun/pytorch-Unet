import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dice_coefficient(pred:Tensor, target:Tensor, num_classes:int, ignore_idx=None):
    assert pred.shape[0] == target.shape[0]
    epsilon = 1e-6
    if num_classes == 2:
        dice = 0
        # if both a and b are 1-D arrays, it is inner product of vectors(without complex conjugation)
        for batch in range(pred.shape[0]):
            pred_1d = pred[batch].view(-1)
            target_1d = target[batch].view(-1)
            inter = (pred_1d * target_1d).sum()
            sum_sets = pred_1d.sum() + target_1d.sum()
            dice += (2*inter+epsilon) / (sum_sets + epsilon)
        return dice / pred.shape[0]
        
    
    elif num_classes == 1:
        dice = 0
        pred = F.Sigmoid(pred)
        for batch in range(pred.shape[0]):
            pred_1d = pred[batch].view(-1)
            target_1d = target[batch].view(-1)
            inter = (pred_1d * target_1d).sum()
            sum_sets = pred_1d.sum() + target_1d.sum()
            dice += (2*inter+epsilon) / (sum_sets + epsilon)
        return dice / pred.shape[0]
        
    else:
        pred = F.softmax(pred, dim=1).float()
        dice = 0
        for c in range(num_classes):
            if c==ignore_idx:
                continue
            dice += dice_coefficient(pred[:, c, :, :], torch.where(target==c, 1, 0), 2, ignore_idx)
        return dice / num_classes 

def dice_loss(pred, target, num_classes, ignore_idx=None):
    if not isinstance(pred, torch.Tensor) :
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(pred)}")

    dice = dice_coefficient(pred, target, num_classes, ignore_idx)
    return 1 - dice

class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_idx=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
    
    def forward(self, pred, target):
        return dice_loss(pred, target, self.num_classes, self.ignore_idx)
