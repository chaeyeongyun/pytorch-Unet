import numpy as np
import torch

def match_pred_n_mask(pred_feature, mask):
    diffY = mask.shape[2] - pred_feature.shape[2] 
    diffX = mask.shape[3] - pred_feature.shape[3]
    return mask[:, :, diffY//2:diffY//2+pred_feature.shape[2], diffX//2:+diffX//2+pred_feature.shape[3]]

def mask_labeling_n_flatten(y_batch, num_classes):
    label_pixels = np.unique(y_batch)
    label_pixels = sorted(label_pixels)
    if len(label_pixels) != num_classes:
        label_pixels = np.array([0, 128, 255])
    
    for i, px in enumerate(label_pixels):
        y_batch = np.where(y_batch==px, i, y_batch)
    
    concat_list = []
    for i in range(len(label_pixels)):
        temp = (y_batch == i)
        concat_list += [temp]
        
    y_batch = np.concatenate(concat_list, axis=1)
    y_batch = y_batch.astype(np.float64)
    # mask flatten
    y_batch = np.reshape(y_batch, (y_batch.shape[0], y_batch.shape[1]*y_batch.shape[2]*y_batch.shape[3]))
    y_batch = torch.from_numpy(y_batch)
    return y_batch
