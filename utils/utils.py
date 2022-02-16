import numpy as np
import torch

# def match_pred_n_mask(pred_feature, mask):
#     diffY = mask.shape[2] - pred_feature.shape[2] 
#     diffX = mask.shape[3] - pred_feature.shape[3]
#     return mask[:, :, diffY//2:diffY//2+pred_feature.shape[2], diffX//2:+diffX//2+pred_feature.shape[3]]

# def mask_labeling_n_flatten(y_batch, num_classes):
#     label_pixels = np.unique(y_batch)
#     label_pixels = sorted(label_pixels)
#     if len(label_pixels) != num_classes:
#         print('label pixels error')
#         label_pixels = np.array([0, 128, 255])
    
#     for i, px in enumerate(label_pixels):
#         y_batch = np.where(y_batch==px, i, y_batch)
    
#     concat_list = []
#     for i in range(len(label_pixels)):
#         temp = (y_batch == i)
#         concat_list += [temp]
        
#     y_batch = np.concatenate(concat_list, axis=1)
#     y_batch = y_batch.astype(np.float64)
#     # mask flatten
#     y_batch = np.reshape(y_batch, (y_batch.shape[0], y_batch.shape[1]*y_batch.shape[2]*y_batch.shape[3]))
#     y_batch = torch.from_numpy(y_batch)
#     return y_batch

# def batch_one_hot(y_batch, num_classes):
#     label_pixels = np.unique(y_batch)
#     label_pixels = sorted(label_pixels)
    
#     if len(label_pixels) != num_classes:
#         print('label pixels error')
#         label_pixels = np.array([0, 128, 255])
    
#     for i, px in enumerate(label_pixels):
#         y_batch = np.where(y_batch==px, i, y_batch)
    
#     y_batch = np.reshape(y_batch, (y_batch.shape[0], 1, y_batch.shape[1], y_batch.shape[2]))
#     concat_list = []
#     for i in range(len(label_pixels)):
#         temp = (y_batch == i)
#         concat_list += [temp]
        
#     y_batch = np.concatenate(concat_list, axis=1)
#     y_batch = y_batch.astype(np.float64)
    
#     # mask flatten
#     y_batch = torch.from_numpy(y_batch)
#     return y_batch

def mask_labeling(y_batch, num_classes):
    label_pixels = np.unique(y_batch)
    label_pixels = sorted(label_pixels)
    if len(label_pixels) != num_classes:
        print('label pixels error')
        label_pixels = np.array([0, 128, 255])
    
    for i, px in enumerate(label_pixels):
        y_batch = np.where(y_batch==px, i, y_batch)
    
    y_batch = torch.from_numpy(y_batch)
    
    return y_batch

def label_to_onehot(y_batch:torch.Tensor, num_classes, ignore_idx=None):
    concat_list = []
    for c in range(num_classes):
        if c==ignore_idx:
            continue
        concat_list += [np.expand_dims(np.where(y_batch==c, 1, 0), axis=1)]
    onehot = np.concatenate(concat_list, axis=1)
    # return Tensor
    onehot = torch.from_numpy(onehot)
    return onehot

if __name__ == '__main__':
    y_batch = torch.from_numpy(np.array([[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[2, 1, 0],[2, 1, 0],[2, 1, 0]]]))
    onehot = label_to_onehot(y_batch, 3)
    print(onehot)
    