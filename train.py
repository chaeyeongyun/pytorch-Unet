import os
import logging
import argparse
import time
from model import Unet
from dataloader import CustomImageDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter

def match_pred_n_mask(pred_feature, mask):
    diffY = mask.shape[2] - pred_feature.shape[2] 
    diffX = mask.shape[2] - pred_feature.shape[3]
    return mask[:, :, diffY//2:diffY//2+pred_feature.shape[2], diffX//2:+diffX//2+pred_feature.shape[3]]

def train(opt):
    # parameters
    num_epochs, batch_size, lr = opt.num_epochs, opt.batch_size, opt.init_lr
    input_size = opt.input_size
    start_epoch, load_model, dataset_path, save_txt = opt.start_epoch, opt.load_model, opt.dataset_path, opt.save_txt
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet().to(device)

    # data load
    training_data = CustomImageDataset(os.path.join(dataset_path, 'train_images'), os.path.join(dataset_path, 'train_masks'), resize=input_size)
    val_data = CustomImageDataset(os.path.join(dataset_path, 'val_images'), os.path.join(dataset_path, 'val_masks'), resize=input_size)
    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_data, batch_size=1, shuffle=True)

    trainloader_length = len(trainloader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    # start training
    logging.info(f'''Starting training:
        Epochs:          {num_epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(training_data)}
        Validation size: {len(val_data)}
        Device:          {device.type}
    ''')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_acc = 0
        train_loss = 0

        for x_batch, y_batch in trainloader:
            
            # mask labeling
            label_pixels = np.unique(y_batch)
            label_pixels = sorted(label_pixels)
            for i, px in enumerate(label_pixels):
                y_batch = np.where(y_batch==px, i, y_batch)
                if epoch==0:
                    print('pixel:', px, '-->', 'label:', i)
            concat_list = []
            for i in range(len(label_pixels)):
                temp = (y_batch == i)
                concat_list += [temp]
            y_batch = np.concatenate(concat_list, axis=1)
            y_batch = y_batch.astype(float)
            # mask flatten
            y_batch = np.reshape(y_batch, (y_batch.shape[0], y_batch.shape[1]*y_batch.shape[2]*y_batch.shape[3]))
            y_batch = torch.from_numpy(y_batch)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # initialize optimizer
            optimizer.zero_grad()
            # prediction
            pred = model(x_batch)
            pred = F.sigmoid(pred)
            # prediction flatten
            pred = torch.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2]*pred.shape[3]))
            loss = nn.BCELoss()
            loss_output = loss(pred, y_batch)
            loss_output.backward()
            #accuracy calculate
            
            # train loss / train accuracy
            train_acc += 0
            train_loss += loss_output.item()


        # lr_scheduler.step(test_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=25, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--dataset_path', type=str, default='../cropweed/rice_s_n_w', help='dataset directory path')
    parser.add_argument('--input_size', type=int, default=512, help='input image size')
    parser.add_argument('--start_epoch', type=int, default=0, help='the start number of epochs')
    parser.add_argument('--load_model',default=None, type=str, help='the name of saved model file (.pt)')
    parser.add_argument('--save_txt', type=bool, default=True, help='if it''s true, the result of trainig will be saved as txt file.')
    
    opt = parser.parse_args()
    train(opt)