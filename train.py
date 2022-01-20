import os
import logging
import argparse
from model import Unet
from dataloader import CustomImageDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

def train(opt):
    # parameters
    num_epochs, batch_size, lr = opt.num_epochs, opt.batch_size, opt.init_lr
    start_epoch, load_model, dataset_path, save_txt = opt.start_epoch, opt.load_model, opt.dataset_path, opt.save_txt
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet().to(device)

    # data load
    training_data = CustomImageDataset(os.path.join(dataset_path, 'train_images'), os.path.join(dataset_path, 'train_masks'))
    val_data = CustomImageDataset(os.path.join(dataset_path, 'val_images'), os.path.join(dataset_path, 'val_masks'))
    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_data, batch_size=1, shuffle=True)

    trainloader_length = len(trainloader)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
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


        for x_batch, y_batch in trainloader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = nn.CrossEntropyLoss(weight=, )

        lr_scheduler.step(test_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=25, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--init_lr', type=int, default=8, help='initial learning rate')
    parser.add_argument('--dataset_path', type=str, default='../cropweed/rice_s_n_w', help='dataset directory path')
    parser.add_argument('--start_epoch', type=int, default=0, help='the start number of epochs')
    parser.add_argument('--load_model',default=None, type=str, help='the name of saved model file (.pt)')
    parser.add_argument('--save_txt', type=bool, default=True, help='if it''s true, the result of trainig will be saved as txt file.')
    
    opt = parser.parse_args()
    train(opt)