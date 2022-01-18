import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataloader import CustomImageDataset

def train(opt):
    num_epochs, batch_size = opt.num_epochs, opt.batch_size
    start_epoch, load_model, dataset_path, save_txt = opt.start_epoch, opt.load_model, opt.dataset_path, opt.save_txt
    training_data = CustomImageDataset(os.path.join(dataset_path, 'train_images'), os.path.join(dataset_path, 'train_masks'))
    val_data = CustomImageDataset(os.path.join(dataset_path, 'val_images'), os.path.join(dataset_path, 'val_masks'))
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    for epoch in range(start_epoch, num_epochs):

        for x_batch, y_batch in train_dataloader:
            print(x_batch.size())
            print(y_batch.size())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=25, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--dataset_path', type=str, default='../cropweed/rice_s_n_w', help='dataset directory path')
    parser.add_argument('--start_epoch', type=int, default=0, help='the start number of epochs')
    parser.add_argument('--load_model',default=None, type=str, help='the name of saved model file (.pt)')
    parser.add_argument('--save_txt', type=bool, default=True, help='if it''s true, the result of trainig will be saved as txt file.')
    opt = parser.parse_args()
    train(opt)