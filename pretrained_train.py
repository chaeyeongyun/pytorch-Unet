import os
import logging
import argparse
import time
import datetime
from models.model import Unet
from models.pretrained_model import ResNetUnet
from dataloader import CustomImageDataset
from utils import mask_labeling, batch_one_hot
from evaluate import accuracy_per_pixel, evaluate
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def train(opt, model):
    # parameters
    num_epochs, batch_size, lr, num_classes = opt.num_epochs, opt.batch_size, opt.init_lr, opt.num_classes
    input_size, ignore_idx = opt.input_size, opt.ignore_idx
    start_epoch, load_model, dataset_path, save_txt = opt.start_epoch, opt.load_model, opt.dataset_path, opt.save_txt
    save_checkpoint, checkpoint_dir, pretrained = opt.save_checkpoint, opt.checkpoint_dir, opt.pretrained
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if pretrained:
        model = ResNetUnet(num_classes=num_classes)
    elif load_model is not None:
        model = Unet(num_classes=num_classes)
        model.load_state_dict(torch.load(load_model))
    else :
        model = Unet(num_classes=num_classes)    

    model.to(device)
    # data load
    training_data = CustomImageDataset(os.path.join(dataset_path, 'train_images'), os.path.join(dataset_path, 'train_masks'), resize=input_size, pretrained=True)
    val_data = CustomImageDataset(os.path.join(dataset_path, 'val_images'), os.path.join(dataset_path, 'val_masks'), resize=input_size, pretrained=True)
    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_data, batch_size=1)

    trainloader_length = len(trainloader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # for tensorboard
    writer = SummaryWriter()
    
    dt = datetime.datetime.now()
    if save_checkpoint:
        save_model_path = os.path.join(checkpoint_dir, f"{dt.month}-{dt.day}-{dt.hour}-{dt.minute}")
        os.makedirs(save_model_path)
    start = time.time()
    if save_txt:
        f = open(os.path.join(save_model_path, 'result.txt'),'w')
    
    best_val_acc = 0
    start = time.time()
    # start training
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_acc_pixel = 0
        train_loss = 0

        iter = 0
        for x_batch, y_batch in trainloader:
            iter +=1
            msg = '\riteration  %d / %d'%(iter, trainloader_length)
            print(' '*len(msg), end='')
            print(msg, end='')
            time.sleep(0.1)
            
            x_batch = x_batch.to(device, dtype=torch.float32)
            # initialize optimizer
            optimizer.zero_grad()
            # prediction
            pred = model(x_batch) # (N, C(n_classes), H, W)
            
            y_batch = mask_labeling(y_batch, num_classes) # (N, H, W) and has [0, numclass -1] value
            # y_batch = torch.reshape(y_batch, (y_batch.shape[0], -1)) # (N, 1xHxW)
            # y_batch = batch_one_hot(y_batch, num_classes)
            y_batch = y_batch.to(device, dtype=torch.long)
            
            # pred = torch.reshape(pred, (pred.shape[0], pred.shape[1], -1))# (N, C, HxW)
            # pred = pred.type(torch.float64)
            # pred = pred.softmax(dim=1)

            loss = nn.CrossEntropyLoss()
            # pred_ = torch.reshape(pred, [pred.shape[0]*pred.shape[2]*pred.shape[3], pred.shape[1]])
            # lab_ = torch.reshape(onehot_y, [onehot_y.shape[0]*onehot_y.shape[2]*onehot_y.shape[3], onehot_y.shape[1]])
            loss_output = loss(pred, y_batch)
            # loss_output = torch.mean(lab_*torch.log(pred_ + 1e-7), -1)
            loss_output.backward()
            
            #accuracy calculate
            copied_pred = pred.data.cpu().numpy() 
            copied_y_batch = y_batch.data.cpu().numpy() 
            accuracy_px = accuracy_per_pixel(copied_pred, copied_y_batch, ignore_idx)
            
            # train loss / train accuracy
            train_acc_pixel += accuracy_px
            train_loss += loss_output.item()
        
        # evaluate after 1 epoch training
        val_loss, val_acc_pixel = evaluate(model, valloader, device, num_classes, ignore_idx)
        torch.cuda.empty_cache()
        
        if save_checkpoint:
            if val_acc_pixel > best_val_acc:
                torch.save(model.state_dict(), os.path.join(save_model_path, f'{epoch}_{num_epochs}ep_{batch_size}b_best.pt'))
                best_val_acc = val_acc_pixel
                
        train_acc_pixel = train_acc_pixel / trainloader_length
        train_loss = train_loss / trainloader_length
        lr_scheduler.step(val_loss)
        writer.add_scalars('Loss', {'trainloss':train_loss, 'valloss':val_loss}, epoch)
        writer.add_scalars('Accuracy', {'trainacc':train_acc_pixel, 'valacc':val_acc_pixel}, epoch)
        
         # Metrics calculation
        result_txt = "\nEpoch: %d, loss: %.8f, Train accuracy: %.8f, Test accuracy: %.8f, Test loss: %.8f, lr: %5f" % (epoch+1, train_loss, train_acc_pixel, val_acc_pixel, val_loss, optimizer.param_groups[0]['lr'])
        if save_txt:
            f.write(result_txt)
        print(result_txt)
    
    finish = time.time()
    print("---------training finish---------")
    total_result_txt = "\nTotal time: %d(sec), Total Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f, Test loss: %.5f" % (finish-start, num_epochs, train_loss, train_acc_pixel, val_acc_pixel, val_loss)
    # print("\nTotal time: %d(sec), Total Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f, Test loss: %.5f" % (finish-start, num_epochs, train_loss, train_acc, test_acc, test_loss))
    print(total_result_txt)
    if save_txt:
        f.write(total_result_txt)
        f.close()
    torch.save(model.state_dict(), os.path.join(save_model_path, f'{num_epochs}ep_{batch_size}b_final.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=bool, default=True, help='if it''s true, pretrained weights will loaded from torch hub')
    parser.add_argument('--num_epochs', type=int, default=25, help='the number of epochs')
    parser.add_argument('--num_classes', type=int, default=3, help='the number of classes')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--init_lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--ignore_idx', type=int, default=None, help='ignore index i.e. background class')
    parser.add_argument('--dataset_path', type=str, default='../cropweed/IJRR2017', help='dataset directory path')
    parser.add_argument('--input_size', type=int, default=512, help='input image size')
    parser.add_argument('--start_epoch', type=int, default=0, help='the start number of epochs')
    parser.add_argument('--load_model',default=None, type=str, help='the name of saved model file (.pt)')
    parser.add_argument('--save_txt', type=bool, default=False, help='if it''s true, the result of trainig will be saved as txt file.')
    parser.add_argument('--save_checkpoint', type=bool, default=False, help='if it''s true, the model will saved at checkpoint dir')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='path to save checkpoint file')
    
    opt = parser.parse_args()
    # try:
    #     model = None
    #     train(opt, model)
    # except:
    #      print("---------!!!training interrupt!!!---------")
    #      if opt.save_checkpoint and model is not None:
    #          torch.save(model.state_dict(), os.path.join(opt.save_model_path, f'interrupt.pt'))
    model = None
    train(opt, model)