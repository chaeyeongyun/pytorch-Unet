#%%
import os
import logging
import matplotlib.pyplot as plt
import argparse
import time
import datetime
from models.model import Unet
from models.pretrained_model import ResNetUnet
from dataloader import CustomImageDataset
from utils.utils import mask_labeling
from utils.dice_loss import DiceLoss
from utils.focal_loss import FocalLoss
from evaluate import accuracy_per_pixel, evaluate, miou, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def train(opt, model):
    # parameters
    num_epochs, batch_size, lr, num_classes = opt.num_epochs, opt.batch_size, opt.init_lr, opt.num_classes
    input_size, ignore_idx = opt.input_size, opt.ignore_idx
    start_epoch, load_model, dataset_path, save_txt = opt.start_epoch, opt.load_model, opt.dataset_path, opt.save_txt
    save_checkpoint, checkpoint_dir, save_imgs, pretrained = opt.save_checkpoint, opt.checkpoint_dir, opt.save_imgs, opt.pretrained
    loss_fn = opt.loss_fn
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   
    if pretrained:
        print('ResNetUnet')
        model = ResNetUnet(num_classes=num_classes)
    
    elif pretrained and load_model is not None:
        print('ResNetUnet -- load model... --')
        model = ResNetUnet(num_classes=num_classes, init_weights=False)
        model.load_state_dict(torch.load(load_model))
        
    elif not pretrained and load_model is not None:
        print('Unet -- load model... --')
        model = Unet(num_classes=num_classes, init_weights=False)
        model.load_state_dict(torch.load(load_model))
    else :
        print('Unet')
        model = Unet(num_classes=num_classes)    

    model.to(device)
    # data load
    training_data = CustomImageDataset(os.path.join(dataset_path, 'train_images'), os.path.join(dataset_path, 'train_masks'), resize=input_size, pretrained=pretrained)
    val_data = CustomImageDataset(os.path.join(dataset_path, 'val_images'), os.path.join(dataset_path, 'val_masks'), resize=input_size, pretrained=pretrained)
    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(val_data, batch_size=1)

    trainloader_length = len(trainloader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        
    dt = datetime.datetime.now()
    save_model_path = os.path.join(checkpoint_dir, f"{dataset_path.split('/')[-1]}-{model.__class__.__name__}-{num_epochs}-{dt.month}{dt.day}")
    if save_checkpoint:
        os.makedirs(save_model_path)
         # for tensorboard
        writer = SummaryWriter(log_dir=os.path.join(save_model_path, 'runs'))
        
    start = time.time()
    if save_txt:
        f = open(os.path.join(save_model_path, 'result.txt'),'w')
        # file = open(os.path.join(save_model_path,'train information.txt'), 'w')
        information = "dataset path : %s \npretrain %s, batch size %d, loss: %s, num_epochs %d, init_lr %.8f, input size %d, device  %s\n" % (dataset_path, pretrained, batch_size, loss_fn, num_epochs, lr, input_size, device)
        print(information)
        f.write(information)
        # file.close()
    
    if save_imgs:
        img_dir = os.path.join(save_model_path, 'imgs')
        os.makedirs(img_dir)
        
    best_val_miou = 0
    conf_sum = np.zeros((num_classes, 3))
    start = time.time()
    # start training
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_acc_pixel = 0
        train_loss = 0
        train_miou = 0
        iou_per_class = np.array([0]*num_classes, dtype=np.float64)
        
        iter = 0

        for x_batch, y_batch in trainloader:
            iter +=1
            msg = '\riteration  %d / %d'%(iter, trainloader_length)
            print(' '*len(msg), end='')
            print(msg, end='')# %%
            time.sleep(0.1)
            
            x_batch = x_batch.to(device, dtype=torch.float32)
            # initialize optimizer
            optimizer.zero_grad()
            # prediction
            pred = model(x_batch) # (N, C(n_classes), H, W)
            
            y_batch = mask_labeling(y_batch, num_classes) # (N, H, W) and has [0, numclass -1] value
            y_batch = y_batch.to(device, dtype=torch.long)
            
            # if ignore_idx is not None:
            #     loss = nn.CrossEntropyLoss(ignore_index=ignore_idx)
  
            if loss_fn == 'ce':
                # cross entropy loss
                if ignore_idx is not None:
                    loss = nn.CrossEntropyLoss(ignore_idx=ignore_idx)
                
                else: loss = nn.CrossEntropyLoss()
                
            elif loss_fn == 'dice':
                # dice loss
                loss = DiceLoss(num_classes, ignore_idx)
            
            elif loss_fn == 'focal':
                loss = FocalLoss(num_classes)
                
            loss_output = loss(pred, y_batch)  
            loss_output.backward()
            
            # update parameters
            optimizer.step()
            
            if iter == len(trainloader):
                color_map = np.array([[0,0,0],[0,0,255],[255, 0, 0]], np.uint8) # (black, blue, red)
                temp_pred = pred.data.cpu().numpy() # (N, C, H, W)
                temp_pred = np.swapaxes(temp_pred, 1, 3) # (N, H, W, C)
                temp_pred = np.argmax(temp_pred, -1) # (N, H, W)
                show_predict = color_map[temp_pred]     # (N, H, W, 3)
                concat_list = []
                for b in range(batch_size):
                    concat_list += [show_predict[b]]
                plt.imsave(os.path.join(img_dir, '{}ep_predict.png'.format(epoch)), np.concatenate(concat_list, 0))
            
            
            #accuracy calculate
            copied_pred = pred.data.cpu().numpy() 
            copied_y_batch = y_batch.data.cpu().numpy() 
            accuracy_px = accuracy_per_pixel(copied_pred, copied_y_batch, ignore_idx)
            miou_per_batch, iou_ndarray = miou(copied_pred, copied_y_batch, num_classes, ignore_idx)
            conf = confusion_matrix(copied_pred, copied_y_batch, num_classes, ignore_idx)
            
            # train loss / train accuracy
            train_acc_pixel += accuracy_px
            train_loss += loss_output.item()
            train_miou += miou_per_batch
            iou_per_class += iou_ndarray
            conf_sum += conf
            
        # evaluate after 1 epoch training
        val_loss, val_acc_pixel, val_miou, val_ious, val_m_miou, val_m_ious =\
            evaluate(model, valloader, device, num_classes, loss_fn, ignore_idx)
        torch.cuda.empty_cache()
                
        train_acc_pixel = train_acc_pixel / trainloader_length
        train_loss = train_loss / trainloader_length
        train_miou = train_miou / trainloader_length
        train_ious = (np.round(iou_per_class / trainloader_length, 5)).tolist()
        ## modified ious
        train_m_ious = conf_sum[:, 0] / (conf_sum[:, 1]+conf_sum[:, 0])
        train_m_miou = np.mean(train_m_ious)
               
        if save_checkpoint:
            writer.add_scalars('Loss', {'trainloss':train_loss, 'valloss':val_loss}, epoch)
            writer.add_scalars('Accuracy', {'trainacc':train_acc_pixel, 'valacc':val_acc_pixel}, epoch)
            writer.add_scalars('mIOU', {'trainmiou':train_miou, 'valmiou':val_miou}, epoch)
            if val_miou > best_val_miou:
                torch.save(model.state_dict(), os.path.join(save_model_path, f'best.pt'))
                best_val_miou = val_miou
        
        # lr_scheduler.step(val_loss)
        
        
        #  Metrics calculation
        result_txt = "\nEpoch: %d, loss: %.8f, Train accuracy: %.8f, Train miou: %.8f, Val accuracy: %.8f, Val miou: %.8f, Val loss: %.8f, lr: %5f" % (epoch+1, train_loss, train_acc_pixel, train_miou, val_acc_pixel, val_miou, val_loss, optimizer.param_groups[0]['lr'])
        result_txt += f"\n train iou per class = {train_ious}, val iou per class = {val_ious}"
        result_txt += f"\n modified train miou = {train_m_miou[0]}, val iou per class = {val_m_miou[0]}"
        if save_txt:
            f.write(result_txt)
        print(result_txt)
    
    finish = time.time()
    
    print("---------training finish---------")
    
    total_result_txt = "\nTotal time: %d(sec), Total Epoch: %d, loss: %.5f, Train accuracy: %.5f, Train miou : %.5f, Val accuracy: %.5f, Val miou: %.5f, Val loss: %.5f" % (finish-start, num_epochs, train_loss, train_acc_pixel, train_miou, val_acc_pixel, val_miou, val_loss)
    print(total_result_txt)
    
    if save_txt:
        f.write(total_result_txt)
        f.close()
    
    if save_checkpoint:
        torch.save(model.state_dict(), os.path.join(save_model_path, f'{num_epochs}ep_{batch_size}b_final.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=bool, default=False, help='if it''s true, use ResNetUnet model, and pretrained weights will loaded from torch hub')
    parser.add_argument('--num_epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--num_classes', type=int, default=3, help='the number of classes')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--init_lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--loss_fn', type=str, default='focal', help='loss function. ce / dice /focal ')
    parser.add_argument('--ignore_idx', type=int, default=None, help='ignore index i.e. background class')
    parser.add_argument('--dataset_path', type=str, default='../cropweed/rice_s_n_w', help='dataset directory path')
    parser.add_argument('--input_size', type=int, default=512, help='input image size')
    parser.add_argument('--start_epoch', type=int, default=0, help='the start number of epochs')
    parser.add_argument('--load_model',default=None, type=str, help='the name of saved model file (.pt)')
    parser.add_argument('--save_txt', type=bool, default=True, help='if it''s true, the result of trainig will be saved as txt file.')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='if it''s true, the model will saved at checkpoint dir')
    parser.add_argument('--save_imgs', type=bool, default=True, help='if it''s true, the predict images will saved at image dir')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='path to save checkpoint file')
    
    opt = parser.parse_args()
    # try:
    #     model = None
    #     train(opt, model)
    # except:
    #      print("---------!!!training interrupt!!!---------")
    #      if opt.save_checkpoint and model is not None:
    #          torch.save(model.state_dict(), os.path.join(opt.save_model_path, f'interrupt.pt'))
    # model = None
    # train(opt, model)
    # datasets = '../cropweed'
    datasets = '../blured_cropweed_strong'
    for dataset in os.listdir(datasets):
        if 'IJRR' in dataset : continue
        opt.dataset_path = os.path.join(datasets, dataset)
        model=None
        train(opt, model)

