import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from evaluate import accuracy_per_pixel, miou
from utils import mask_labeling
from dataloader import CustomImageDataset
from models.model import Unet
from models.pretrained_model import ResNetUnet
from torch.utils.data import DataLoader

def test(opt):
    pretrained, num_classes, ignore_idx, save_path, dataset_path, input_size, load_model, save_txt, save_imgs =\
        opt.pretrained, opt.num_classes, opt.ignore_idx, opt.save_path, opt.dataset_path, opt.input_size, opt.load_model, opt.save_txt, opt.save_imgs
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = CustomImageDataset(os.path.join(dataset_path, 'test_images'), os.path.join(dataset_path, 'test_masks'), resize=input_size, pretrained=pretrained)
    testloader = DataLoader(test_data, batch_size=1)
    
    if pretrained:
        print('ResnetUnet')
        model = ResNetUnet(num_classes=num_classes)
    else:
        print('Unet')
        model = Unet(num_classes=num_classes)  
    
    if load_model is None:
        print("no loaded model")
        sys.exit()
    else:
        print('load_model...')
        model.load_state_dict(torch.load(load_model))
    
    if save_txt:
        os.makedirs(save_path, exist_ok=True)
        f = open(os.path.join(save_path, 'testresult.txt'),'w')
            
    if save_imgs:
        img_dir = os.path.join(save_path, 'imgs')
        os.makedirs(img_dir, exist_ok=True)
    
    model.to(device) 
    model.eval()
    test_acc_pixel = 0
    test_loss = 0
    test_miou = 0
    iter = 0
    for x, y in testloader:
        org_mask = y.data.cpu().numpy()
        x = x.to(device, dtype=torch.float32)
        pred = model(x)
        # mask labeling and flatten
        y = mask_labeling(y, num_classes)
        y = y.to(device, dtype=torch.long)
    
        loss = nn.CrossEntropyLoss()
        loss_output = loss(pred, y).item()
        
        
        if save_imgs:
            color_map = np.array([[0,0,0],[0,0,255],[255, 0, 0]], np.uint8)
            temp_pred = pred.data.cpu().numpy() # (1, C, H, W)
            temp_pred = np.swapaxes(temp_pred, 1, 3) # (1, W, H, C)
            temp_pred = np.argmax(temp_pred, -1) # (1, W, H)
            show_predict = color_map[temp_pred]     # (1, W, H, 3)
            img_name = test_data.images[iter]
            org_mask = np.transpose(org_mask, (2, 1, 0)) # (1, H, W ) -> (W, H, 1)
            org_mask = np.concatenate([org_mask]*3, axis=2) # (W, H, 3)
            org_mask = Image.fromarray(org_mask, mode="RGB")
            org_mask = np.array(org_mask)
            plt.imsave(os.path.join(img_dir, '{}'.format(img_name)), np.concatenate((org_mask, show_predict[0]), 1))
        
        copied_pred = pred.data.cpu().numpy() 
        copied_y_batch = y.data.cpu().numpy() 
        accuracy_px = accuracy_per_pixel(copied_pred, copied_y_batch, ignore_idx)
        miou_per_batch = miou(copied_pred, copied_y_batch, num_classes, ignore_idx) 
        
        # test loss / test accuracy
        test_acc_pixel += accuracy_px
        test_loss += loss_output
        test_miou += miou_per_batch
        
        iter += 1
    
    test_acc_pixel = test_acc_pixel/ len(testloader)
    test_loss = test_loss / len(testloader)
    test_miou = test_miou / len(testloader)
    
        
    result_txt = "load model(.pt) : %s \n loss: %.8f, Testaccuracy: %.8f, Test miou: %.8f" % (load_model, test_loss, test_acc_pixel, test_miou)       
    if save_txt:
        f.write(result_txt)
        f.close()
    print('----test finish----\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=bool, default=True, help='if it''s true, pretrained encoder model mode')
    parser.add_argument('--num_classes', type=int, default=3, help='the number of classes')
    parser.add_argument('--ignore_idx', type=int, default=None, help='ignore index i.e. background class')
    parser.add_argument('--dataset_path', type=str, default='../cropweed/IJRR2017', help='dataset directory path')
    parser.add_argument('--save_path', type=str, default='./test/test0', help='dataset directory path')
    parser.add_argument('--input_size', type=int, default=512, help='input image size')
    parser.add_argument('--load_model',default='./checkpoint/pretrained/25ep_2b_final.pt', type=str, help='the name of saved model file (.pt)')
    parser.add_argument('--save_txt', type=bool, default=True, help='if it''s true, the result of trainig will be saved as txt file.')
    parser.add_argument('--save_imgs', type=bool, default=True, help='if it''s true, the predict images will saved at image dir')
    
    
    opt = parser.parse_args()
    test(opt)