import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from evaluate import accuracy_per_pixel, miou, confusion_matrix
from utils.utils import mask_labeling
from utils.dice_loss import DiceLoss
from utils.focal_loss import FocalLoss
from dataloader import CustomImageDataset
from models.model import Unet
from models.pretrained_model import ResNetUnet
from torch.utils.data import DataLoader
import glob

def test(opt):
    pretrained, num_classes, ignore_idx, save_path, dataset_path, input_size, load_model, save_txt, save_imgs =\
        opt.pretrained, opt.num_classes, opt.ignore_idx, opt.save_path, opt.dataset_path, opt.input_size, opt.load_model, opt.save_txt, opt.save_imgs
    loss_fn = opt.loss_fn
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = CustomImageDataset(os.path.join(dataset_path, 'test_images'), os.path.join(dataset_path, 'test_masks'), resize=input_size, pretrained=pretrained)
    testloader = DataLoader(test_data, batch_size=1, shuffle=False)
    
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
    
    # testdirs = os.listdir(save_path)
    # testdir = 'test0' if testdirs==[] else 'test' + str(len(testdirs))

    save_path = os.path.join(save_path, f"{dataset_path.split('/')[-1]}-{model.__class__.__name__}-"+str(len(os.listdir(save_path))))
    if save_txt:
        os.makedirs(save_path, exist_ok=True)
        f = open(os.path.join(save_path, 'testresult.txt'),'w')
        f.write(f"dataset : {dataset_path}\n")
            
    if save_imgs:
        img_dir = os.path.join(save_path, 'imgs')
        os.makedirs(img_dir, exist_ok=True)
    
    model.to(device) 
    model.eval()
    test_acc_pixel = 0
    test_loss = 0
    test_miou = 0
    iou_per_class = np.array([0]*num_classes, dtype=np.float64)
    conf_sum = np.zeros((num_classes, 3))
    
    iter = 0
    
    for x, y in testloader:
        org_mask = y.data.cpu().numpy()
        x = x.to(device, dtype=torch.float32)
        pred = model(x)
        # mask labeling and flatten
        y = mask_labeling(y, num_classes)
        y = y.to(device, dtype=torch.long)

        if loss_fn == 'ce':
            # cross entropy loss
            if ignore_idx is not None:
                loss = nn.CrossEntropyLoss()
            else: 
                loss = nn.CrossEntropyLoss()
        
        elif loss_fn == 'dice':
            # dice loss
            loss = DiceLoss(num_classes, ignore_idx)
        
        elif loss_fn == 'focal':
            # focal loss
            loss = FocalLoss(num_classes)
        
        loss_output = loss(pred, y).item()
        
        if iter % 3 == 0:
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
                if not pretrained:
                    org_img = x.data.cpu().numpy()[0][:, 4:4+input_size, 4:4+input_size] * 255 # (3, H, W)
                else:
                    org_img = x.data.cpu().numpy()[0][:, 2:2+input_size, 2:2+input_size] * 255 # (3, H, W)
                org_img = np.swapaxes(org_img, 0, 2)
                org_img = np.uint8(org_img)
                plt.imsave(os.path.join(img_dir, '{}'.format(img_name)), np.concatenate(( org_img , org_mask, show_predict[0]), 1))
            
        copied_pred = pred.data.cpu().numpy() 
        copied_y_batch = y.data.cpu().numpy() 
        accuracy_px = accuracy_per_pixel(copied_pred, copied_y_batch, ignore_idx)
        miou_per_batch, iou_ndarray = miou(copied_pred, copied_y_batch, num_classes, ignore_idx) 
        # modified iou
        conf = confusion_matrix(copied_pred, copied_y_batch, num_classes, ignore_idx)
        
        # test loss / test accuracy
        test_acc_pixel += accuracy_px
        test_loss += loss_output
        test_miou += miou_per_batch
        iou_per_class += iou_ndarray
        conf_sum += conf
        
        iter += 1
    
    test_acc_pixel = test_acc_pixel/ len(testloader)
    test_loss = test_loss / len(testloader)
    test_miou = test_miou / len(testloader)
    test_ious = np.round((iou_per_class / len(testloader)), 5).tolist()
    ## modified iou
    test_m_ious = conf_sum[:, 2] / (conf_sum[:, 1]+conf_sum[:, 2])
    test_m_miou = np.mean(test_m_ious)
        
    result_txt = "load model(.pt) : %s \n loss: %.8f, Testaccuracy: %.8f, Test miou: %.8f" % (load_model, test_loss, test_acc_pixel, test_miou)       
    result_txt += f"\niou per class {test_ious}"
    result_txt += f"\nmodified iou {test_m_ious}, modified miou{test_m_miou}"
    if save_txt:
        f.write(result_txt)
        f.close()
    print('----test finish----\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=bool, default=False, help='if it''s true, pretrained encoder model mode')
    parser.add_argument('--num_classes', type=int, default=3, help='the number of classes')
    parser.add_argument('--loss_fn', type=str, default='focal', help='loss function. ce / dice ')
    parser.add_argument('--ignore_idx', type=int, default=None, help='ignore index i.e. background class')
    parser.add_argument('--dataset_path', type=str, default='../cropweed/CWFID', help='dataset directory path')
    parser.add_argument('--save_path', type=str, default='./test', help='dataset directory path')
    parser.add_argument('--input_size', type=int, default=512, help='input image size')
    parser.add_argument('--load_model',default='./diceloss-checkpoint/CWFID-Unet-50-215/50ep_2b_final.pt', type=str, help='the name of saved model file (.pt)')
    parser.add_argument('--save_txt', type=bool, default=True, help='if it''s true, the result of trainig will be saved as txt file.')
    parser.add_argument('--save_imgs', type=bool, default=True, help='if it''s true, the predict images will saved at image dir')
    
    
    opt = parser.parse_args()
    # test(opt)

    # datasetdir = '../cropweed'
    # checkpointdir = './diceloss-checkpoint'
    # checkpoints = os.listdir(checkpointdir)
    # for dataset in os.listdir(datasetdir):
    #     opt.dataset_path = os.path.join(datasetdir, dataset)
    #     for checkpoint in checkpoints:
    #         if checkpoint.split('-')[0]  == dataset:
    #             checkpoint_path = os.path.join(checkpointdir, checkpoint)
    #             weights = glob.glob(checkpoint_path+'/*.pt')
    #             for weight in weights:
    #                 opt.load_model = weight
    #                 test(opt) 
    
    blurdatasetdir = '../blured_cropweed_strong'
    checkpointdir = './blurtrain-checkpoint'
    checkpoints = os.listdir(checkpointdir)
    for dataset in os.listdir(blurdatasetdir):
        opt.dataset_path = os.path.join(blurdatasetdir, dataset)
        for checkpoint in checkpoints:
            # if checkpoint.split('-')[0] + '_b' == dataset:
            if checkpoint.split('-')[0] == dataset:
                checkpoint_path = os.path.join(checkpointdir, checkpoint)
                weights = glob.glob(checkpoint_path+'/*.pt')
                for weight in weights:
                    opt.load_model = weight
                    test(opt)    
   
                    
