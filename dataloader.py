import glob
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, resize=512, transform=None, target_transform=None):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.resize=resize
        self.images = glob.glob(os.path.join(self.img_dir, '*.png'))
        self.images = [img.split('/')[-1] for img in self.images]
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):   
        img_path = os.path.join(self.img_dir, self.images[idx])
        tf = transforms.ToTensor()
        image = Image.open(img_path)
        image = image.resize((self.resize, self.resize))
        image = tf(image)
        image *= 255
        image = F.pad(image, (4, 4, 4, 4))
        
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.resize, self.resize), resample=Image.NEAREST)
        mask = tf(mask)
        mask *= 255
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask

if __name__ == '__main__':
    train_dataset = CustomImageDataset('../cropweed/IJRR2017/train_images', '../cropweed/IJRR2017/train_masks')
    print(len(train_dataset))
    print(train_dataset[2])

