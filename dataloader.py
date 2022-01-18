import glob
import os
import PIL
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.images = glob.glob(os.path.join(self.img_dir, '*.png'))
        self.images = [img.split('/')[-1] for img in self.images]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):   
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = PIL.Image.open(img_path)
        tf = transforms.ToTensor()
        image = tf(image)
        mask_path = self.mask_dir + '/' + self.images[idx]
        mask = PIL.Image.open(mask_path)
        tf = transforms.ToTensor()
        mask = tf(mask)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

if __name__ == '__main__':
    train_dataset = CustomImageDataset('../cropweed/IJRR2017/train_images', '../cropweed/IJRR2017/train_masks')
    print(len(train_dataset))
    print(train_dataset[2])

