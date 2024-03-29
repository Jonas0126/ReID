import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import torch
import random



class VeRIWildDataset(Dataset):
    def __init__(self, transform=None, img_dir='datasets/train/images', label_dir='datasets/train/label', classes_num=1024, pic_num=4):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.classes_num = classes_num
        self.pic_num = pic_num
        self.data_num = classes_num * pic_num
        self.last4img = []
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idex):
        label = idex // self.pic_num
        label_path = os.path.join(self.label_dir, f'{label}.txt')
        img_idx = self.get_img_idx(label_path)
        

        img_path = os.path.join(self.img_dir, f'{img_idx}.jpg')
        img = read_image(img_path)
        
        if self.transform is not None:
            img = self.transform(img)

        return img.to(torch.float32), label



    def get_img_idx(self, label_path):
        file = open(label_path, 'r')
        line_split = file.readline().split()
        start_idx = int(line_split[0])
        end_idx = int(line_split[1])
        
        img_idx = start_idx
        while(img_idx<=end_idx):
            if img_idx not in self.last4img:
                self.last4img.append(img_idx)
                if len(self.last4img) > self.pic_num:
                    del self.last4img[0]
                break
            img_idx += 1


        '''
        #random choice img 
        while(1):
            img_idx = random.randint(start_idx, end_idx)
            if img_idx not in self.last4img:
                self.last4img.append(img_idx)
                if len(self.last4img) > self.pic_num:
                    del self.last4img[0]
                break
        '''

        return img_idx
    






        
if __name__ == '__main__':

    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip()
    ])

    
    train_set = VeRIWildDataset()
    train_loader = DataLoader(dataset=train_set, batch_size=32, num_workers=6)

    img_list = torch.empty((0, 3, 224, 224))
    for img, lbl in tqdm(train_loader, dynamic_ncols=True):
        
        print(img.shape)
        img_list = torch.cat((img_list, img), 0)
    print(img_list.shape)
