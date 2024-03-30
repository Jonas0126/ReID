import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import torch
import random



class VeRIWildDataset(Dataset):
    def __init__(self, transform=None, img_dir='datasets/train/images', classes_num=2500, pic_num=4):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.classes_num = classes_num
        self.pic_num = pic_num
        self.data_num = classes_num * pic_num
        self.last4img = []
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idex):
        label = idex // self.pic_num

        img_idx = idex % self.pic_num
        

        img_path = os.path.join(self.img_dir,f'{label}', f'{img_idx}.jpg')
        img = read_image(img_path)
        
        if self.transform is not None:
            img = self.transform(img)

        return img.to(torch.float32), label






        
if __name__ == '__main__':

    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip()
    ])

    
    train_set = VeRIWildDataset(trans, 'test/TTT/images', 5, 2)
    train_loader = DataLoader(dataset=train_set, batch_size=2,num_workers=6)

    img_list = torch.empty((0, 3, 224, 224))
    for img, lbl in tqdm(train_loader, dynamic_ncols=True):
        print(f'img : {img}')
        print(f'label : {lbl}')
        print(img.shape)
        save_image(img/255, f'test/TTTSave/{lbl[0]}.jpg')
        img_list = torch.cat((img_list, img), 0)
    print(img_list.shape)
