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
    def __init__(self, transform=None, img_dir='datasets/train/images', classes_num=1024, pic_num=4):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.classes_num = classes_num
        self.pic_num = pic_num
        self.data_num = classes_num * pic_num
        self.last4img = []
        self.lastLabel = 0
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idex):
        label = idex // self.pic_num
        if self.lastLabel != label:
            self.lastLabel = label
            self.last4img = []
            
        img_idx = self.get_imgidx(label)

        img_path = os.path.join(self.img_dir,f'{label}', f'{img_idx}.jpg')
        img = read_image(img_path).to(torch.float32) / 255
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def get_imgidx(self, label):
        img_path = os.path.join(self.img_dir, str(label))
        idx_range = len(os.listdir(img_path))
        while(1):
            img_idx = random.randint(0,idx_range-1)
            if img_idx not in self.last4img:
                self.last4img.append(img_idx)
                return img_idx

class VeRIWildTest(Dataset):
    def __init__(self, transform=None, t='gallery'):
        self.img_dir = os.path.join(t,'images')
        print(f'len of {self.img_dir} : {len(os.listdir(self.img_dir))}')
        self.label_dir = os.path.join(t,'labels')
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idex):
        img_path = os.path.join(self.img_dir, f'{idex}.jpg')
        label_path = os.path.join(self.label_dir, f'{idex}.txt')
        f = open(label_path, 'r')
        label = f.readline()
        f.close()
        raw_img = read_image(img_path).to(torch.float32) / 255
        
        #resize
        trans = self.transform[0]
        raw_img = trans(raw_img)
        #normalize
        trans = self.transform[1]
        img = trans(raw_img)
        return raw_img, img, int(label)

if __name__ == '__main__':

    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip()
    ])

    
    train_set = VeRIWildDataset(trans, 'test/TTT/images', 64, 4)
    train_loader = DataLoader(dataset=train_set, batch_size=32,num_workers=6)

    img_list = torch.empty((0, 3, 224, 224))
    i = 0
    for img, lbl in tqdm(train_loader, dynamic_ncols=True):
        print(f'label : {lbl}')
        print(img.shape)
        save_image(img/255, f'test/TTTSave/{i}.jpg')
        i += 1
        img_list = torch.cat((img_list, img), 0)
    print(img_list.shape)
