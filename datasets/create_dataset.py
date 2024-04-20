import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import torch
import random
import cv2


class VeRIWildDataset(Dataset):
    def __init__(self, transform=None, img_dir='datasets/VeRIWild/train/images', classes_num=1024, pic_num=4):
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
    def __init__(self, transform=None, t='gallery', L = 71):
        self.img_dir = os.path.join(t,'images')
        print(f'len of {self.img_dir} : {len(os.listdir(self.img_dir))}')
        self.label_dir = os.path.join(t,'labels')
        self.transform = transform
        self.L = L
    def __len__(self):
        return self.L
    
    def __getitem__(self, idex):
        
        img_path = os.path.join(self.img_dir, f'{idex}.jpg')
        label_path = os.path.join(self.label_dir, f'{idex}.txt')
        f = open(label_path, 'r')
        label = f.readline()
        f.close()
        img = read_image(img_path).to(torch.float32) / 255
        
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)



class AiCupDataset(Dataset):
    def __init__(self, transform=None, img_dir='datasets/aicup/aicup_reid/train/images', class_num=None, image_num=2):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.image_num = image_num #num of image in each class

        image_files = os.listdir(img_dir)

        #filter categories with fewer images than
        self.image_file_list = self.filt_image_files(image_files, img_dir) 
        if class_num is not None:
            self.class_num = class_num
        else:
            self.class_num = len(self.image_file_list)
        print(f'class num : {self.class_num}')
        self.last_img_index = []
        self.last_file_index = 0

    def __len__(self):
        return self.class_num * self.image_num

    def __getitem__(self, idex):
        file_index = idex // self.image_num
        if self.last_file_index != file_index:
            self.last_file_index = file_index
            self.last_img_index = []
        
        img_index = self.get_img_index(file_index)
        img_path = os.path.join(self.img_dir, self.image_file_list[file_index], f'{img_index}.jpg')
        img = read_image(img_path).to(torch.float32) / 255
        
        if self.transform is not None:
            img = self.transform(img)

        return img, int(self.image_file_list[file_index])

    def get_img_index(self, file_index):
        image_file_path = os.path.join(self.img_dir, self.image_file_list[file_index])
        index_range = len(os.listdir(image_file_path))
        while(1):
            img_index = random.randint(0,index_range-1)
            if img_index not in self.last_img_index:
                self.last_img_index.append(img_index)
                return img_index
            
    def filt_image_files(self, image_files, img_dir):
        #Filter files with fewer than 2 images
        image_file_list = []
        for image_file in image_files:
            image_file_path = os.path.join(img_dir, image_file)
            images = os.listdir(image_file_path)
            if len(images) >= self.image_num:
                image_file_list.append(image_file)
        return image_file_list





if __name__ == '__main__':


    trans = transforms.Compose([
        transforms.Resize((256, 256)),

        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip()
    ])
    
    train_set = AiCupDataset(trans, 'aicup/aicup_reid/train/images/',None, 4)
    train_loader = DataLoader(dataset=train_set, batch_size=8, num_workers=6)

    # img_list = torch.empty((0, 3, 224, 224))
    # i = 0
    # for img, lbl in tqdm(train_loader, dynamic_ncols=True):
    #     print(f'label : {lbl}')
    #     print(img.shape)
    #     save_image(img, f'../test/TTTSave/{i}.jpg')
    #     i += 1
    #     img_list = torch.cat((img_list, img), 0)
    # print(img_list.shape)
