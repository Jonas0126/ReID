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


class trainDataset(Dataset):
    def __init__(self, transform=None, img_dir='datasets/VeRIWild/train/images', class_num=None, image_num=2):
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
        self.picked_file = []
        self.current_img_file = random.choice(self.image_file_list)

        
    def __len__(self):
        return self.class_num * self.image_num

    def __getitem__(self, idex):
        file_index = idex // self.image_num
        if self.last_file_index != file_index:
            self.image_file_list.remove(self.current_img_file) 
            self.last_file_index = file_index
            self.current_img_file = random.choice(self.image_file_list) 
            self.last_img_index = []

        img_index = self.get_img_index()
        img_path = os.path.join(self.img_dir, self.current_img_file, f'{img_index}.jpg')

        img = read_image(img_path).to(torch.float32) / 255
        if self.transform is not None:
             img = self.transform(img)
        return img, int(self.current_img_file)
    

    def get_img_index(self):
        image_file_path = os.path.join(self.img_dir, self.current_img_file)
        index_range = len(os.listdir(image_file_path))
        img_index=0
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

class testDataset(Dataset):
    def __init__(self, transform=None, img_dir='datasets/VeRIWild/train/images', class_num=None, image_num=2):
        super().__init__()

        self.img_dir = img_dir
        self.transform = transform
        self.image_num = image_num+1 #num of image in each class. The reason for adding one is to have a query img.
        self.query = 1
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
        self.picked_file = []
        self.current_img_file = random.choice(self.image_file_list)

        
    def __len__(self):
        return self.class_num * self.image_num

    def __getitem__(self, idex):
        file_index = idex // self.image_num

        if self.last_file_index != file_index:
            self.query = 1
            self.image_file_list.remove(self.current_img_file) 
            self.last_file_index = file_index
            self.current_img_file = random.choice(self.image_file_list) 
            self.last_img_index = []

        img_index = self.get_img_index()
        img_path = os.path.join(self.img_dir, self.current_img_file, f'{img_index}.jpg')
        img = read_image(img_path).to(torch.float32) / 255
        if self.transform is not None:
             img = self.transform(img)
        
        if self.query == 1:
            self.query = 0
            return 1, img, int(self.current_img_file)
        else:
            return -1, img, int(self.current_img_file)

    def get_img_index(self):
        image_file_path = os.path.join(self.img_dir, self.current_img_file)
        index_range = len(os.listdir(image_file_path))
        img_index=0
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
        transforms.Resize((224, 224))
    ])
    
    train_set = VeRIWildTest(trans, 'datasets/VeRIWild/train/images/', None, 4)
    train_loader = DataLoader(dataset=train_set, batch_size=8, num_workers=0)

    gallery_img_list = torch.empty((0, 3, 224, 224))
    query_img_list = torch.empty((0, 3, 224, 224))
    i = 0
    for query, img, lbl in train_loader:
        print(query)
        for p in range(len(query)):
            if query[p] == -1:
                gallery_img_list = torch.cat((gallery_img_list, torch.unsqueeze(img[p], dim=0)), 0)
            else:
                query_img_list = torch.cat((query_img_list, torch.unsqueeze(img[p], dim=0)), 0)
        
        print(f'label : {lbl}')
        print(i)
        i += 1
        if i == 3:
            break
    save_image(gallery_img_list, './test/gallery.jpg')
    save_image(query_img_list, './test/query.jpg')
    print(gallery_img_list.shape)
    print(query_img_list.shape)
