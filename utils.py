import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
import torchvision.transforms.functional as FF
from torchvision import transforms


def convert(W, H, x_center_norm, y_center_norm, w_norm, h_norm):
    x_center = x_center_norm * W
    y_center = y_center_norm * H
    w = int(w_norm * W)
    h = int(h_norm * H)
    left = int(x_center - (w/2))
    top =  int(y_center - (h/2))
    
    return left, top, w, h

def get_image_info(info, W, H):
    x_center_norm = float(info[1])
    y_center_norm = float(info[2])
    w_norm = float(info[3])
    h_norm = float(info[4])


    left, top, w, h = convert(W, H, x_center_norm, y_center_norm, w_norm, h_norm)
    
    return left, top, w, h


def crop_frame(image_path, label_path):
    image = read_image(image_path)
    H = image.shape[1]
    W = image.shape[2]

    #get bounding box info
    label = open(label_path, 'r')
    info_list = []
    info = label.readline()
    cropped_regions = torch.empty((0, 3, 224, 224))
    while info:
        info = info.split(' ')
        left, top, w, h= get_image_info(info, W, H)
        info_list.append([left, top, left+w, top+h])
        #crop img
        croped_img = (FF.crop(image, top, left, h, w))/255
        transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])
        croped_img = transform(croped_img)
        cropped_regions = torch.cat((cropped_regions, torch.unsqueeze(croped_img, 0)))
        info = label.readline()
    return cropped_regions, info_list


def top_k(k, dist_matrix, label_list, knn_idx):
    acc = 0
    start = 1

    for i in range(len(dist_matrix)):
        target = int(label_list[i])
        label_count = dict()   
        for j in range(1,k+1):
            label = int(label_list[knn_idx[i][j]])
            if label not in label_count:                    
                label_count[label] = 1
            else:
                label_count[label] += 1
        predict = max(label_count, key=label_count.get)

        if predict == target:
            acc += 1
    return acc/len(dist_matrix)

def top_k_test(k, query_label, gallery_label_list, knn_idx):
    acc = 0
    target = int(query_label)
    label_count = dict()
    for j in range(0,k):
        label = int(gallery_label_list[knn_idx[j]])
        if label not in label_count:                    
            label_count[label] = 1
        else:
            label_count[label] += 1
    predict = max(label_count, key=label_count.get)
    if predict == target:
            acc += 1
    return acc



def compute_dist_sqr(x):
    
    x_len= len(x)
    dist_matrix = torch.empty((x_len, x_len))
    for i in range(x_len):
        for j in range(i, x_len):    

            dist_matrix[i][j] = dist_matrix[j][i] = F.cosine_similarity(x[i], x[j], dim=0)

    return dist_matrix

def compute_dist_rect(x, y):
    y_len = len(y)
    x_len= len(x)
    dist_matrix = torch.empty((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):    

            dist_matrix[i][j] = F.cosine_similarity(x[i], y[j], dim=0)

    return dist_matrix

def draw_loss(loss, save_path):
    x = np.linspace(1, len(loss), num=len(loss))
    plt.clf()
    plt.plot(x, loss)
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.savefig(f'{save_path}')


def draw_acc(acc, save_path):

    x = np.linspace(1, len(acc), num=len(acc))
    plt.clf()
    plt.plot(x, acc)
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.title('acc curve')
    plt.savefig(f'{save_path}')


if __name__ == '__main__':
    cropped_regions = crop_frame('IMAGE/0902_150000_151900/0_00001.jpg', 'LABEL/0902_150000_151900/0_00001.txt')
    for i in range(len(cropped_regions)):
        save_image(cropped_regions[i], f'test/picture/{i}.jpg')