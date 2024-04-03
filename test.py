from datasets.create_dataset import VeRIWildTest
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.dist import *
from utils.find_knn import top_k
from torchvision.utils import save_image
import os
import torch


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gallery_images_dir', '-gi', default='datasets/test/gallery/',type=str)
    parser.add_argument('--query_images_dir', '-qi', default='datasets/test/query/',type=str)
    parser.add_argument('--gallery_labels_dir', '-gl', default='datasets/test/gallery/',type=str)
    parser.add_argument('--query_labels_dir', '-ql', default='datasets/test/query/',type=str)
    parser.add_argument('--model_name', '-n', type=str)
    parser.add_argument('--width', '-w', type=int)
    args = parser.parse_args()


    #define transform for dataset
    trans = [
        transforms.Resize((args.width, args.width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


    #create data set
    gallery_img_set = VeRIWildTest(trans, args.gallery_images_dir)
    gallery_loader = DataLoader(dataset=gallery_img_set, batch_size=32, num_workers=6)    
    query_img_set = VeRIWildTest(trans, args.query_images_dir)
    query_loader = DataLoader(dataset=query_img_set, batch_size=32, num_workers=6)
    
    #load model
    model = torch.load(f'{args.model_name}.pt').to('cpu')
    model.eval()

    #load data and feature
    raw_gallery_img_list = torch.empty((0, 3, args.width, args.width))
    gallery_feature = torch.empty(0, 2048)
    norm_gallery_img_list = torch.empty((0, 3, args.width, args.width))
    gallery_label_list = torch.empty(0)
    for raw_img, img, label in tqdm(gallery_loader, dynamic_ncols=True, desc=f'load gallery data'):
        raw_gallery_img_list = torch.cat((raw_gallery_img_list, raw_img), 0)
        norm_gallery_img_list = torch.cat((norm_gallery_img_list, img), 0)
        
        feature = model(img)
        gallery_feature = torch.cat((gallery_feature, feature), 0)
        
        gallery_label_list = torch.cat((gallery_label_list, label))

    raw_query_img_list = torch.empty((0, 3, args.width, args.width))
    norm_query_img_list = torch.empty((0, 3, args.width, args.width))
    query_feature = torch.empty(0, 2048)
    query_label_list = torch.empty(0)
    for raw_img, img, label in tqdm(query_loader, dynamic_ncols=True, desc=f'load query data'):
        raw_query_img_list = torch.cat((raw_query_img_list, raw_img), 0)
        norm_query_img_list = torch.cat((norm_query_img_list, img), 0)
        feature = model(img)
        query_feature = torch.cat((query_feature, feature), 0)
        query_label_list = torch.cat((query_label_list, label))

    
    dist_matrix = compute_dist_rect(query_feature, gallery_feature)
    knn_idx = torch.argsort(dist_matrix, descending=True, dim=1)

    k = [1, 3]

    for i in range(len(k)):
        acc = top_k(k[i],dist_matrix,gallery_label_list, knn_idx,1)
        print(f', top {k[i]} acc : {acc} ', end='')
    print('')


    
    for i in range(100):
        save_list = torch.empty((0, 3, args.width, args.width))
        save_list = torch.cat((save_list, torch.unsqueeze(raw_query_img_list[i], 0)), 0)
        for j in range(5):
            save_list = torch.cat((save_list, torch.unsqueeze(raw_gallery_img_list[knn_idx[i][j]], 0)), 0)

        save_image(save_list, f'test_result/img_{i}.jpg')

    
            


