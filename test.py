from datasets.create_dataset import VeRIWildTest
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import *

from torchvision.utils import save_image
from torchvision.io import read_image
import os
import torch


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gallery_images_dir', '-gi', default='datasets/VeRIWild/test/gallery/',type=str)
    parser.add_argument('--query_images_dir', '-qi', default='datasets/VeRIWild/test/query/',type=str)
    parser.add_argument('--gallery_labels_dir', '-gl', default='datasets/VeRIWild/test/gallery/',type=str)
    parser.add_argument('--query_labels_dir', '-ql', default='datasets/VeRIWild/test/query/',type=str)
    parser.add_argument('--save', '-s', type=str)
    parser.add_argument('--model_parameter', type=str)
    parser.add_argument('--model_name', '-n', type=str)
    parser.add_argument('--width', '-w', type=int)
    args = parser.parse_args()


    #define transform for dataset
    trans = transforms.Compose([
        transforms.Resize((args.width, args.width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    #create data set
    gallery_img_set = VeRIWildTest(trans, args.gallery_images_dir, 239)
    gallery_loader = DataLoader(dataset=gallery_img_set, batch_size=1, num_workers=1)    
    query_img_set = VeRIWildTest(trans, args.query_images_dir, 40)
    query_loader = DataLoader(dataset=query_img_set, batch_size=1, num_workers=1)
    
    #load model
    model = torch.load(f'{args.model_parameter}').to('cpu')
    model.eval()
    

    k = [1,3,5]
    total_acc = [0,0,0]
    class_num = 0
    #load gallery data
    gallery_label_list = torch.empty(0)
    gallery_img_list = torch.empty((0, 3, args.width, args.width))
    for img, label in tqdm(gallery_loader, dynamic_ncols=True, desc=f'load gallery data'):
        gallery_img_list = torch.cat((gallery_img_list, img), 0)
        gallery_label_list = torch.cat((gallery_label_list, label))
       
    gallery_feature = model(gallery_img_list)

    for query_img, query_label in tqdm(query_loader, dynamic_ncols=True, desc=f'load query data'):
        #initial some parameter
        query_feature = model(query_img)
        dist_matrix = torch.empty(0)
        acc = [0,0,0]
        
        #calculate dist between query and gallery
        dist = compute_dist_rect(query_feature, gallery_feature)    
        dist_matrix=torch.cat((dist_matrix, dist))
        
        #find nearest neighbor
        knn_idx = torch.argsort(torch.squeeze(dist_matrix), descending=True)

        #save image
        save_list = torch.empty((0, 3, args.width, args.width))
        img = read_image(f'{args.query_images_dir}images/{int(query_label)}.jpg').to(torch.float32) / 255
        transform = transforms.Resize((args.width, args.width))
        img = transform(img)
        save_list = torch.cat((save_list, torch.unsqueeze(img, 0)), 0)

        for i in range(5):
            img = read_image(f'{args.gallery_images_dir}images/{knn_idx[i]}.jpg').to(torch.float32) / 255
            img = transform(img)
            save_list = torch.cat((save_list, torch.unsqueeze(img, 0)), 0)

        save_path = os.path.join(args.save, f'{args.model_name}_test_result', f'{args.model_name}_img_{class_num}.jpg')
        save_image(save_list, save_path)
        class_num += 1
        
        #cal top k acc
        for i in range(3):
            acc[i] = top_k_test(k[i], query_label, gallery_label_list, knn_idx)

            total_acc[i] = total_acc[i]+acc[i]
        print('')
    #recording Log
    save_path = os.path.join(args.save, f'{args.model_name}_test_result', f'{args.model_name}_acc_Log.txt')
    f = open(save_path, 'w')
    for i in range(3):
        print(f'total top{k[i]} acc : {total_acc[i]/40}, ', end='')
    
        f.write(f'total top{k[i]} acc : {total_acc[i]/40}\n')
    f.close()
    print('')
    
            


