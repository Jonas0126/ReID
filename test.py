from create_dataset import testDataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import *
from torchvision.utils import save_image
from torchvision.io import read_image
from models.ResNet import PretrainedResNet
import os
import torch

TEST_CLASS_NUM = 60
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--images_dir', '-i', default='datasets/VeRIWild/test',type=str)
    parser.add_argument('--save_dir', '-s', type=str)
    parser.add_argument('--model_parameter', type=str)
    parser.add_argument('--model_name', '-n', type=str)
    parser.add_argument('--width', '-w', type=int)
    args = parser.parse_args()


    #define transform for dataset
    trans = transforms.Compose([
        transforms.Resize((args.width, args.width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    invTrans = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ],std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ]),
        ])
    
    ave_acc = [0,0,0]
    for iter_ in tqdm(range(1), dynamic_ncols=True, desc=f'test model'):
    #create data set
        test_set = testDataset(trans, args.images_dir, TEST_CLASS_NUM, 3)
        test_loader = DataLoader(dataset=test_set, batch_size=32, num_workers=0)

        #load model
        model = torch.load(f'{args.model_parameter}').to('cpu')
        model.eval()
        

        k = [1,3,5]
        
        class_num = 0

    
        gallery_img_list = torch.empty((0, 3, args.width, args.width))
        query_img_list = torch.empty((0, 3, args.width, args.width))
        gallery_label_list = torch.empty(0)
        query_label_list = torch.empty(0)
        #load gallery and query data
        for query, img, label in test_loader:
            for p in range(len(query)):
                if query[p] == -1:
                    gallery_img_list = torch.cat((gallery_img_list, torch.unsqueeze(img[p], dim=0)), 0)
                    gallery_label_list = torch.cat((gallery_label_list, torch.unsqueeze(label[p], dim=0)), 0)
                else:
                    query_img_list = torch.cat((query_img_list, torch.unsqueeze(img[p], dim=0)), 0)
                    query_label_list = torch.cat((query_label_list, torch.unsqueeze(label[p], dim=0)), 0)
        
        gallery_feature = model(gallery_img_list)

    
        total_acc = [0,0,0]
        for p in range(len(query_img_list)):
            #initial some parameter
            query_feature = model(torch.unsqueeze(query_img_list[p], dim=0))
            dist_matrix = torch.empty(0)
            acc = [0,0,0]

            #calculate dist between query and gallery
            dist = compute_dist_rect(query_feature, gallery_feature)    
            dist_matrix=torch.cat((dist_matrix, dist))

            #find nearest neighbor
            knn_idx = torch.argsort(torch.squeeze(dist_matrix), descending=True)

            #save image
            save_list = torch.empty((0, 3, args.width, args.width))
            img = query_img_list[p]
            save_list = torch.cat((save_list, torch.unsqueeze(invTrans(img), 0)), 0)

            for i in range(5):
                img = gallery_img_list[knn_idx[i]]
                save_list = torch.cat((save_list, torch.unsqueeze(invTrans(img), 0)), 0)

            save_path = os.path.join(args.save_dir, f'{args.model_name}_test_result')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            knn_label = f'_{gallery_label_list[knn_idx[0]]}_{gallery_label_list[knn_idx[1]]}_{gallery_label_list[knn_idx[2]]}_{gallery_label_list[knn_idx[3]]}_{gallery_label_list[knn_idx[4]]}'
            save_image(save_list, f'{save_path}/{args.model_name}_img_{class_num}_{query_label_list[p]}{knn_label}.jpg')
            class_num += 1

            #cal top k acc
            for i in range(3):
                acc[i] = top_k_test(k[i], query_label_list[p], gallery_label_list, knn_idx)

                total_acc[i] = total_acc[i]+acc[i]
            
        for i in range(3):
            print(f'total top{k[i]} acc : {total_acc[i] / TEST_CLASS_NUM}, ')
            ave_acc[i] += total_acc[i] / TEST_CLASS_NUM
    # for query_img, query_label in tqdm(query_loader, dynamic_ncols=True, desc=f'load query data'):
    #     #initial some parameter
    #     query_feature = model(query_img)
    #     dist_matrix = torch.empty(0)
    #     acc = [0,0,0]
        
    #     #calculate dist between query and gallery
    #     dist = compute_dist_rect(query_feature, gallery_feature)    
    #     dist_matrix=torch.cat((dist_matrix, dist))
        
    #     #find nearest neighbor
    #     knn_idx = torch.argsort(torch.squeeze(dist_matrix), descending=True)

    #     #save image
    #     save_list = torch.empty((0, 3, args.width, args.width))
    #     img = read_image(f'{args.query_images_dir}images/{int(query_label)}.jpg').to(torch.float32) / 255
    #     transform = transforms.Resize((args.width, args.width))
    #     img = transform(img)
    #     save_list = torch.cat((save_list, torch.unsqueeze(img, 0)), 0)

    #     for i in range(5):
    #         img = read_image(f'{args.gallery_images_dir}images/{knn_idx[i]}.jpg').to(torch.float32) / 255
    #         img = transform(img)
    #         save_list = torch.cat((save_list, torch.unsqueeze(img, 0)), 0)

    #     save_path = os.path.join(args.save, f'{args.model_name}_test_result')
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    #     save_image(save_list, f'{save_path}/{args.model_name}_img_{class_num}.jpg')
    #     class_num += 1
        
    #     #cal top k acc
    #     for i in range(3):
    #         acc[i] = top_k_test(k[i], query_label, gallery_label_list, knn_idx)

    #         total_acc[i] = total_acc[i]+acc[i]
    #     print('')

    #recording Log
    save_path = os.path.join(args.save_dir, f'{args.model_name}_test_result')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    f = open(f'{save_path}/{args.model_name}_acc_Log.txt', 'w')

    for i in range(3):
        print(f'ave top{k[i]} acc : {ave_acc[i]}, ', end='')
    
        f.write(f'ave top{k[i]} acc : {ave_acc[i]}\n')
    f.close()
    print('')
    
            


