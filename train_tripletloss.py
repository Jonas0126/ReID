from create_dataset import trainDataset
from models.ResNet import PretrainedResNet
from models.loss import tripletLoss
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
import os
import torch
from terminaltables import AsciiTable
from memory_profiler import profile
import psutil
TEST_PIC_NUM = 10 #Number of pictures in each class in the test dataset
TRAIN_PIC_NUM = 4 #Number of pictures in each class in the train dataset
TEST_CLASS_NUM = 150 #The number of classes used in the test dataset.
KNN = [1, 3, 5]
VALID_IMAGE_DIR = 'valid/images/'
TRAIN_IMAGE_DIR = 'train/images/'
SAVE_DIR = 'trained_result'

if not os.path.exists('trained_result/aicup/'):
    os.mkdir('trained_result/aicup/')

if not os.path.exists('trained_result/VeRIWild/'):
    os.mkdir('trained_result/VeRIWild/')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--learning_rate', '-lr', default=0.001, type=float)
    parser.add_argument('--epochs', '-e', default=50, type=int)
    parser.add_argument('--batch',  '-b', default=32, type=int)  
    parser.add_argument('--width',  '-w', default=256, type=int)
    parser.add_argument('--margin', '-m', default=0.5,type=float)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--datasets', type=str)
    args = parser.parse_args()


    epochs = args.epochs
    width = args.width
    
    
    save_dir = os.path.join(SAVE_DIR, f'{args.datasets}', f'{args.model_name}_{str(args.width)}_{str(args.margin)}')
    if not os.path.exists(save_dir):
        os.mkdir(f'{save_dir}')

    train_image_dir = os.path.join('datasets', f'{args.datasets}', 'train/images')
    test_image_dir = os.path.join('datasets', f'{args.datasets}', 'valid/images')
    
    if not os.path.exists(train_image_dir):
        print(f'{train_image_dir} does not exist.')
        exit()

    if not os.path.exists(test_image_dir):
        print(f'{test_image_dir} does not exist.')
        exit()


    train_classes_num = len(os.listdir(train_image_dir))
    print(f'Number of classes for training : {train_classes_num}')
    print(f'Number of classes for valid : {TEST_CLASS_NUM}')


    #record the parameter
    f = open(f'{save_dir}/{args.model_name}_log.txt', 'w')
    f.write(f'lr : {args.learning_rate}, epochs : {args.epochs}, batch size : {args.batch}, image width : {args.width}, margin : {args.margin}\n')

    #define transform for train dataset
    train_transform = transforms.Compose([
        transforms.Resize((width, width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(p=0.4)
    ])

    #define transform for test dataset
    test_transform = transforms.Compose([
        transforms.Resize((width, width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'training on {device}')

    #Instantiate the model
    feature_extractor = PretrainedResNet()
    feature_extractor = feature_extractor.to(device)

 
    #Freeze certain layers of the model.
    for name, param in feature_extractor.named_parameters():
        if 'layer1' in name or 'layer3' in name or 'layer2' in name:
            param.requires_grad = False

    #load train data
    train_set = trainDataset(train_transform, train_image_dir, train_classes_num, TRAIN_PIC_NUM) 
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch, num_workers=6)

    #load test data
    test_set = trainDataset(test_transform, test_image_dir, TEST_CLASS_NUM, TEST_PIC_NUM)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch, num_workers=6)


    img_list = torch.empty((0, 3, width, width))
    label_list = torch.empty(0)
    for img, label in tqdm(test_loader, dynamic_ncols=True, desc=f'load test data'):
        img_list = torch.cat((img_list, img), 0)
        label_list = torch.cat((label_list, label))



    train_loss = []
    valid_acc = [[],[],[]] #index 0 = top 1, index 1 = top 3, index 2 = top 5
    max_acc = 0
    count = 0
    optimizer = torch.optim.Adam(params=feature_extractor.parameters(), lr=args.learning_rate)

    #start train
    for epoch in range(epochs):

        total_loss = 0
        feature_extractor.train()
        feature_extractor = feature_extractor.to(device)

        for bch, data in enumerate(tqdm(train_loader, dynamic_ncols=True, desc=f'epoch {epoch+1}/{epochs}')):
            
            img, label = data 
            img, label = img.to(device), label.to(device)

            feature = feature_extractor(img)
    
            #compute loss
            optimizer.zero_grad()
            triplet_loss = tripletLoss(feature, TRAIN_PIC_NUM, args.margin)(feature)
            loss = triplet_loss
            
            #backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f'loss : {total_loss}, ', end='')
        train_loss.append(total_loss)
        

        #valid model
        feature_extractor.eval()
        feature_extractor = feature_extractor.to('cpu')
        with torch.no_grad():
            feature_set= feature_extractor(img_list.to('cpu'))
            dist_matrix = compute_dist_sqr(feature_set)
            
            knn_idx = torch.argsort(dist_matrix, descending=True, dim=1)

            #find top k
            for i in range(len(KNN)):
                acc = top_k(KNN[i], dist_matrix,label_list, knn_idx)
                valid_acc[i].append(acc)
                print(f', top {KNN[i]} acc : {acc:.4f} ', end='')
            print('')
        
        #Recording accuracy and loss.
        table_data = [
            ['type', 'value'],
            ['top 1', f'{valid_acc[0][epoch]}'],
            ['top 3', f'{valid_acc[1][epoch]}'],
            ['top 5', f'{valid_acc[2][epoch]}'],
            ['total loss', f'{total_loss}']
        ]
        table = AsciiTable(table_data)
        f.write(f'\n{table.table}\n')


        #early stopping
        if max_acc < acc: #top 5 acc
            max_acc = acc
            torch.save(feature_extractor, f'{save_dir}/best.pt')
            count = 0
        else:
            count += 1

        if count >= 10:
            torch.save(feature_extractor, f'{save_dir}/{args.model_name}_{str(args.width)}_{str(args.margin)}.pt')
            draw_loss(train_loss, f'{save_dir}/{args.model_name}_{str(args.width)}_{str(args.margin)}_loss.jpg')
            for i in range(len(KNN)):
                draw_acc(valid_acc[i], f'{save_dir}/{args.model_name}_{str(args.width)}_{str(args.margin)}_top{i}acc.jpg')
            f.close()
            exit()


    #draw loss and acc
    torch.save(feature_extractor, f'{save_dir}/{args.model_name}_{str(args.width)}_{str(args.margin)}.pt')
    draw_loss(train_loss, f'{save_dir}/{args.model_name}_{str(args.width)}_{str(args.margin)}_loss.jpg')
    for i in range(len(KNN)):
        draw_acc(valid_acc[i], f'{save_dir}/{args.model_name}_{str(args.width)}_{str(args.margin)}_top{i}acc.jpg')
    f.close()