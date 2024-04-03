from datasets.create_dataset import VeRIWildDataset
from models.ResNet import ResNet, PretrainedResNet
from models.loss import tripletLoss
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.draw_loss import draw_loss, draw_acc 
from utils.dist import *
from utils.find_knn import top_k
import os
import torch
from terminaltables import AsciiTable

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_image_dir', '-i', default='datasets/train/images/',type=str)
    parser.add_argument('--test_image_dir', '-testimg', default='datasets/valid/images/',type=str)
    parser.add_argument('--save_dir', '-s', default='result/', type=str)
    parser.add_argument('--model_name', default='resnet', type=str)

    parser.add_argument('--epochs', '-e', default=100,type=int)
    parser.add_argument('--batch', '-b', default=32, type=int)  
    parser.add_argument('--learning_rate', '-lr', default=0.01, type=float)
    parser.add_argument('--knn', '-k', default=[1,3,5], type=list)
    parser.add_argument('--width', '-w', default=256, type=int)
    parser.add_argument('--train_pic_num', default=4,type=int)
    parser.add_argument('--test_pic_num', default=10,type=int)
    parser.add_argument('--margin', '-m', default=1,type=float)
    parser.add_argument('--block', default=[3, 4, 26, 3], type=list)

    args = parser.parse_args()

    if not os.path.exists(args.train_image_dir):
        print(f'{args.train_image_dir} does not exist.')
        exit()

    if not os.path.exists(args.test_image_dir):
        print(f'{args.test_image_dir} does not exist.')
        exit()

    save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(save_dir):
        os.mkdir(f'{save_dir}')


    train_classes_num = len(os.listdir(args.train_image_dir))
    test_classes_num = len(os.listdir(args.test_image_dir))
    print(f'trains classes : {train_classes_num}')
    print(f'test classes : {test_classes_num}')

    epochs = args.epochs
    k = args.knn
    width = args.width
    no_block = args.block #num of block in ResNet

    #log parameter
    f = open(f'{save_dir}/{args.model_name}_log.txt', 'w')
    f.write(f'lr : {args.learning_rate}, epochs : {args.epochs}, batch size : {args.batch}, image width : {args.width}, margin : {args.margin}\n')

    #define transform for dataset
    train_transform = transforms.Compose([
        transforms.Resize((width, width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.ColorJitter(brightness=0.5),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomErasing(p=0.4)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((width, width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'training on {device}')

    #Instantiate the model
    feature_extractor = PretrainedResNet()
    feature_extractor = feature_extractor.to(device)


    for name, param in feature_extractor.named_parameters():
        if 'layer1' in name or 'layer3' in name or 'layer2' in name:
            param.requires_grad = False


    #load train data
    train_set = VeRIWildDataset(train_transform, args.train_image_dir, train_classes_num, args.train_pic_num) 
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch, num_workers=6)

    #load test data
    test_set = VeRIWildDataset(test_transform, args.test_image_dir, 100, args.test_pic_num)
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

            feature= feature_extractor(img)
    
            #compute loss
            optimizer.zero_grad()
            #cross_entropy_loss = torch.nn.CrossEntropyLoss()
            #id_loss = cross_entropy_loss(id_logits, label)
            triplet_loss = tripletLoss(feature, args.train_pic_num, args.margin)(feature)
            loss = triplet_loss
            
            #backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f'loss : {total_loss}, triplet_loss: {triplet_loss}', end='')
        train_loss.append(total_loss)
        

        #test model
        feature_extractor.eval()
        feature_extractor = feature_extractor.to('cpu')
        with torch.no_grad():
            feature_set= feature_extractor(img_list.to('cpu'))
            dist_matrix = compute_dist_sqr(feature_set, feature_set)
            
            knn_idx = torch.argsort(dist_matrix, descending=True, dim=1)

            #find top k
            for i in range(len(k)):
                acc = top_k(k[i],dist_matrix,label_list, knn_idx)
                valid_acc[i].append(acc)
                print(f', top {k[i]} acc : {acc} ', end='')
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

        # save best model
        if max_acc <= acc: #top 5 acc
            max_acc = acc
            torch.save(feature_extractor, f'{save_dir}/best.pt')


        #early stopping
        # if max_acc < acc: #top 5 acc
        #     max_acc = acc
        #     torch.save(feature_extractor, f'{save_dir}/best.pt')
        #     count = 0
        # else:
        #     count += 1

        # if count >= 10:
        #     draw_loss(train_loss, f'{save_dir}/{args.model_name}_loss.jpg')
        #     draw_acc(valid_acc, f'{save_dir}/{args.model_name}_acc.jpg')
        #     f.write(f'early stop at epoch {epoch}\n')
        #     f.close()
        #     exit()


    #draw loss and acc
    torch.save(feature_extractor, f'{save_dir}/{args.model_name}.pt')
    draw_loss(train_loss, f'{save_dir}/{args.model_name}_loss.jpg')
    for i in range(len(k)):
        draw_acc(valid_acc[i], f'{save_dir}/{args.model_name}_top{i}acc.jpg')
    f.close()