from datasets.create_dataset import VeRIWildDataset
from models.ResNet import ResNet
from models.loss import tripletLoss
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.draw_loss import draw_loss 
from utils.dist import *
import os
import torch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_image_dir', '-i', default='datasets/train/images/',type=str)
    parser.add_argument('--train_label_dir', '-l', default='datasets/train/label/',type=str)
    parser.add_argument('--test_image_dir', '-testimg', default='datasets/test/images/',type=str)
    parser.add_argument('--test_label_dir', '-testlbl', default='datasets/test/label/',type=str)
    parser.add_argument('--save_dir', '-s', default='trained_model/', type=str)

    parser.add_argument('--epochs', '-e', default=100,type=int)
    parser.add_argument('--batch', '-b', default=32, type=int)  
    parser.add_argument('--learning_rate', '-lr', default=0.01, type=float)
    parser.add_argument('--knn', '-k', default=3, type=int)
    parser.add_argument('--width', '-w', default=256, type=int)
    parser.add_argument('--pic_num', '-p', default=4,type=int)
    parser.add_argument('--margin', '-m', default=1,type=int)
    args = parser.parse_args()

    if not os.path.exists(args.train_image_dir):
        print(f'{args.train_image_dir} does not exist.')
        exit()

    if not os.path.exists(args.test_image_dir):
        print(f'{args.test_image_dir} does not exist.')
        exit()

    if not os.path.exists(args.test_label_dir):
        print(f'{args.test_label_dir} does not exist.')
        exit()

    if not os.path.exists(args.train_label_dir):
        print(f'{args.train_label_dir} does not exist.')
        exit()

    if not os.path.exists(args.save_dir):
        os.mkdir(f'{args.save_dir}')


train_classes_num = len(os.listdir(args.train_label_dir))
test_classes_num = len(os.listdir(args.test_label_dir))

epochs = args.epochs
k = args.knn
width = args.width
#define transform for train dataset
trans = transforms.Compose([
    transforms.Resize((width, width)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomHorizontalFlip()
])

#num of block in ResNet101
no_block = [3, 4, 26, 3]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'training on {device}')

#Instantiate the model
feature_extractor = ResNet(no_block, train_classes_num)
feature_extractor = feature_extractor.to(device)


#load train data
train_set = VeRIWildDataset(trans, args.train_image_dir, args.train_label_dir, train_classes_num, args.pic_num) 
train_loader = DataLoader(dataset=train_set, batch_size=args.batch, num_workers=6)

#load test data
test_set = VeRIWildDataset(transforms.Resize((width, width)), args.train_image_dir, args.train_label_dir, k, args.pic_num)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch, num_workers=6)

img_list = torch.empty((0, 3, width, width))
label_list = torch.empty(0)
for img, label in tqdm(test_loader, dynamic_ncols=True, desc=f'load test data'):
    
    img_list = torch.cat((img_list, img), 0)
    label_list = torch.cat((label_list, label))



train_loss = []
optimizer = torch.optim.Adam(params=feature_extractor.parameters(), lr=args.learning_rate)

#start train
for epoch in range(epochs):
    total_loss = 0
    feature_extractor.train()
    feature_extractor = feature_extractor.to(device)
    for bch, data in enumerate(tqdm(train_loader, dynamic_ncols=True, desc=f'epoch{epoch+1}/{epochs}')):
        
        img, label = data 
        img, label = img.to(device), label.to(device)

        feature, id_logits = feature_extractor(img)
        

        #compute loss
        optimizer.zero_grad()
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        id_loss = cross_entropy_loss(id_logits, label)
        triplet_loss = tripletLoss(feature, args.pic_num, args.margin)(feature)
        loss = triplet_loss
        
        #backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f'loss : {total_loss}, triplet_loss: {triplet_loss}, id_loss: {id_loss}', end='')
    train_loss.append(total_loss)
    torch.save(feature_extractor, f'{args.save_dir}resnet.pt')

    #test model
    acc = 0
    
    feature_extractor.eval()
    feature_extractor = feature_extractor.to('cpu')
    with torch.no_grad():
        feature_set, _ = feature_extractor(img_list.to('cpu'))
        dist_matrix = compute_dist(feature_set, 0)
        print(f'dist matrix : {dist_matrix}')
        knn_idx = torch.argsort(dist_matrix, dim=1)
        print(f'knn_idx : {knn_idx}')
        for i in range(len(dist_matrix)):
            target = label_list[i]
            corect_num = 0
            for j in range(1,k+1):
                if target == label_list[knn_idx[i][j]]:
                    corect_num+=1
            acc+= corect_num / k
        
        print(f'acc : {acc/len(dist_matrix)}')

draw_loss(train_loss, 'result/resnet_v1.jpg')