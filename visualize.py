from utils import *
from argparse import ArgumentParser
import torch
import os
from collections import defaultdict, deque
from models.ResNet import PretrainedResNet
from tqdm import tqdm
from Matcher import Matcher
import cv2
class Palette:
    def __init__(self):     
        self.colors = {}


    def get_color(self, id):
        if not id in self.colors:
            color = list(np.random.choice(range(256), size=3))
            color = (int(color[0]), int(color[1]), int(color[2]))

            self.colors[id] = color

        return self.colors[id]



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--params', '-p', type=str, default='./goodshit/resnet_v14/best.pt', help='the path to the pytorch model')
    parser.add_argument('--images', '-i', type=str, default='./IMAGE/0902_150000_151900', help='path to the images')
    parser.add_argument('--labels', '-l', type=str, default='./LABEL/0902_150000_151900', help='path to the corresponding labels')
    parser.add_argument('--detail', type=str, default='off', help='record all information include dist_matrix etc.')
    parser.add_argument('--cam', '-c', type=int, help='the camera you want to visualize')
    parser.add_argument('--out', type=str, help='the path to save the video')
    parser.add_argument('--fps', type=int, default=2)

    args = parser.parse_args()


    camera_tracks = defaultdict(list)
    for file in os.listdir(args.images):
        camera_id = int(file[0])
        camera_tracks[camera_id].append(os.path.join(args.images, file))
    for k, v in camera_tracks.items():
        v.sort()


    camera_labels = defaultdict(list)
    for file in os.listdir(args.labels):
        camera_id = int(file[0])
        camera_labels[camera_id].append(os.path.join(args.labels, file))
    for k, v in camera_labels.items():
        v.sort()


    save_path = os.path.join(f'{args.out}', 'video/output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    out = cv2.VideoWriter(save_path, fourcc, 2, (1280,  720)) 

    imgs = camera_tracks[args.cam]
    labels = camera_labels[args.cam]
    
    #load extractor
    extracter = PretrainedResNet()
    extracter = torch.load(args.params)
    extracter.eval()

    #create Matcher
    matcher = Matcher(threshold=0.55)

    pre_feature = []
    pre_idlist = []
    ids = 0

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    palette = Palette()


    for i in tqdm(range(len(imgs))):
        current_imgs = []    
        current_feature = []
        info_list = []
        match_pair = dict()
        current_imgs, info_list = crop_frame(image_path=imgs[i], label_path=labels[i])
        id_list = [-1] * len(current_imgs)


        save_picture_path = os.path.join(f'{args.out}', f'picture/frame{i+1}/')
        if not os.path.exists(save_picture_path):
            os.mkdir(save_picture_path)
            
        text_save_path = os.path.join(save_picture_path, f'frame{i+1}_dist_log.txt')
        f = open(text_save_path, 'w')

        for j in range(len(current_imgs)):
            img = transform(current_imgs[j])

            #get feature
            feature = extracter(torch.unsqueeze(img,0))
            current_feature.append(torch.squeeze(feature))
            
            #In case of need, record all information.
            if args.detail == 'on':
                if pre_feature:
                    #compute dist between feature
                    dist_matrix = torch.squeeze(compute_dist_rect(feature, pre_feature))
                    sorted_dist_matrix = torch.argsort(torch.squeeze(dist_matrix), descending=True)
                    output = []
                    output.append(current_imgs[j])
                    if len(pre_feature) == 1:
                        output.append(pre_imgs[sorted_dist_matrix])
                        f.write(f'{dist_matrix:5.4f}, ')
                    else:
                        for index in range(len(sorted_dist_matrix)):
                            output.append(pre_imgs[sorted_dist_matrix[index]])
                            f.write(f'{dist_matrix[sorted_dist_matrix[index]]:5.4f}, ')
                    f.write('\n')

                    path = os.path.join(save_picture_path, f'{j}.jpg')
                    save_image(output, path)

        #Match with the ID from previous frame
        if pre_feature:
            dist_matrix = compute_dist_rect(current_feature, pre_feature)
            if dist_matrix.dim() > 2:
                dist_matrix = torch.squeeze(dist_matrix)

            match_pair = matcher.match(dist_matrix, len(id_list), len(pre_idlist))
            for key in range(len(id_list)):
                id_list[key] = match_pair[key]

        #assign id
        for x in range(len(id_list)):
            if id_list[x] == -1:
                id_list[x] = ids
                ids += 1
            else:
                id_list[x] = pre_idlist[id_list[x]]

        #draw bounding box
        image = cv2.imread(imgs[i])
        for n in range(len(info_list)):
            color = palette.get_color(id_list[n])
            cv2.rectangle(image, (info_list[n][0], info_list[n][1]), (info_list[n][2], info_list[n][3]), color, 2)
            cv2.putText(image, text=str(id_list[n]), org=(info_list[n][0], info_list[n][1] - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color, thickness=3)
    
        out.write(image)
        pre_idlist = id_list
        pre_feature = current_feature
        pre_imgs = current_imgs
    out.release()