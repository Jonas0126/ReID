import os
import shutil
from tqdm import tqdm
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--image_dir', '-i',type=str)
parser.add_argument('--save_dir', '-s', type=str)
args = parser.parse_args()


if not os.path.exists(args.image_dir):
    print(f'{args.image_dir} does not exist.')
    exit()


if not os.path.exists(args.save_dir):
    print(f'{args.save_dir} does not exist.')
    exit()



id = 0

files = os.listdir(args.image_dir)

save_img_path = os.path.join(args.save_dir, "images")


if not os.path.isdir(save_img_path):
    os.mkdir(save_img_path)


for file in tqdm(files, dynamic_ncols=True):
    file_path = os.path.join(args.image_dir, file)
    images = os.listdir(file_path)
     
    i = 0
    save_path = os.path.join(save_img_path, f'{id}/')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for img in images:
        
        img_drc = os.path.join(save_path, f'{i}.jpg')
        img_src = os.path.join(file_path, img)
        
        shutil.copyfile(img_src,img_drc)
        i+=1



    id += 1



        
