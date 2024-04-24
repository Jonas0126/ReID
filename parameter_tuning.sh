#! /bin/bash

lr=(0.0001 0.00001)
margin=(0)
img_num=(8 16)
width=(256)
batch=(32 64)
name=0
for ((i=0; i < ${#lr[@]}; i++))
do
    for ((j=0; j < ${#margin[@]}; j++))
    do
        for ((k=0; k < ${#width[@]}; k++))
        do
            for ((n=0; n < ${#batch[@]}; n++))
            do
                for ((q=0; q < ${#img_num[@]}; q++))
                do
                    echo train_SPloss.py -i datasets/VeRIWild/train/images/ -ti datasets/VeRIWild/valid/images/ -s trained_result/resnet_SP_256/ -lr ${lr[i]} -w ${width[k]} -b ${batch[n]} --img_num ${img_num[q]} --model_name resnet_SP_v$name 
                    python train_SPloss.py -i datasets/VeRIWild/train/images/ -ti datasets/VeRIWild/valid/images/ -s trained_result/resnet_SP_256/ -lr ${lr[i]} -w ${width[k]} -b ${batch[n]} --img_num ${img_num[q]} --model_name resnet_SP_v$name 
                    ((name++))
                done
            done
        done
    done

done       