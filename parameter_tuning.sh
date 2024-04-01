#! /bin/bash

lr=(0.001 0.0003 0.0001)
margin=(0.6 0.3 0 -0.2)
width=(224 256 300)
batch=(32)
name=0
for ((i=0; i < ${#lr[@]}; i++))
do
    for ((j=0; j < ${#margin[@]}; j++))
    do
        for ((k=0; k < ${#width[@]}; k++))
        do
            for ((n=0; n < ${#batch[@]}; n++))
            do
                
                echo train.py -i datasets/train/images/ -testimg datasets/valid/images/ -s result/ -m ${margin[j]} -lr ${lr[i]} -w ${width[k]} -b ${batch[n]} --model_name resnet_v$name --test_pic_num 10 --train_pic_num 4
                python train.py -i datasets/train/images/ -testimg datasets/valid/images/ -s result/ -m ${margin[j]} -lr ${lr[i]} -w ${width[k]} -b ${batch[n]} --model_name resnet_v$name --test_pic_num 10 --train_pic_num 4

                ((name++))
            done
        done
    done

done    