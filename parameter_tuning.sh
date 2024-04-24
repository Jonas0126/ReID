#! /bin/bash

lr=(0.001)
margin=(0.5 0.4 0.3 0.2 0.1 0.0)
img_num=()
width=(224 256 300)
batch=(32)
for ((i=0; i < ${#lr[@]}; i++))
do
    for ((j=0; j < ${#margin[@]}; j++))
    do
        for ((k=0; k < ${#width[@]}; k++))
        do
            for ((n=0; n < ${#batch[@]}; n++))
            do
                echo train_tripletloss.py -lr ${lr[i]} -w ${width[k]} -b ${batch[n]} -m ${margin[j]} --model_name ResNet101 --datasets VeRIWild
                python train_tripletloss.py -lr ${lr[i]} -w ${width[k]} -b ${batch[n]} -m ${margin[j]} --model_name ResNet101 --datasets VeRIWild
            done
        done
    done

done       