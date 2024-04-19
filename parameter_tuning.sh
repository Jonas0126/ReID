#! /bin/bash

lr=(0.001)
margin=(0.3 0.2 0.1 0.0 -0.1 -0.2 -0.3)
width=(300)
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
                
                echo train.py -i datasets/VeRIWild/train/images/ -ti datasets/VeRIWild/valid/images/ -s yrained_result/resnet_300/ -m ${margin[j]} -lr ${lr[i]} -w ${width[k]} -b ${batch[n]} --model_name resnet_v$name 
                python train.py -i datasets/VeRIWild/train/images/ -ti datasets/VeRIWild/valid/images/ -s trained_result/resnet_300/ -m ${margin[j]} -lr ${lr[i]} -w ${width[k]} -b ${batch[n]} --model_name resnet_v$name 

                ((name++))
            done
        done
    done

done       