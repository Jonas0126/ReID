#! /bin/bash

lr=(0.001)
margin=(0.3)
width=(224)
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
                
                echo fine_tuning.py -i datasets/aicup/aicup_reid/train/images/ -testimg datasets/aicup/aicup_reid/valid/images/ -s fine_tune_result/ --save_model_name test_V$name --fine_tuning_model trained_result/resnet_224/resnet_v0/best.pt -m ${margin[j]} -lr ${lr[i]} -w ${width[k]} -b ${batch[n]} -e 30
                python fine_tuning.py -i datasets/aicup/aicup_reid/train/images/ -testimg datasets/aicup/aicup_reid/valid/images/ -s fine_tune_result/ --save_model_name test_V$name --fine_tuning_model trained_result/resnet_224/resnet_v0/best.pt -m ${margin[j]} -lr ${lr[i]} -w ${width[k]} -b ${batch[n]} -e 30

                ((name++))
            done
        done
    done

done       