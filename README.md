## 簡介
這個project用於訓練在REID中特徵擷取器的角色，MODEL使用Pretrained ResNet101，可以根據需求使用不同的資料集，資料集格式如下所示:
```
datasets
├── aicup
│   ├── test
│   │   └── images
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── valid
│       └── images
└── VeRIWild
    ├── test
    │   └── images
    ├── train
    │   ├── images
    │   └── label
    └── valid
        └── images
```
label資料夾為空，而images的格式如下所示:
```
images
├── 0
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   └── 6.jpg
├── 1
│   ├── 0.jpg
│   ├── 1.jpg
.
.
.
```
在VeRIWild上的測試結果能在rank 1達到89%準確率。
而在AICUP上的測試結果在rank 1僅能達到68%準確率。
## 使用方式
[dataset下載](https://drive.google.com/drive/folders/1MhmSHuEQSpRSL2NUNwYp7YwtVBDA2xSo?usp=sharing)

若想要訓練model，需要先在目錄中建立兩個資料夾分別為trained_result和test_result，訓練結果將存放至trained_result，測試結果則存放到test_result，可以使用以下指令：
```
python train_tripletloss.py -lr 0.001 −w 256 -e 50 -b 32 −m 0.3 --model_name ResNet101 --datasets VeRIWild
```
- -lr -> learning rate
- -w -> image width
- -e -> number of epoch
- -m -> margin
- --model_name -> file name saved
- --datasets -> dataset in used(VeRIWild or aicup)

訓練好的模型將存在trained_result/(--datasets的參數)/(--model_name的參數)\_(-w的參數)\_(-m的參數)/

ex. trained_result/VeRIWild/ResNet101_256_0.3/

測試訓練好的model則使用以下指令:
```
python test.py --model_parameter trained_result/VeRIWild/ResNet101_256_0.3/best.pt -n ResNet101_256_0.3 -w 256 --datasets VeRIWild
```
- --model_parameter -> trained model parameters location
- -n -> file name saved
- -w -> image width
- --datasets -> dataset in used(VeRIWild or aicup)
