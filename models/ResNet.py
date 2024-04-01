import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()

        res_channel = out_channels // 4
        stride = 1

        #輸入通道數和輸出通道數不一致，下採樣
        self.projection = in_channels != out_channels 
        if self.projection:
            self.p = Conv(in_channels, out_channels, 1, 2, 0)
            stride = 2
            res_channel = in_channels // 2
        
        #第一層resblock
        if first:
            self.p = Conv(in_channels, out_channels, 1, 1, 0)
            stride = 1
            res_channel = in_channels
        
        self.conv1 = Conv(in_channels, res_channel, 1, 1, 0)
        self.conv2 = Conv(res_channel, res_channel, 3, stride, 1)
        self.conv3 = Conv(res_channel, out_channels, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.relu(self.conv1(x))
        f = self.relu(self.conv2(f))
        f = self.conv3(f)

        if self.projection:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h


class PretrainedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # out_features = [256, 512, 1024, 2048]
        self.head = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.head.fc = nn.Identity()

        self.bn = nn.BatchNorm1d(2048)
    def forward(self, x):
        x = self.head(x)
        x0 = self.bn(x)
        return x0
    
class ResNet(nn.Module):
    def __init__(self, no_block, in_channels=3):
        super().__init__()
        out_features = [256, 512, 1024, 2048]
        
        self.blocks = nn.ModuleList([ResBlock(64, 256, True)])

        for i in range(len(out_features)):
            if i > 0:
                self.blocks.append(ResBlock(out_features[i-1], out_features[i]))
            for _ in range(no_block[i]-1):
                self.blocks.append(ResBlock(out_features[i], out_features[i]))

        self.conv1 = Conv(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_features[-1])
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)

        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = self.bn(x)
        embedding = torch.flatten(x, 1)
        return embedding
    

    
if __name__ == '__main__':

    model = ResNet([3, 4, 23, 3])
    print(model)

    

    