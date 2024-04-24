import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights




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
    
    
if __name__ == '__main__':

    feature_extractor = PretrainedResNet()
    input_ = torch.rand(2, 3,224,224)

    output_ = feature_extractor(input_)
    print(f'dot product before norm : {torch.dot(output_[0], output_[0])}') 
    output_len = torch.sqrt(torch.dot(output_[0], output_[0]))
    print(f'len of output : {output_len}')
    output_[0] /= output_len
    print(f'dot product after norm : {torch.dot(output_[0], output_[0])}') 
    output_len = torch.sqrt(torch.dot(output_[0], output_[0]))
    print(f'len of output : {output_len}')
    

    