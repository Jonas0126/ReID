import torch
import torch.nn as nn
import sys


class tripletLoss(nn.Module):
    def __init__(self, feature_set, pic_num=4, margin=1):
        super().__init__()
        self.row = len(feature_set)
        self.pic_num = pic_num
        self.margin =margin
        
    def forward(self, x):
        
        dist_matrix = compute_dist(x)
        
        triplet_loss = 0
        for i in range(self.row):
            
            id = i // self.pic_num
            positive_mask = torch.zeros(self.row).to('cuda')
            positive_mask[id*4:id*4+4] = 1
            
            negative_mask = torch.ones(self.row).to('cuda')
            negative_mask -= positive_mask
            
            positive = (dist_matrix[i].mul(positive_mask)).max()

            temp = dist_matrix[i].mul(negative_mask)
            try:
                negative = temp[temp > 0].min() #過濾0值
            except:
                print(f'dist between feature : {temp}')
            loss = max((positive - negative + self.margin), 0)
            # triplet_loss.append(loss)
            triplet_loss += loss

        return triplet_loss


#compute eucliden dist
def compute_dist(x):
    x_len= len(x)
    x_sqr = torch.pow(x, 2).sum(1, keepdim=True).expand(x_len, x_len)
    

    dist = x_sqr + x_sqr.t()
    dist.addmm(1, -2, x, x.t())
    dist = dist.clamp(min=1e-12).sqrt()

    return dist

if __name__ == '__main__':
    x = torch.rand((12,4)).to('cuda')
    print(f'x = {x}')

    triploss = tripletLoss(x)
    print(triploss(x))