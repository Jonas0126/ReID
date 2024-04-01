import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

class tripletLoss(nn.Module):
    def __init__(self, feature_set, pic_num=4, margin=1):
        super().__init__()
        self.row = len(feature_set)
        self.pic_num = pic_num
        self.margin =margin
        
    def forward(self, x):
        
        dist_matrix = compute_dist(x)
        
        triplet_loss = 0
        non_zero_num = 0
        for i in range(self.row):
            id = i // self.pic_num

            for j in range(i, self.row):
                if id == j//self.pic_num:
                    loss = 1 - dist_matrix[i, j]
                    if loss != 0:
                        non_zero_num += 1
                    triplet_loss += loss
                else:
                    loss = max(0, dist_matrix[i][j]-self.margin)
                    if loss != 0:
                        non_zero_num += 1
                    triplet_loss += loss
            
        triplet_loss /= non_zero_num
        if triplet_loss == 0:
            print(f'loss is 0')
            return torch.tensor(0.0,requires_grad=True)
        '''
        for i in range(self.row):

            id = i // self.pic_num
            positive_mask = torch.zeros(self.row).to('cpu')
            positive_mask[id*self.pic_num : id*self.pic_num+self.pic_num] = 1

            negative_mask = torch.ones(self.row).to('cpu')
            negative_mask -= positive_mask

            positive = (dist_matrix[i].mul(positive_mask))[id*self.pic_num : id*self.pic_num+self.pic_num].min()
            negative = dist_matrix[i].mul(negative_mask).max()

            loss = max((positive - negative + self.margin), 0)
            # triplet_loss.append(loss)
            triplet_loss += loss
        if triplet_loss == 0:
            print(f'loss is 0')
            return torch.tensor(0.0,requires_grad=True)
        '''
        return triplet_loss


#compute cos dist
def compute_dist(x):
    
    x_len= len(x)
    dist_matrix = torch.empty((x_len, x_len))
    for i in range(x_len):
        for j in range(i, x_len):    

            dist_matrix[i][j] = dist_matrix[j][i] = F.cosine_similarity(x[i], x[j], dim=0)

    return dist_matrix

if __name__ == '__main__':
    x = torch.rand((8,4)).to('cuda')
    print(f'x = {x}')

    dist_m = compute_dist(x)
    print(f'dist_m = {dist_m}')
    positive_mask = torch.zeros(8)
    positive_mask[0 : 2] = 1
    print(f'positive mask : {positive_mask}')
    positive = (dist_m[0].mul(positive_mask))
    print(f'positive : {positive}')
    positive = positive[0:2].min()
    print(f'positive : {positive}')
    triploss = tripletLoss(x, 2)
    print(triploss(x))