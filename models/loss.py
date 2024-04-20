import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
import time
class SP_loss(nn.Module):
    def __init__(self, feature_set, T=0.04):
        super().__init__()
        self.row = len(feature_set) 
        self.classes = self.row // 8
        self.pic_num = 4   #number of instance per identity
        self.T = T
        self.device = 'cuda'

    def forward(self, x):
        
        dot_matrix = dot_product(x)

        loss = 0

        negative_S = self.compute_negative_S(dot_matrix).clone().detach().requires_grad_(True)

        hard_positive_S, least_hard_positive_S = self.compute_hard_positive_S(dot_matrix)

        hard_positive_S = hard_positive_S.clone().detach().requires_grad_(True)
        least_hard_positive_S = least_hard_positive_S.clone().detach().requires_grad_(True)


        # hard_positive_S = torch.tensor(self.compute_hard_positive_S())
        # least_hard_positive_S = torch.tensor(self.compute_least_hard_positive_S())
        alpha = torch.empty((0), device=self.device, requires_grad=True)

        for i in range(len(hard_positive_S)):
            if hard_positive_S[i] >= 0:
                H_mean = (2 * least_hard_positive_S[i] * hard_positive_S[i]) / (hard_positive_S[i] + least_hard_positive_S[i])
                alpha = torch.cat((alpha, torch.unsqueeze(H_mean, dim=0).to(self.device)))
            else:
                alpha = torch.cat((alpha, torch.unsqueeze(torch.tensor(0.0, requires_grad=True, device=self.device), dim=0)))

        for i in range(len(hard_positive_S)):
            temp_1 = alpha[i] * hard_positive_S[i]           
            temp_2 = ((1 - alpha[i]) * least_hard_positive_S[i])
            # temp_1 + temp_2 = weighted positive similarity term
            # temp_3 is the exponential term of AdaSP loss   
            temp_3 = torch.exp((negative_S[i] - (temp_1 + temp_2)) / self.T)
            
            loss = loss + torch.log(1 + temp_3) 

        del x, alpha, negative_S, least_hard_positive_S
        torch.cuda.empty_cache()
        return loss / len(hard_positive_S)

    # def compute_least_hard_positive_S(self):
    #     least_positive_S = torch.empty((0), device='cpu', requires_grad=True)
    #     least_positive_similarity = 0
    #     for i in range(self.row):
    #         class_i = i // self.pic_num
    #         temp = 0
    #         for j in range(self.row):
    #             if j // self.pic_num == class_i:
    #                 temp = temp + torch.exp(-1 * self.dot_matrix[i][j] / self.T)
    #         least_positive_similarity = least_positive_similarity + 1/temp

    #         if (i+1) // self.pic_num != class_i:
    #             least_positive_S = torch.cat((least_positive_S, torch.unsqueeze(self.T * torch.log(least_positive_similarity), dim=0).to('cpu')))
    #             least_positive_similarity = 0
    #     return least_positive_S

    def compute_hard_positive_S(self, dot_matrix):
        positive_S = torch.empty((0), device=self.device, requires_grad=True)
        positive_similarity = 0

        least_positive_S = torch.empty((0), device=self.device, requires_grad=True)
        least_positive_similarity = 0
        for i in range(self.row):
            class_i = i // self.pic_num
            temp = 0
            for j in range(self.row):
                if j // self.pic_num == class_i:
                    positive_similarity = positive_similarity + torch.exp(-1 * dot_matrix[i][j] / self.T)
                    temp = temp + torch.exp(-1 * dot_matrix[i][j] / self.T)

            least_positive_similarity = least_positive_similarity + (1 / temp)

            if (i+1) // self.pic_num != class_i:
                least_positive_S = torch.cat((least_positive_S, torch.unsqueeze(self.T * torch.log(least_positive_similarity), dim=0).to(self.device)))
                least_positive_similarity = 0
                positive_S = torch.cat((positive_S, torch.unsqueeze(-1 * self.T * torch.log(positive_similarity), dim=0).to(self.device)))
                positive_similarity = 0

        return positive_S, least_positive_S
    
    def compute_negative_S(self, dot_matrix):
        negative_S = torch.empty((0), device=self.device, requires_grad=True)
        negetive_similarity = 0

        for i in range(self.row):
            class_i = i // self.pic_num

            for j in range(self.row):
                if j // self.pic_num != class_i:
                    negetive_similarity = negetive_similarity + torch.exp(dot_matrix[i][j] / self.T)

            if (i+1) // self.pic_num != class_i:
                negative_S = torch.cat((negative_S, torch.unsqueeze(self.T * torch.log(negetive_similarity), dim=0).to(self.device)))
                negetive_similarity = 0

        return negative_S
    
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


def dot_product(x):
    y = torch.transpose(x, 0, 1)
    dot_matrix = torch.mm(x, y)

    return dot_matrix

if __name__ == '__main__':
    x = torch.rand(16,4).to('cuda')
    #normalized
    for i in range(len(x)):
        length = torch.sqrt(torch.dot(x[i], x[i]))
        x[i] /= length
    
    for i in range(len(x)):
        print(f'x[{i}] : {x[i]}')
        print(f'len of x[{i}] : {torch.dot(x[i], x[i])}')
    

    AdaSp = SP_loss(x)
    print(AdaSp())
    