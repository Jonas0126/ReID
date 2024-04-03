import torch
import torch.nn.functional as F
from tqdm import tqdm
'''
#compute eucliden dist
def compute_dist(x, e=1e-12):
    x_len= len(x)
    x_sqr = torch.pow(x, 2).sum(1, keepdim=True).expand(x_len, x_len)
    

    dist = x_sqr + x_sqr.t()
    dist.addmm_(1, -2, x, x.t())
    dist = dist.clamp(min=e).sqrt()

    return dist
'''

def compute_dist_sqr(x):
    
    x_len= len(x)
    dist_matrix = torch.empty((x_len, x_len))
    for i in range(x_len):
        for j in range(i, x_len):    

            dist_matrix[i][j] = dist_matrix[j][i] = F.cosine_similarity(x[i], x[j], dim=0)

    return dist_matrix

def compute_dist_rect(x, y):
    y_len = len(y)
    x_len= len(x)
    dist_matrix = torch.empty((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):    

            dist_matrix[i][j] = F.cosine_similarity(x[i], y[j], dim=0)

    return dist_matrix

if __name__ == '__main__':
    x = torch.rand(12,4)
    y = torch.rand(20,4)
    dist_matrix = compute_dist_rect(x, y)
    print(f'dist matrix : {dist_matrix.shape}')