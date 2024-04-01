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

def compute_dist(x):
    
    x_len= len(x)
    dist_matrix = torch.empty((x_len, x_len))
    for i in range(x_len):
        for j in range(i, x_len):    

            dist_matrix[i][j] = dist_matrix[j][i] = F.cosine_similarity(x[i], x[j], dim=0)

    return dist_matrix

if __name__ == '__main__':
    x = torch.randint(1,8,(12,4))
    dist_matrix = compute_dist(x, 0)
    print(f'dist matrix : {dist_matrix}')