import torch



#compute eucliden dist
def compute_dist(x, e=1e-12):
    x_len= len(x)
    x_sqr = torch.pow(x, 2).sum(1, keepdim=True).expand(x_len, x_len)
    

    dist = x_sqr + x_sqr.t()
    dist.addmm_(1, -2, x, x.t())
    dist = dist.clamp(min=e).sqrt()

    return dist


if __name__ == '__main__':
    x = torch.randint(1,8,(12,4))
    dist_matrix = compute_dist(x, 0)
    print(f'dist matrix : {dist_matrix}')