


def top_k(k, dist_matrix, label_list, knn_idx, mode=0):
    acc = 0
    start = 1
    if mode==0:
        start = 1
    else:
        start = 0
    for i in range(len(dist_matrix)):
        target = int(label_list[i])
        label_count = dict()   
        for j in range(start,k+start):
            label = int(label_list[knn_idx[i][j]])
            if label not in label_count:                    
                label_count[label] = 1
            else:
                label_count[label] += 1
        predict = max(label_count, key=label_count.get)

        if predict == target:
            acc += 1
    return acc/len(dist_matrix)