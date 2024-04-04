


def top_k(k, dist_matrix, label_list, knn_idx):
    acc = 0
    start = 1

    for i in range(len(dist_matrix)):
        target = int(label_list[i])
        label_count = dict()   
        for j in range(1,k+1):
            label = int(label_list[knn_idx[i][j]])
            if label not in label_count:                    
                label_count[label] = 1
            else:
                label_count[label] += 1
        predict = max(label_count, key=label_count.get)

        if predict == target:
            acc += 1
    return acc/len(dist_matrix)

def top_k_test(k, dist_matrix, query_label, gallery_label_list, knn_idx):
    acc = 0
    target = int(query_label)
    label_count = dict()
    for j in range(0,k):
        label = int(gallery_label_list[knn_idx[j]])
        if label not in label_count:                    
            label_count[label] = 1
        else:
            label_count[label] += 1
    predict = max(label_count, key=label_count.get)
    if predict == target:
            acc += 1
    return acc

    for i in range(len(dist_matrix)):
        target = int(query_label_list[i])
        label_count = dict()   
        for j in range(0,k):
            label = int(gallery_label_list[knn_idx[j]])
            if label not in label_count:                    
                label_count[label] = 1
            else:
                label_count[label] += 1
        predict = max(label_count, key=label_count.get)

        if predict == target:
            acc += 1
    return acc/len(dist_matrix)