import torch
import torch.nn.functional as F
import cv2
import numpy as np



class Matcher():
    def __init__(self, threshold):
        self.threshold = threshold
    def match(self, dist_matrix, row_num, col_num):
        match_pair = dict()
        matched = []
        for i in range(row_num):
            #get max value and index
            max_dist = dist_matrix.max()
            max_index = dist_matrix.argmax()
            row = (max_index // col_num).item() 
            col = (max_index % col_num).item()
            dist_matrix[row,:] = -2
            dist_matrix[:,col] = -2
            
            if max_dist == -2:
                break
            if max_dist >= self.threshold:
                match_pair[row] = col
            else:
                match_pair[row] = -1
            matched.append(row)

        for i in range(row_num):
            if i not in matched:
                match_pair[i] = -1
        return match_pair



if __name__ == '__main__':
    matrix = torch.tensor([[12, 21, 3],
                           [-4, -5, -6],
                           [7, 82, 9]])
    
    matcher = Matcher(0)
    match_pair = matcher.match(matrix)
    print(match_pair)
    print(match_pair[0])
    print(match_pair[1])
    print(match_pair[2])