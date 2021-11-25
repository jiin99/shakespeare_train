# import some packages you need here
import os
import unidecode
import string
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np


class Shakespeare(Dataset):
    def __init__(self, input_file, chunck_size=30):
        data = open(input_file).read().strip()
        char = list(sorted(set(data)))
        
        ix_to_char = {i: ch for i, ch in enumerate(char)}
        char_to_ix = {ch: i for i, ch in enumerate(char)}
        data = [char_to_ix[c] for c in data]
        data = [data[i : i+chunck_size] for i in range(0,len(data)-chunck_size)]  
        
        if len(data[-1]) < chunck_size :
            data = data[:-1] 

        self.data = np.array(data)
        self.chunck_size = chunck_size

    def __len__(self):

        return len(self.data)-1
        
    def __getitem__(self, index):

        x = self.data[index,:]  
        y = self.data[index + 1,:]  
        
        return x, y

if __name__ == '__main__':
    dataset = Shakespeare(r'./shakespeare_train.txt')
    tr_loader = DataLoader(dataset, batch_size = 1, shuffle = False)
    for i in range(len(tr_loader)):
        x,y = next(iter(tr_loader))
        import ipdb; ipdb.set_trace()
