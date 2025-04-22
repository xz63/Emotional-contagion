import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import torch
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class AUDataset(Dataset):   
    def __init__(self, X,Y, chunk_length = 1, n_samples = 1):
        self.X = torch.tensor(X, dtype=torch.double)
        self.Y = torch.tensor(X, dtype=torch.double)
        self.n_samples = n_samples
        self.chunk_length = chunk_length

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        index1 = index * math.ceil(self.chunk_length)
        return (self.X[index1 : index1 + self.chunk_length, 0:17] ,self.Y[index1 : index1 + self.chunk_length, 0:17])

