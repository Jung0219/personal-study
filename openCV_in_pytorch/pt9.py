# data transformation
"""
1. pytorch built-in transformations
    a. pytorch has built-in transformation functions applicable to a variety of datatypes
    b. e.g., adding padding to images, linear transformations on tensors, operations on ndarrays, etc
    c. also possible to write a lambda function for our own (making a class)
2. custom class
    a. when creating a custom transformation class, add __call__ 
3. applying multiple transformations
    a. torchvision.trnasform.Compose([trans1, trans2])
"""

import torch
import torchvision
from sklearn.datasets import load_wine
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math

data = load_wine()
data = pd.DataFrame(data.data)
print(data)

# create a class --> of batches


class WineDataset(Dataset):
    def __init__(self, transform=None):
        # xy = torch.tensor(pd.DataFrame(load_wine().data))
        self.x = pd.DataFrame(load_wine().data).values
        self.y = pd.DataFrame(load_wine().target).values
        self.n_samples = self.x.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        retval = self.x[index], self.y[index]
        if self.transform:
            retval = self.transform(retval)
        return retval

    def __len__(self):
        return self.n_samples


class ToTensor():
    def __call__(self, sample):
        features, labels = sample
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


class DivisionTransform():
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        features, labels = sample
        return features / self.factor, labels


composed = torchvision.transforms.Compose([ToTensor(), DivisionTransform(5)])

dataset = WineDataset(transform=composed)
print(dataset[0])
dataloader = DataLoader(dataset)
iterator = iter(dataloader)
iterator = next(iterator)
print(iterator)
