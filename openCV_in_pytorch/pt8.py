# batch training
# dataset and dataloader
"""
1. epoch
    a. epoch = the entire training dataset
2. batch training
    a. dividing the entire dataset into smaller subsets--called batches--and training the model with it.
    b. after each training with batches, the weights are updated.
    c. iteration = number of batches
3. gradient descent with batches
    a. because gradient calculation for every input is very expensive, so instead we do it with respet to the batch and
        not the individual data
4. features
    a. independent variables
5. labels
    a. dependent variables
6. DataLoader
    a. dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
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
    def __init__(self):
        # xy = torch.tensor(pd.DataFrame(load_wine().data))
        self.x = torch.tensor(pd.DataFrame(
            load_wine().data).values, dtype=torch.float32)
        self.y = torch.tensor(pd.DataFrame(
            load_wine().target).values, dtype=torch.float32)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def get_features(self, index):
        return self.x[index]

    def get_labels(self, index):
        return self.y[index]

    def __len__(self):
        return self.n_samples


n_epochs = 3
batch_size = 5
n_iterations = math.ceil(len(data) / batch_size)

data = WineDataset()
dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
"""
dataIter = iter(dataloader)
d = next(dataIter)
features, labels = d
print(features, labels)
"""


for epoch in range(n_epochs):
    for i, (features, labels) in enumerate(dataloader):
        # the training part
        print(features, labels)
        print(i)
