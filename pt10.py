# Softmax and cross entropy
"""
1. softmax
    a. pytorch has its built-in softmax function
    b. softmax = nn.Softmax(dimensions)
        i. here, the dimensions have to be always one lower than the input's dimensions
    c. torch.softmax(input_features)
2. cross entropy
    a. this is a commonly used loss function for classification problems
    b. this measures the difference between two probability distributions
    c. pytorch built-in function is nn.CrossEntropyLoss()
    d. WHEN YOU USE CROSS ENTROPY, SOFTMAX IS ALREADY APPLIED
    e. it expects raw logits as input, without any operation done on it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math


class ClassificationModel(nn.Module):
    def __init__(self, n_features, n_hidden_layers1, n_hidden_layers2, n_labels):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden_layers1)
        self.fc2 = nn.Linear(n_hidden_layers1, n_hidden_layers2)
        self.fc3 = nn.Linear(n_hidden_layers2, n_labels)

    def forward(self, data):
        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))
        data = self.fc3(data)
        # logits = F.softmax(data, dim=1)
        return data


class WineDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = pd.DataFrame(x).values
        self.y = pd.DataFrame(y).values
        self.n_features = self.x.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        features, labels = self.x[index], self.y[index]
        if self.transform:
            features, labels = self.transform(features), self.transform(labels)
        return features, torch.tensor(labels, dtype=torch.long).squeeze()

    def __len__(self):
        return self.n_features


class ToTensor():
    def __call__(self, data):
        return torch.tensor(data, dtype=torch.float32)


model = ClassificationModel(13, 32, 16, 3)
n_epochs = 30

x_train, x_test, y_train, y_test = train_test_split(
    load_wine().data, load_wine().target, train_size=0.8)

train_data = WineDataset(x_train, y_train, transform=ToTensor())

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)


criterion = nn.CrossEntropyLoss()  # wants long
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(n_epochs):
    for features, labels in train_dataloader:
        # forward
        predictions = model.forward(features)
        loss = criterion(predictions, labels)
        # print(f"prediction:{[max(i)for i in predictions][0]:.5f}")
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss:{loss:.4f}")

test_data = WineDataset(x_test, y_test, ToTensor())
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)


for features, labels in test_dataloader:
    prediction = model.forward(features)

    print(f"prediction:{[max(i)
          for i in prediction][0]:.5f}, true value: {labels}")
