from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

X, Y = datasets.make_classification(
    n_samples=100, n_features=20, n_classes=2, random_state=17)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)


model = LogisticRegressionModel(20)

criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=0.03)

n_epochs = 100
for i in range(n_epochs):
    predictions = model(x_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(loss)


def score(prediction, true):
    with torch.no_grad():
        correct = (prediction == true).sum().item()
        print(prediction == true)
        return (correct/len(true))


print(score(model(x_test).round(), y_test))
