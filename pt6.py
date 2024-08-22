# designing a model

"""
1. torch.nn
    a. base class for all pytorch neural network classes
    b. whenever you create a model, you're essentially creating a subclass of this base class
        i. WHICH IS WHY YOU SUPER INIT THIS MF
        ii. super() refers to its parent class, so the __init__ part calls the initialization of the parent class
2. initialization
    a. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self, n_input, n_output):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def forward(self, x):
        return self.linear(x)


model = SimpleNN(10, 10)

X, Y = datasets.make_regression(
    n_samples=100, n_features=1, noise=15, random_state=42)


X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32).reshape(100, -1)

print(X.shape, Y.shape)

"""
data = [[i] for i in range(10)]
X = torch.tensor(data, dtype=torch.float32)
Y = 2 * X
"""
# create a loss function object
criterion = nn.MSELoss()

# create an optimizer object
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for i in range(100):
    prediction = model.forward(X)  # get the prediction using the model
    loss = criterion(prediction, Y)  # call the loss function
    loss.backward()  # calculate gradient
    optimizer.step()  # apply optimization
    optimizer.zero_grad()  # reset the gradient

for name, param in model.named_parameters():
    print(f"{name} = {param.item()}, loss = {loss}")

w = model.weight.item()
b = model.bias.item()

x = np.linspace(-5, 5, 5)
y = w * x + b

plt.scatter(X, Y)
plt.plot(x, y, color="red")
plt.waitforbuttonpress()
plt.close("all")
