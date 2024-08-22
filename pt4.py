# Model Optimization with Autograd

"""
Start manually, then replace them with pytorch framework
Make a linear regression model from scratch, gradually replace the components with pytorch framework

1. construct model
2. construct loss and optimizer
3. training loop
    here's where the steps above go in
    a. foward 
    b. back propagation
    c. calculate gradient and adjust parameters (optimization)

1. pytorch loss function
    a. loss = torch.nn.MSEloss()
2. pytorch optimizer
    a. optimizer = torch.optim.SGD([variable we want to calculate gradient with respect to], lr=learning_rate)
    b. to run the optimizer, optimizer.step()
    c. you must still zero the gradients for every ste
        i. optimizer.zero_grad()
3. simple models are already designed (normally we have to design them ourselves)
    a. nn.Linear()
    b. this is on its own a linear regression model
    c. in neural network, you add hundreds of these --> that's why you need so much computational power

"""

import numpy as np
import torch
import torch.nn as nn

# y = w * x
# w = 3
x = torch.empty(10, 1, dtype=torch.float32)
for index, item in enumerate(x):
    x[index] = index

y = x * 3
w = torch.tensor(0, dtype=torch.float32, requires_grad=True)
learning_rate = 0.01


def model_forward(w, x):
    return w * x


n_samples, n_features = x.shape
input_size, output_size = n_features, n_features

model = nn.Linear(input_size, output_size)


# this is what we want to reduce as much as possible
# replace this function with pytorch's built-in loss function


def loss_function(predicted, actual):
    # 1/n ((predicted - actual) ** 2).sum()
    return torch.mean((predicted - actual) ** 2)


def back_propagate(x, y_pred, y_true):
    # derivative of the loss function
    # 1/n * (wx - y) ** 2 --> 1/n * 2x * (predicted - actual)
    # return the gradient
    return torch.mean(2 * x * (y_pred - y_true))


print("before training: {}".format(w))

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(20):
    prediction = model(x)
    loss = loss_function(prediction, y)
    loss.backward()
    gradient = back_propagate(x, prediction, y)

    optimizer.step()
    optimizer.zero_grad()

    [w, b] = model.parameters()

    print(f'w = {w[0][0].item():.3f}, loss = {loss:.8f}')

print("after training:{}".format(w))
