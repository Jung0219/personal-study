# model training pipeline

"""
make a simple linear regression model using pytorch only
these are the syntaxes you'll be using to train models
1. criterion (loss function)
    a. you first call the class to create an object. (in this case, nn.MSELoss())
    b. then you have to actually calculate the loss using it
        i. loss = criterion(y_predicted, y_true)
    c. then, to compute the gradients, you do loss.backward()
2. optimizer
    a. once the criterion done its work, you call the optimizer to make change to the weight and bias (or whatever parameters)
    b. just like criterion, you initialize it before the training loop
        i. torch.optim.SGD(model.parameters(), lr=learning_rate)
        ii. SGD is stochastic gradient descent (popular algorithm for gradient calculation)
        iii. PARAMETERS ARE WHAT YOU WANT THE MACHINE TO LEARN AND CHANGE
            1. this is obtained from the model, however you designed it.
        iv. then feed the learning rate
    c. after initializing the optimizer, call it in the loop
        i. optimizer.step() to do the optimization
    d. don't forget to zero the gradients
        i. optimizer.zero_grad()
"""

import torch
import torch.nn as nn

# prepare data
data = [[i] for i in range(10)]
X = torch.tensor(data, dtype=torch.float32)
Y = 2 * X


n_sample, n_features = X.shape
input_size, output_size = n_features, n_features

# our model to be trained
# this part will be more complicated once I learn how to create models with classes
model = nn.Linear(input_size, output_size)


"""---------------------------------------------------the training part----------------------------------------------------------------"""

# create a loss function object
criterion = nn.MSELoss()

# create an optimizer object
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)


for i in range(1000):
    prediction = model(X)  # get the prediction using the model
    loss = criterion(prediction, Y)  # call the loss function
    loss.backward()  # calculate gradient
    optimizer.step()  # apply optimization
    optimizer.zero_grad()  # reset the gradient

for name, param in model.named_parameters():
    print(f"{name} = {param.item()}, loss = {loss}")
