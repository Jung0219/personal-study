# Back Propagation --> It's nothing more than calculating derivatives backwards!

"""
Machine learning is all calculus
    1. Chain Rule
        a. for calculating the gradient of a complex model, especially the ones for neural networks, you utilize the chain rule
        b. say there are three operations being done from input x to input z
        c. dz/dx = dz/dy * dy/dx
    2. Local gradients
        a. in the case above, dz/dy would be the local gradient
        b. not the final, but the gradient you have to calculate in order to get the original gradient
    3. Steps
        a. foward path
        b. calculate the loss function
        c. back propagation
        d. get gradient
    4. Gradient calculation
        a. CALCULATE THE GRADIENT WITH RESPECT TO THE VALUES THAT WILL CHANGE
        b. things like x or y val are fixed. What the machine should be adjusting are weight and bias, hence take it with respect to w and b

"""

# simulating linear regression and calculating the gradient myself
# linear regression formula = y = wx + b
# data = [x = 5, y = 22]
# hence the correct weight would be 4 and bias 2
# random intial value for w, b = 3, 1
# step 1: implement the forward path
import torch 

def model(data, weight, bias):
    return data * weight + bias

def mse(predicted, actual):
    return 0.5 * (y_predict - y_actual) ** 2

x = torch.tensor(5, dtype=torch.float32)

y_actual = torch.tensor(22, dtype=torch.float32)
y_predict = torch.empty(1, dtype=torch.float32)

w = torch.tensor(3, dtype=torch.float32, requires_grad=True)
b = torch.tensor(1, dtype=torch.float32, requires_grad=True)

y_predict = model(x, w, b)

# step 2: compute the loss function (mean squared error)
error = mse(y_predict, y_actual) # (22 - 16) ** 2 = 36
print(error)
# step 3: back propagation
# taking derivative of the loss function


error.backward()
print(error, w.grad, b.grad)