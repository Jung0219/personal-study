# gradient
"""
1. gradient == derivative
2. loss function
    a. loss function is a function that becomes the guidance for the model's performance
        basd on the value, the model takes into account how well it predicted
3. forward and backward paths
    a. forward path is simply calculating the function as it is
    b. backward path is getting the gradient (i.e., the derivative)
4. for every mathematical operation, pytorch will calculate the gradient
5. Important Deep Learning concept = Back Propagation
    a. this is going backwards in the function, in other words calculating the derivative 
    b. we do this so we can find the optimal parameter for models (like in the case of gradient descent)
6. given data, you do whatever operation on it, and at the final step, you calculate the "gradient", or the 
    backward derivative
    a. torch.tensor(values, required_grad=True)
        i. must have be initialized to enable gradient calculation
    b. result.backward()
        i. this gives the gradient of the result with respect to the original values
        ii. access using values.grad to get the gradient value
        iii. when calculating gradients, it uses "vector Jacobian product" which is a matrix multiplication of 
            partial derivatives and gradient vectors resulting in gradient values
        iv. for this reason, when accessing gradients, if it's not already a scalar value, you must give it a matching 
            tensor, i.e., the gradient vectors of the same size
        v. whenever you call the backward function, the gradients stack up. each calculation will be accumulated into the 
            grad attribute of the data
            1. empty the grad attribute with data.grad.zero_()
7. to stop tracking gradients
    a. data.requires_grad_(False)
        i. changes the attribute of the data
    b. data.detach()
        i. copies the original but without tracking gradients
    c. with torch.no_grad(): ...
    d. zero the gradient
        data.grad.zero_()
8. Weight (internal parameter)
    a. strengh/probability/likelihood of one node having some sort of connection with another
    b. initially randomly assigned, but the machine "chooses" the weights
9. New parameter = old parameter - (gradient * learning rate)


"""

import torch

x = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)

y = x + 2
z = y * y * 2
# z = z.mean()

z.backward(torch.tensor([1, 2, 3]))

new = x.detach()

with torch.no_grad():
    w = x - 2

weights = torch.tensor([2, 3, 4], dtype=torch.float32, requires_grad=True)
for i in range(5):
    model_output = (weights * 3).mean()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()
