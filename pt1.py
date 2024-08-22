# pytorch tutorial
"""
1. tensor (the basic datatype)
    a. tensor is a scalar, vector, matrix, and everything. It's simply number in dimensions. 
        scalar would be 1d tensor, vector 2d, matrix 3d, and so on.
    b. and this is like a building block I guess
2. numbers
    a. torch.empty([dimensions], dtype=torch.dtype)
        i. initializes a tensor with whatever values
        ii. you can assign specific data types like in numpy
            1. torch.floa, torch.double, torch.half, torch.int8, torch.uint8, torch.int16,32,64
    b. torch.rand([dimensions], dtype=torch.dtype)
        i. initializes a tensor with random values
    c. torch.zeros([dimensions], dtype=torch.dtype)
    d. torch.ones([dimensions], dtype=torch.dtype)
        i. the same as numpy
    e. torch.fill_(value, dtype=torch.dtype)
        i. fill the tensor with a value
    f. find datatypes with tensor.dtype
        i. default is float32 and int64
3. convert list into a pytorch tensor
    a. the same as numpy array! 
    b. torch.tensor([list])
4. numerical operations
    a. +, -, *, / 
        i. matrix additino, subtraction, multiplication, division
    b. achieve the same result
        i. a += b, a = a + b, a.add_(b) a = torch.add(a, b)
        ii. same for subtraction, multiplication, division
    c. // for floor division
    d. % or torch.remainder(a, b) for remainder
    e. ** or torch.pow(a, b) for power
    f. torch.sum(tensor, dim=dimension)
        i. sum of elements in a tensor specified by dimension
    g. slicing tensors like you would do with pandas
        i. tensor[start:stop:step]
5. reshaping
    a. torch.reshape(tensor, dimension)
6. with numpy array
    a. when making tensor into np array or the other way around, when using cpu, they share the same memory location
        meaning that modification to one will affect the other identically
    b. numpy only handles cpu tensors
        i. if using gpu, we can't use it
"""
import torch

x = torch.rand(2, 2, 3, dtype=torch.float32)
y = torch.tensor([[1,1], [5,5]])

a = torch.empty(2, 2)
b = torch.empty(2, 2)
a.fill_(3)
b.fill_(5)

a += b

print(x)
print(x[0:0])  

