# activation functions

"""
1. Non-linear transformation
    a. activation functions apply non-linear transformation to data before sending it to nodes
    b. functions applied to nodes that are  non linear
    c. linear functions only has its limits
2. Types of activation functions
    a. Sigmoid 
        i. output between 0 and 1
    b. hyperbolic tangent
        i. output between -1 and 1 (basically similar to sigmoid with range extended to -1)
        ii. f(x) = 2 / (1 + np.exp(-2 * x)) - 1
    c. rectified linear unit (ReLU)
        i. output between 0 and infinity
        ii. max(0, x)
            1. return 0 if x < 0 else x
    d. Softmax
        i. outputs a list, which sums up to 1 
        ii. e.g., (0.3, 0.2, 0.5)
        iii. often used for multiclass classification, probability distribution
    e. Leaky ReLU
        i. altered version of relu
        ii. return a * x if x < 0 else x
            alpha is a small number like 0.01
        iii. has advantage over normal relu by not always having 0 as an output
"""
