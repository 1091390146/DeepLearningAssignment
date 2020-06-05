import time

import numpy as np

def sigmoid(x):
    """
     Compute sigmoid of x.

     Arguments:
     x -- A scalar

     Return:
     s -- sigmoid(x)
     """

    s = 1/(1 + np.exp(-x))

    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)

    return ds

def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)

    x = x/x_norm

    return x

def softmax(x):
    x_exp = np.exp(x)

    x_sum = np.sum(x_exp, axis=1, keepdims= True)

    s = x_exp/x_sum

    return s

def L1(yhat, y):

    loss = np.sum(np.abs(y - yhat))

    return loss

def L2(yhat, y):

    loss = np.dot((y - yhat),(y - yhat).T)

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))