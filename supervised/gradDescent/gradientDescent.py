from computeCost import computeCost
import numpy as np


def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples
    for i in range(num_iters):
        hx = hxClosure(theta) # Create a closure with the current theta Value
        #Update each theta component simultaneously
        for i in range(theta.size):
            theta[i] -= alpha*partial_derivative(hx, X, y, i)
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

def partial_derivative(hx, X, Y, ith):
    '''
        Returns the partial_derivative in respect to the
        ith theta specified;
    '''
    temp = 0
    for i in range(Y.size):
        temp += (hx(X[i]) - Y[i])*X[i][ith]
    temp /= Y.size

    return temp

def hxClosure(thetaVector):
    def hx(Xi):
        return thetaVector.dot(Xi)
    return hx
