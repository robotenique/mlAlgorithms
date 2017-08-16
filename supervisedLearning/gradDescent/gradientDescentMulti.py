from computeCostMulti import computeCostMulti
from gradientDescent import gradientDescent

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
    return theta, J_history
