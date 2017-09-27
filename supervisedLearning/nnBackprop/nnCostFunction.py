import numpy as np
import sys
sys.path.append("../")
from nnMultiClass.mlCalcs import sigmoid
from sigmoidGradient import sigmoidGradient

def feed_forward(t1, t2, nn_params, input_sz, hidden_sz, num_labels, X, y):
    m, _ = X.shape
    p = np.zeros(m)
    a1_matrix = np.concatenate((np.ones((m, 1)), X), axis=1).T
    a2_matrix = np.concatenate((np.ones((1, m)), sigmoid(t1 @ a1_matrix)), axis=0)
    return sigmoid(t2 @ a2_matrix)

def reg_term(t1, t2, m, Lambda):
    # t[:, 1] isn't regularized (bias term parameters)
    ret = np.sum(t1[:, 1:]**2) + np.sum(t2[:, 1:]**2)
    return Lambda*ret/(2*m)

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, Lambda):

    """ computes the cost and gradient of the neural network. The
        parameters for the neural network are "unrolled" into the vector
        nn_params and need to be converted back into the weight matrices.

        The returned parameter grad should be a "unrolled" vector of the
        partial derivatives of the neural network.
    """
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1), order='F').copy()

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, (hidden_layer_size + 1)), order='F').copy()
    ff_result = feed_forward(Theta1, Theta2, nn_params, input_layer_size,
                 hidden_layer_size, num_labels, X, y).T


    m, _ = X.shape
    y_matrix = np.array(
                [np.arange(10) == y[i] - 1 for i in range(m)],
                dtype=int
               )

    J = np.sum(
            np.sum(
                -y_matrix*np.log(ff_result)
                -(1 - y_matrix)*np.log(1 - ff_result)
            )
    )
    J /= m
    J += reg_term(Theta1, Theta2, m, Lambda)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #
    # =========================================================================

    # Unroll gradient
    #grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))
    grad = np.zeros(m)
    return J, grad
