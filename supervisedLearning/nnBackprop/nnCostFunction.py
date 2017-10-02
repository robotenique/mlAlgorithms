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
    z2_matrix = t1 @ a1_matrix
    return sigmoid(t2 @ a2_matrix), a1_matrix, a2_matrix, z2_matrix

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

    ff = feed_forward(Theta1, Theta2, nn_params, input_layer_size,
         hidden_layer_size, num_labels, X, y)
    ff_result = ff[0].T
    m, _ = X.shape
    val = lambda yi: yi - 1 if num_labels == 10 else yi
    y_matrix = np.array(
                [np.arange(num_labels) == val(y[i]) for i in range(m)],
                dtype=int
               )
    J = np.sum(
                -y_matrix*np.log(ff_result)
                -(1 - y_matrix)*np.log(1 - ff_result)
    )
    J /= m
    J += reg_term(Theta1, Theta2, m, Lambda)

    # Calculating the gradient
    grad = np.zeros(m)
    a3 = ff_result
    a1 = ff[1]
    a2 = ff[2]
    z2 = ff[3].T
    # Backpropagation
    err3 = a3 - y_matrix
    err2 = np.dot(err3, Theta2)[:, 1:]*sigmoidGradient(z2)
    acc2 = (a2@err3).T
    acc1 = (a1@err2).T
    derivative_1 = acc1/m
    derivative_2 = acc2/m
    derivative_1[:, 1:] += Lambda*Theta1[:, 1:]/m
    derivative_2[:, 1:] += Lambda*Theta2[:, 1:]/m
    grad = np.concatenate((derivative_1.T.ravel(), derivative_2.T.ravel()))

    return J, grad
