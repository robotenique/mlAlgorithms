import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""
    g = (1 + np.exp(-z))**-1
    return g

def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """
    # Useful values
    m, _ = X.shape
    p = np.zeros(m)
    # Important: The BIAS column goes FIRST! >:)
    # Create the input layer matrix, adding the '1' bias feature
    a1_matrix = np.concatenate((np.ones((m, 1)), X), axis=1).T
    """ Create the hidden layer, multiplying Theta1 by the a1, then taking
        the sigmoid of each entry. By this time, we have an n x m matrix,
        where each column is the hidden layer z² with respect to each example.
        Then, we add in the axis 0 a row of ones, in fact, this adds the BIAS
        feature to the top of each hidden layer.
    """
    a2_matrix = np.concatenate((np.ones((1, m)), sigmoid(Theta1 @ a1_matrix)), axis=0)
    """ We generate the output layer by multiplying Theta2 by the a2, then taking
        the sigmoid of every entry. With this operation, all the columns in the
        matrix are the hx of each example. To get the prediction, we get the index
        of each column with the maximum value, for every column. So, we need to get
        maximum value by the axis 0.
    """
    p = np.argmax(sigmoid(Theta2 @ a2_matrix), axis=0)

    return p + 1  # add 1 to offset index of maximum in A row
