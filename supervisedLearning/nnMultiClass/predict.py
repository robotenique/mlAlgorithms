import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""
    g = (1 + np.exp(-z))**-1
    return g

def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """
    # TODO: fix this calculation...
    # Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape
    p = np.zeros((m, 1))
    # Add ones to the X data matrix, constructing the input layer
    a1 = np.concatenate((X.T, np.ones((1, m)))).T
    #a1 = np.column_stack((np.ones((m, 1)), X))
    print(a1)
    for i in range(m): # For our m examples
        x = a1[i] # A picture of a number
        print(x.shape)
        # Hidden Layer
        a2 = np.concatenate((sigmoid(Theta1 @ x.T), [1]))
        # Output layer
        hx = sigmoid(Theta2 @ a2.T)
        # Get the highest probability
        p[i] = np.argmax(hx)

    return p + 1  # add 1 to offset index of maximum in A row
