import numpy as np


def randInitializeWeights(L_in, L_out):
    """ Randomly initializes the weights of a layer with L_in
        incoming connections and L_out outgoing
        connections.

        Note that W should be set to a matrix of size(L_out, 1 + L_in)
        as the column row of W handles the "bias" terms
    """
    # A good estimate of the EPSILON could be sqrt(6)/s.sqrt(lin+lout)
    init_epsilon = 0.12 # hardcoded >:D
    # Add +1 for the bias unit
    return np.random.random((L_out, L_in + 1))*2*init_epsilon - init_epsilon
