import numpy as np


def emailFeatures(word_indices):
    """takes in a word_indices vector and
    produces a feature vector from the word indices.
    """

    # Total number of words in the dictionary
    n = 1899

    x = np.zeros(n) # Add the 0th position...
    for word_idx in word_indices:
        x[word_idx] = 1

    return x
