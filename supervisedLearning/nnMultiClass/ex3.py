import sys
import scipy.io
import numpy as np
from matplotlib import use
use('TkAgg')
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
from displayData import displayData

# 20x20 Input Images of Digits
input_layer_size = 400
# 10 labels, from 1 to 10 ("0" is mapped to 10)
num_labels = 10

#  =========== Part 1: Loading and Visualizing Data =============
# This is a dataset with handwritten digits. Let's visualize it
print('Loading and Visualizing Data ...')
# training data stored in arrays X, y (pre-defined)
data = scipy.io.loadmat('ex3data1.mat')
X = data['X']
y = data['y']
X
y
m, _ = X.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]


displayData(sel)

input('Program paused. Press Enter to continue...')

#  ============ Part 2: Vectorize Logistic Regression ============

print('Training One-vs-All Logistic Regression...')

Lambda = 0.1
all_theta = oneVsAll(X, y, num_labels, Lambda)

input('Program paused. Press Enter to continue...')

#  ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X)
y = np.squeeze(y)
pred = np.squeeze(pred)

accuracy = np.mean(np.double(pred == y)) * 100
print('\nTraining Set Accuracy: %f\n' % accuracy)
