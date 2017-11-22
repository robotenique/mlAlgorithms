# mlAlgorithms
Machine Learning algorithms in practice, from programming tasks of the Machine Learning course by Andrew Ng, in **Python**

## Supervised
#### 1. gradDescent

Full implementation of the *gradient descent algorithm* for linear/polynomial regression, with single and multiple variables, including feature normalization and comparison with the normal equation (analytic methods) and the **scikit-learn** implementation (*linear_model.LinearRegression*)

![gradDescent](http://blog.datumbox.com/wp-content/uploads/2013/10/gradient-descent.png)

#### 2. logRegression
Implementation of *logistic regression* for classification problems. *Feature mapping* is also implemented, along with *regularization* to prevent overfitting. I implemented the logistic cost/loss function (using the *sigmoid*) and the gradient (derivatives) of the function to use *scipy.optimize.minimize* from **scipy**.

![logRegression](https://cdn-images-1.medium.com/max/1200/1*nsphNzg5aAtTWbI4jwKItw.png)


#### 3. nnMultiClass
Logistic regression for *multi-class classification* (*One-vs-All* approach), in this instance used ub a subset of the [MNIST](http://yann.lecun.com/exdb/mnist/) database for recognizing handwritten digits. It's also implemented the *feedforward propagation* of a simple **neural network** for the same problem, comparing the accuracy of One-vs-All approach in logistic regression vs a trained multilayer [*FFNN*](https://en.wikipedia.org/wiki/Feedforward_neural_network).

![ffnn](http://matlabgeeks.com/wp-content/uploads/2011/06/Multi-layer-perceptron.png)

#### 4. nnBackprop
Implementation of the *Backpropagation algorithm* to compute the gradient of the neural network cost function. Regularization is also implemented, and the optimization is made using *scipy.optimize.minimize*, providing the backpropagation algorithm to obtain the gradient matrix of the neural network. This uses the same subset of the *MNIST* database as before.

![backprop](http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop_files/img18.gif)

#### 5. modelEvaluation
A study in model evaluation, for visualizing the different effects of the regularization parameter in linear regression, the learning curve plot, also evaluation and how to build polynomial regression, bias vs variance, and how to use cross validation in a model. It's also implemented the cross validation curve to better visualize how the choosen parameters affect the generalization performance of the implemented model.

![lcurve](https://www.safaribooksonline.com/library/view/hands-on-machine-learning/9781491962282/assets/mlst_04in04.png)

#### 6. SVMclassifier
Linear [SVM] (https://en.wikipedia.org/wiki/Support_vector_machine) classifier and non linear SVM (with [rbf kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)), using *sklearn.svm.SVC* from *scikit-learn*, and also a simple implementation of [hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter) grid search for the . Also, implements an e-mail spam classifier using the [Porter Stemming](https://en.wikipedia.org/wiki/Stemming) algorithm, and mapping the most used words in the dataset ([Apache SpamAssassin] (http://spamassassin.apache.org/)) to numbers. 

![svm](http://scikit-learn.org/stable/_images/sphx_glr_plot_iris_001.png)

