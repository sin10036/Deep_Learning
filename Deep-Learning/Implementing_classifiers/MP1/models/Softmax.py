import numpy as np
import time

class Softmax():
    def __init__(self):
        """
        Initialises Softmax classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.w = None
        self.alpha = 0.01
        self.epochs = 20
        self.reg_const = 0.01
        self.batch_size = 10
    
    def calc_gradient(self, X_train, y_train):
        """
        Calculate gradient of the softmax loss
          
        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing a minibatch of data.
        - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.

        Returns:
        - gradient with respect to weights W; an array of same shape as W
        """
        scores = np.dot(X_train, self.w)
        correct_scores = np.choose(y_train, scores.T)

        exp_matrix = scores
        for i in range(len(exp_matrix)):
            exp_matrix[i][y_train[i]] = 0

        loss = -np.sum(correct_scores) + np.log(np.sum(np.exp(exp_matrix))) + 0.5 * self.reg_const * np.sum(self.w ** 2)

        exp_sum = np.sum(np.exp(exp_matrix), axis=1)
        exp_grad = (exp_matrix.T / exp_sum).T
        for i in range(len(exp_grad)):
            exp_grad[i][y_train[i]] -= 1

        grad = np.dot(X_train.T, exp_grad)
        grad /= self.batch_size
        grad += self.reg_const * self.w

        return grad, loss
    
    def train(self, X_train, y_train):
        """
        Train Softmax classifier using stochastic gradient descent.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing training data;
        N examples with D dimensions
        - y_train: A numpy array of shape (N,) containing training labels;
        
        Hint : Operate with Minibatches of the data for SGD
        """

        X_train = np.hstack((X_train, np.vstack(np.ones(len(X_train)))))
        self.w = np.zeros((3073,10)) #better than random
        for epoch in range(self.epochs):
            self.alpha = 0.01 * 100 / (100 + epoch) #decay
            total_loss = 0
            start = time.time()
            length = int(len(X_train) / self.batch_size)
            for i in range(length):
                grad, loss = self.calc_gradient(X_train[i*self.batch_size:(i+1)*self.batch_size], y_train[i*self.batch_size:(i+1)*self.batch_size])
                #self.w *= (1 - self.alpha * self.reg_const / self.batch_size)
                self.w -= self.alpha * grad
                total_loss += loss
            print("time: " + str(time.time() - start) + " loss: " + str(total_loss))

    
    def predict(self, X_test):
        """
        Use the trained weights of softmax classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        X_test = np.hstack((X_test, np.vstack(np.ones(len(X_test)))))
        pred = np.dot(X_test, self.w)
        ret = np.argmax(pred, axis=1)

        return ret 



