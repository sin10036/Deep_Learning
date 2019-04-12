import numpy as np
import scipy
import time

class Perceptron():
    def __init__(self):
        """
        Initialises Perceptron classifier with initializing 
        weights, alpha(learning rate) and number of epochs.
        """
        self.w = None
        self.alpha = 0.5
        self.epochs = 100
        
    def train(self, X_train, y_train):
        """
        Train the Perceptron classifier. Use the perceptron update rule
        as introduced in Lecture 3.

        Inputs:
        - X_train: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y_train: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        X_train = np.hstack((X_train, np.vstack(np.ones(len(X_train)))))
        self.w = np.zeros((3073,10)) #better than random
        for epoch in range(self.epochs):
            self.alpha = 0.5 * 100 / (100 + epoch) #decay
            start = time.time()
            loss = 0
            for i in range(len(X_train)):
                x = X_train[i]
                y = int(y_train[i])
                pred_arr = np.dot(x, self.w)
                base = pred_arr[y]
                self.w = self.w.transpose()
                false_count = 0
                for j in range(10):
                    if j != y and pred_arr[j] >= base:
                        loss += pred_arr[j] - base #calculate loss
                        self.w[j] -= x * self.alpha
                        false_count += 1
                self.w[y] += x * self.alpha * false_count
                self.w = self.w.transpose()
            #shuffle
            
            tmp = np.hstack((X_train, np.vstack(y_train)))
            np.random.shuffle(tmp)
            X_train = tmp[:,:-1]
            y_train = tmp[:,-1]
            
            print("time: " + str(time.time() - start) + " loss: " + str(loss))


    def predict(self, X_test):
        """
        Predict labels for test data using the trained weights.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        X_test = np.hstack((X_test, np.vstack(np.ones(len(X_test)))))
        pred = np.dot(X_test, self.w)
        ret = np.argmax(pred, axis=1)

        return ret 




