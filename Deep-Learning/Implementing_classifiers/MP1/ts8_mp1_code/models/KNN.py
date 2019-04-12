import numpy as np
from scipy.spatial import distance
import operator
import time

class KNN():
    def __init__(self, k):
        """
        Initializes the KNN classifier with the k.
        """
        self.k = k
        self.X = None
        self.y = None
    
    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X = X
        self.y = y
    
    def find_dist(self, X_test):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train.

        Hint : Use scipy.spatial.distance.cdist

        Returns :
        - dist_ : Distances between each test point and training point
        """
        return distance.cdist(X_test, self.X, 'cityblock')
    
    def predict(self, X_test):
        """
        Predict labels for test data using the computed distances.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - ret: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        start = time.time()
        dist = self.find_dist(X_test)
        print("time to calculate distance: " + str(time.time() - start))
        start = time.time()
        dist_top = np.argpartition(dist, self.k)
        ret = np.zeros(len(X_test), dtype=int)
        
        for i in range(len(X_test)):
            tmp = {}
            for j in range(self.k):
                clazz = self.y[dist_top[i][j]]
                if clazz not in tmp:
                    tmp[clazz] = 1
                else:
                    tmp[clazz] += 1
            pred = sorted(tmp.items(), key=operator.itemgetter(1), reverse=True)[0][0]
            ret[i] = int(pred)
        print("time to finish: " + str(time.time() - start))

        return ret






