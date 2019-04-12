import random
import numpy as np
from data_process import get_CIFAR10_data
import math
from scipy.spatial import distance
from models import KNN, Perceptron, SVM, Softmax
from kaggle_submission import output_submission_csv

# You can change these numbers for experimentation
# For submission we will use the default values 
TRAIN_IMAGES = 4900
VAL_IMAGES = 100
TEST_IMAGES = 500

data = get_CIFAR10_data(TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

def get_acc(pred, y_test):
    return np.sum(y_test==pred)/len(y_test)*100

print("finished reading data")


knn = KNN(5)
knn.train(X_train, y_train)
pred_knn = knn.predict(X_test)
print('The testing accuracy is given by : %f' % (get_acc(pred_knn, y_test)))


'''

knn = KNN(5)
knn.train(X_train, y_train)
pred_knn = knn.predict(X_test)
print('The testing accuracy is given by : %f' % (get_acc(pred_knn, y_test)))

percept_ = Perceptron()
percept_.train(X_train, y_train)
pred_percept = percept_.predict(X_test)
print('The testing accuracy is given by : %f' % (get_acc(pred_percept, y_test)))


svm = SVM()
svm.train(X_train, y_train)
pred_svm = svm.predict(X_test)
print('The testing accuracy is given by : %f' % (get_acc(pred_svm, y_test)))

softmax = Softmax()
softmax.train(X_train, y_train)
pred_softmax = softmax.predict(X_test)
print('The testing accuracy is given by : %f' % (get_acc(pred_softmax, y_test)))

'''








