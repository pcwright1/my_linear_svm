
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#import matplotlib.pyplot as plt
import os
import time
from multiprocessing.dummy import Pool as ThreadPool 
import svm
from sklearn.model_selection import train_test_split
from sklearn import svm as sklearnsvm
import matplotlib.pyplot as plt
from sklearn import datasets

if __name__ == '__main__':
    #import data
    digits = datasets.load_digits()
    # To apply a classifier on this data, we need to flatten the images, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create training and test sets
    X_train = data[:int(n_samples / 2)]
    val_features = data[int(n_samples / 2):]
    y_train = digits.target[:int(n_samples / 2)]
    val_labels = digits.target[int(n_samples / 2):]

    # Standardize the data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    val_features = scaler.fit_transform(val_features)

    # Run classifier
    classifier_betas, i_vals, j_vals, objs = svm.one_v_one_classifiers(x=X_train,
                                                             y=y_train,
                                                             lambd=-1,
                                                             max_iters=100)

    # Misclassification error
    linSVM_misclassification = np.mean(svm.predict_label(val_features, 
                                                     classifier_betas,
                                                     i_vals,j_vals)
                                                     != val_labels)
    print("Misclassification rate:",np.mean(linSVM_misclassification))
    
    clf = sklearnsvm.SVC()
    clf.fit(X_train, y_train)  
    #print(clf.predict(val_features))
    #print(val_labels)
    print("misclassification skl:",np.mean(clf.predict(val_features) 
                                           != val_labels))

    