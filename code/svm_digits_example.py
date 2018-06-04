
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#import matplotlib.pyplot as plt
import os
import time
from multiprocessing.dummy import Pool as ThreadPool 
import svm
from sklearn import datasets, linear_model, multiclass, preprocessing
import matplotlib.pyplot as plt





if __name__ == '__main__':
    """
    Main function for pulling in data, running model and doing prediction on the validation and
    test set using the one_v_one_classifiers and predict_label functions.
    """
    
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
                                                             lambd=1,
                                                             max_iters=100)

    # Misclassification error
    linSVM_misclassification = np.mean(svm.predict_label(val_features, 
                                                     classifier_betas,
                                                     i_vals,j_vals)
                                                     != val_labels)
    print("Misclassification rate:",np.mean(linSVM_misclassification))
    #plt.plot(objs)
    for i in range(objs.shape[0]):
    	svm.plot_misclassification(objs[i],None,show=False,title="Objective Value")
    plt.show()

