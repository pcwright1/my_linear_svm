
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
import matplotlib.pyplot as plt


if __name__ == '__main__':
    #import data
    samples_per_class = 1000
    y_pca = np.repeat([0,1,2],samples_per_class)
    means = [10,200,300]
    variances = [2,30,50]#[100,200,300]
    x_pca = np.append(np.random.normal(loc=means[0],scale=variances[0],size=(samples_per_class,60)),
                      np.random.normal(loc=means[1],scale=variances[1],size=(samples_per_class,60)),
                      axis=0)
    x_pca = np.append(x_pca,
                      np.random.normal(loc=means[2],scale=variances[2],size=(samples_per_class,60)),
                      axis=0)
    X_train, val_features,y_train, val_labels = train_test_split(x_pca,y_pca,test_size=0.25)
    
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
    