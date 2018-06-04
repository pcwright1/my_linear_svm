"""
Linear SVM Module

This module is used to run a 1 vs 1 linear svm with huberized hinge loss through the one_v_one_classifiers and predict_label
functions.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, multiclass, preprocessing, svm
from sklearn import preprocessing
#import matplotlib.pyplot as plt
import os
import time
from multiprocessing.dummy import Pool as ThreadPool 
import matplotlib.pyplot as plt


global X_train
global y_train
global svm_counter
svm_counter = 0
X_train = []
y_train = []

def obj(beta, lambd = .1, x=X_train, y=y_train, h = .5):
    """
    Function for calculating the objective function for the huberized hinge loss.
    
    Inputs:
        beta: numpy vector
            a vector of length d corresponding to the beta vector
        lambd: int
            lambda, the penalization constant. Default = .1
        x: numpy matrix
            a matrix of size nxd
        y: numpy matrix
            a matrix of size nx1
        h: float
            huberized hinge loss parameter. Default = .5
    
    Returns:
        objective value: float, the objective value.
    """
    n, d = x.shape
    o = np.zeros(n)
    o[y*x.dot(beta) < 1 - h] = (1 - y*x.dot(beta))[y*x.dot(beta) < 1 - h]
    o[abs(1 - y*x.dot(beta)) <= h] = ((1 + h - y*(x.dot(beta)))**2 / (4 * h))[abs(1 - y*x.dot(beta)) <= h]
    return  (1.0/n) * np.sum(o) + lambd * np.sum(beta**2)

def computegrad(beta, lambd = .1, x=X_train, y=y_train, h = .5):
    """
    Function for calculating the gradient of the objective function for the huberized hinge loss.
    
    Inputs:
        beta: numpy vector
            a vector of length d corresponding to the beta vector
        lambd: int
            lambda, the penalization constant. Default = .1
        x: numpy matrix
            a matrix of size nxd
        y: numpy matrix
            a matrix of size nx1
        h: float
            huberized hinge loss parameter. Default = .5
    
    Returns:
        gradient: flaot, the gradient value.
    """
    n, d = x.shape
    o = np.zeros([n,d])
    o[y*x.dot(beta) < 1 - h] = (-1.0 * (np.repeat(np.asarray([y]), [d],axis=0)).T*x)[y*x.dot(beta) < 1 - h]
    o[abs(1 - y*x.dot(beta)) <= h] = (np.repeat([(1.0 / (2*h) * (1 + h - y*x.dot(beta)))],[x.shape[1]],axis=0) * 
                                      (-1.0 * np.repeat([y],[x.shape[1]],axis=0)*x.T)).T[abs(1 - y*x.dot(beta)) 
                                      <= h]
    return np.mean(o,axis=0) + 2 * lambd * beta

def backtracking(x, t=1, alpha=0.3,
                 beta=0.1, max_iter=50,
                 lambd=.1,xs=X_train,ys=y_train):
    """
    Function for calculating the step size for the gradient descent.
    
    Inputs:
        x: numpy matrix
            a matrix of size dx1 representing the initial beta
        alpha: float
            normalization term used in the conditional statement
        beta: float
            the factor to decrease t by every iteration.
        max_iters: int
            maximum number of iterations
        lambd: float
            lambda, the penalization constant. Default = .1
        xs: numpy matrix
            a matrix of size nxd

        ys: numpy matrix
            a matrix of size nx1
        h: float
            huberized hinge loss parameter. Default = .5
    
    Returns:
        objective value: int, the objective value.
    """
    grad_x = computegrad(x,lambd=lambd,x=xs,y=ys)  # Gradient at x
    obj_x = obj(x,lambd=lambd,x=xs,y=ys)
    norm_grad_x = np.linalg.norm(grad_x)  # Norm of the gradient at x
    found_t = False
    i = 0 
    while (found_t is False and i < max_iter):
        if obj(x-t*grad_x,lambd=lambd,x=xs,y=ys) <= obj_x - alpha * t * norm_grad_x**2:    
            found_t = True
        elif i == max_iter - 1:
            raise('Maximum number of iterations of backtracking reached')
        else:
            t *= beta
            i += 1
    return t

def mylinearsvm(x_init, t_init=1,
                eps=0.001, max_iter=10000,
                lambd = .1,
                xs = X_train, ys = y_train):
    """
    Function for performing the fast grad algo.
    
    Inputs:
        x_init: numpy matrix
            a matrix of size dx1 representing the initial beta
        t_init: float
            the initial step size, before backtracking
        eps: float
            the stopping criteria for the normalized gradient
        max_iter: int
            maximum number of iterations
        lambd: float
            lambda, the penalization constant. Default = .1
        xs: numpy matrix
            a matrix of size nxd
        ys: numpy matrix
            a matrix of size nx1
    
    Returns:
        x_vals: numpy matrix
            set of betas at each iteration (num_iterations x d)
        objs: numpy matrix
            objective value at each iteration (num_iterations x 1)
    """
    x = x_init
    theta = np.zeros(len(x))
    grad_x = computegrad(theta,lambd=lambd,x=xs,y=ys)
    x_vals = [x]
    objs = [obj(x,lambd = lambd,x=xs,y=ys)]
    iter = 0
    t = t_init
    while iter < max_iter and np.linalg.norm(grad_x) > eps:
        t = backtracking(x, t=t, alpha=0.5, beta=0.1, lambd=lambd, 
                         xs=xs,ys=ys)
        x_new = theta - t*grad_x
        theta = x_new + (iter / (iter+3))*(x_new-x)
        x = x_new
        x_vals.append(x)
        objs.append(obj(x,lambd=lambd,x=xs,y=ys))
        grad_x = computegrad(theta,lambd=lambd,
                             x=xs,y=ys)
        iter += 1
    return x_vals,np.array(objs) 

def CV_k_fold(x,y,k,max_iters,num_lambdas=20):
    """
    Function for performing cross validation to find the optimal lambda.
    
    Inputs:
        x: numpy matrix
            a matrix of size nxd
        y: numpy matrix
            a matrix of size nx1
        k: int
            the number of folds to perform
        max_iters: int
            maximum number of iterations
        num_lambdas: int
            the number of lambdas to check
    
    Returns:
        vals: numpy matrix
            set of (lambdas,misclassification rate) for each lambda tested
        optimal_lambda:  float
            the lambda with the best performance from cross validation
    """
    group = np.random.randint(0,k,len(x))
    lambdas = np.logspace(-2,2,num_lambdas)
    vals = []
    for l in lambdas:
        
        misclassification_val_temp = []
        for i in range(k):
            x_train = x[group != i]
            y_train = y[group != i]
            x_test = x[group == i]
            y_test = y[group == i]
            betas_trains, obj_trains = mylinearsvm(np.zeros(x_train.shape[1]),
                                                          xs = x_train,ys = y_train,
                                                          lambd = l,max_iter = max_iters)
            betas_trains = np.asarray(betas_trains)
            misclassification_val_temp.append(np.mean(((betas_trains.dot(x_test.T) >= 0)*2-1) 
                                                != np.tile(y_test,(betas_trains.shape[0],1)),axis=1)[-1])
        vals.append([l,np.mean(misclassification_val_temp)])
        optimal_lambda = np.asarray(vals)[np.argmin(np.asarray(vals)[:,1]),0]
    return vals, optimal_lambda

def run_svm(features_to_test,
            labels_to_test,
            k=3,
            max_iters=100,
            num_lambdas = 10,
            t_init=1,
            lambd = -1,
            eps=.001):
    """
    Function for running linear SVM with 2 classes.
    
    Inputs:
        features_to_test: numpy matrix
            a matrix of size nxd
        labels_to_test: numpy matrix
            a matrix of size nx1
        k: int
            the number of folds to perform. Default: 3
        max_iters: int
            maximum number of iterations. Default: 100
        num_lambdas: int
            the number of lambdas to check. Default: 10
        t_init: float
            the initial step size, before backtracking. Default: 1
        lambd: float
            lambda, the penalization constant. Default = -1
        eps: float
            the stopping criteria for the normalized gradient. Default: .001

    Returns:
        final_beta: numpy matrix
            the final beta obtained
    """
    global svm_counter
    #run CV_k_fold & SVM
    svm_lambda_min = lambd
    #print("starting CV in run_svm")
    if lambd == -1: #then iterate to find lambd
        svm_lambda_output, svm_lambda_min = CV_k_fold(features_to_test,
                                                      labels_to_test,
                                                      k=k,
                                                      max_iters=max_iters,
                                                      num_lambdas = num_lambdas)
        #print("lambda ", svm_lambda_min)
    #print("starting SVM in run_svm")
    betas, objs = mylinearsvm(np.zeros(features_to_test.shape[1]),
                                t_init=t_init
                                ,lambd = svm_lambda_min
                                ,eps = eps
                                ,max_iter=max_iters
                                ,xs = features_to_test
                                ,ys = labels_to_test)
    final_beta = betas[-1]
    #print("finished run_svm ",svm_counter)
    svm_counter += 1
    return  (final_beta, objs)

def one_v_one_classifiers(x,y,lambd,max_iters,eps=.0001):
    """
    Function for running a 1v1 classifier on many classes using the linearsvm function.
    
    Inputs:
        x: numpy matrix
            a matrix of size nxd
        y: numpy matrix
            a matrix of size nx1
        lambd: float
            lambda, the penalization constant. Default = -1
        max_iters: int
            maximum number of iterations. Default: 100
        eps: float
            the stopping criteria for the normalized gradient. Default: .001

    Returns:
        vals: numpy matrix
            beta values for each pair of classes
        i_vals: numpy matrix
            matrix of first class tested for 1v1 comparison of class i vs class j
        j_vals: numpy matrix
            matrix of second class tested for 1v1 comparison of class i vs class j
    """
    classified_vals = []
    i_vals = []
    j_vals = []
    classes = len(np.unique(y))
    t_init = 10**-1
    t0 = time.time()
    vals_to_run = []
    k=3 # 3 fold CV
    num_lambdas = 3 # num lambdas to try in CV
    vals = []
    vals_to_run = [] # group
    for i in range(classes):
        for j in range(i+1,classes):
            features_to_test = x[(y==i)|(y==j)]
            scaler = preprocessing.StandardScaler()
            features_to_test = scaler.fit_transform(features_to_test)
            labels_to_test = y[(y==i)|(y==j)]
            labels_to_test = ((labels_to_test - min(labels_to_test)) / (max(labels_to_test)-min(labels_to_test)))*2-1
            # save a list of parameters to call run_svm as a list
            
            vals_to_run.append( (features_to_test,
                                labels_to_test,
                                k,
                                max_iters,
                                num_lambdas ,
                                t_init,
                                lambd ,
                                eps)  )
            #classified_vals.append(betas[-1])
            i_vals.append(i)
            j_vals.append(j)
    print("setup complete. Time :",time.time()-t0, " " , time.strftime('%X %x %Z'))
    t0 = time.time()
    #do computation
    pool = ThreadPool(35)
    vals_temp = pool.starmap(run_svm,vals_to_run)
    objs = np.asarray(vals_temp)[:,1]
    vals_temp = np.asarray(vals_temp)[:,0]
    vals = vals + list(vals_temp)
    return np.asarray(vals), np.asarray(i_vals) , np.asarray(j_vals), objs

def predict_label(x, classifier_betas,i_vals,j_vals):
    """
    Function for predicting the class of data using the model data.
    
    Inputs:
        x: numpy matrix
            a matrix of samples to be predicted
        classifier_betas: numpy matrix
            beta values for each pair of classes
        i_vals: numpy matrix
            matrix of first class tested for 1v1 comparison of class i vs class j
        j_vals: numpy matrix
            matrix of second class tested for 1v1 comparison of class i vs class j

    Returns:
        predictions: numpy matrix
            prediction of the class for each sample in x
        
    """
    i_vals = np.asarray(i_vals)
    j_vals = np.asarray(j_vals)
    data = pd.DataFrame(((classifier_betas.dot(x.T) >= 0) * (np.repeat(np.asarray([j_vals - i_vals]),[x.shape[0]],axis=0)).T) + np.repeat([i_vals],[x.shape[0]],axis=0).T)
    predictions = []
    class_modes = pd.DataFrame(data).mode()
    for i in range(data.shape[1]):
        predictions.append(class_modes.loc[np.random.randint(class_modes.loc[:,i].nunique()),i])
    return predictions

def plot_misclassification(data,name,show=False,title = 'test misclassification'):
    """
    function to plot the misclassification over iterations. 
    Inputs:
        data: numpy matrix
            a matrix of samples to be predicted
        name: string
            name of the data to be shown in legend
        show: boolean
            if true, show plot. If false, do not show plot (useful for putting 2 items on 
            the same plot).

    Returns:
        Shows a plot if show=True. Otherwise does not return anything.
    """
    if name is not None:
        plt.loglog(np.asarray(range(len(data)))+1,
                   data,
                   label = name)
        plt.legend()
    else:
        plt.loglog(np.asarray(range(len(data)))+1,
                   data)
    plt.ylabel(title);
    plt.xlabel("iteration");
    plt.title(title)
    if show:
        plt.show();
    
if __name__ == '__main__':
    """
    Main function for pulling in data, running model and doing prediction on the validation and
    test set using the one_v_one_classifiers and predict_label functions.
    """
    
    """print("starting")
    #import data
    data_dir = '/data' #'kaggle2'
    val_features = np.load(os.path.join(data_dir,'val_features.npy'))
    val_labels = np.load(os.path.join(data_dir,'val_labels.npy'))
    X_train = np.load(os.path.join(data_dir,'train_features.npy'))
    y_train = np.load(os.path.join(data_dir,'train_labels.npy'))
    test_features = np.load(os.path.join(data_dir,'test_features.npy'))

    # Standardize the data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
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

    print("starting SVC")
    #t0 = time.time()
    num_classes = 100
    num_features = 409 #6
    classifier_betas, i_vals, j_vals = one_v_one_classifiers(
                                                             x=X_train[y_train<=num_classes,
                                                                       0:num_features],
                                                             y=y_train[y_train<=num_classes],
                                                             lambd=1,
                                                             max_iters=15)

    # Misclassification error
    print(classifier_betas.shape)
    linSVM_misclassification = np.mean(predict_label(val_features[val_labels<=num_classes,0:num_features], 
                                                     classifier_betas,
                                                     i_vals,j_vals)
                                                     != val_labels[val_labels<=num_classes])

    print("Misclassification rate:",np.mean(linSVM_misclassification))
    #pd.DataFrame(predict_label(test_features[:,0:num_features], 
    #                           classifier_betas,
    #                           i_vals,
    #                           j_vals)).to_csv("/data/kaggle2_mysvm_ovo_results.csv")
    #pd.DataFrame([np.mean(linSVM_misclassification)]).to_csv("/data/kaggle2_mysvm_ovo_results_misclassification.csv")














