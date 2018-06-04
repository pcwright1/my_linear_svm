# my_linear_svm
Implementation of my own linear SVM with a huberized hinge loss.

The SVM class does 1v1 classification using a linear SVM with a huberized hinge loss. 
The main components of the class svm.py are one_v_one_classifiers and predict_label.

# Example calls:

Below are example calls to the main functions of the svm class. Further examples are included in the svm_example.py and svm_digits_example.py


```
svm.one_v_one_classifiers(x=X_train,
                            y=y_train,
                            lambd=-1,
                            max_iters=100)


svm.predict_label(val_features, 
                  classifier_betas,
                  i_vals,j_vals)
```

