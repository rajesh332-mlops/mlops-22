## Ignore warnings

import warnings
warnings.filterwarnings("ignore")

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


## Part 1: Import the Libraries

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.tree import DecisionTreeClassifier


## Part 2: Set the ranges of hyper parameters for hyper-parameter tunning

# SVC parameter: We will use gamma and c parameter for tunning

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

# Decision Tree classifier parameter: We will use max_depth,  min_samples_leaf, and criterion parameter for tunning

max_depth = [2, 3, 5, 10]
min_samples_leaf = [5, 10, 20, 50]
criterion =  ["gini", "entropy"]

# 5 different splits of train/test/validation set

train_frac = [0.8, 0.75, 0.7, 0.65, 0.6]
val_frac =   [0.1, 0.125, 0.15, 0.175, 0.2]
test_frac =  [0.1, 0.125, 0.15, 0.175, 0.2]


## Part 3: Load the Digit dataset

digits = datasets.load_digits()
print(f"\nImage size in the digits dataset is: {digits.images.shape}")
print("\n5 Confusion Matrix for 5 different splits of Train/test/Val using SVC and Decision Tree classifier:\n")

## Part 4: Data Pre-processing

acc_df = pd.DataFrame()
acc_svc = list()
acc_dt = list()

best_acc_svc = -1.0
best_model_svc = None
best_h_params_svc = dict()
best_acc_dt = -1.0
best_model_dt = None
best_h_params_dt = dict()

## Train, Test, Val set split

for i in range(5):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_val, y_train, y_val = train_test_split(data, digits.target, test_size = val_frac[i], shuffle = True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = test_frac[i], shuffle = True)
    
    # For Support Vector Machines (SVM)
    
    for g in gamma_list:
        for c in c_list:            
            
            ## Define the model
            ## Create a classifier: a Support Vector classifier
            
            clf = svm.SVC(gamma = g, C = c, random_state = 42)

            ## Train the model
            ## Learn the digits on the train subset
            
            clf.fit(X_train, y_train)

            ## Get the prediction of validation set
            
            predicted = clf.predict(X_val)
            cur_acc = metrics.accuracy_score(y_pred = predicted, y_true = y_val)    
                
            ## Identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
                
            if (cur_acc > best_acc_svc):
                best_acc_svc = cur_acc
                best_model_svc = clf
                best_h_params_svc[i] = {"gamma":g, "C": c}

    
    # For Decision Tree Clssifier    
    
    for d in max_depth:
        for l in min_samples_leaf:
            for c in criterion:
            
                ## Define the model
                ## Create a classifier: a Decision Tree classifier
                
                clf = DecisionTreeClassifier(max_depth = d, min_samples_leaf = l, criterion = c, random_state = 42)

                ## Train the model
                ## Learn the digits on the train subset
                
                clf.fit(X_train, y_train)

                ## Get the prediction of validation set
                
                predicted = clf.predict(X_val)
                cur_acc = metrics.accuracy_score(y_pred = predicted, y_true = y_val)
                
                ## Identify the combination-of-hyper-parameter for which validation set accuracy is the highest
                
                if (cur_acc > best_acc_dt):
                    best_acc_dt = cur_acc
                    best_model_dt = clf
                    best_h_params_dt[i] = {"max_depth":d, "min_samples_leaf": l, "criterion": c}

    
    ## Get the test set prediction on best params i.e. using best model (SVM model) and calculate the accuracy of test set using best model

    pred_svc = best_model_svc.predict(X_test)
    acc = metrics.accuracy_score(y_pred = pred_svc, y_true = y_test)
    acc_svc.append(acc)
    
    ## Confusion Matrix

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred_svc)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"\nConfusion matrix_SVC:\n\n{disp.confusion_matrix}\n")
    plt.show()

    
    ## Get the test set prediction on best params i.e. using best model (Decision Tree model) and calculate the accuracy of test set using best model
    
    pred_dt = best_model_dt.predict(X_test)
    acc = metrics.accuracy_score(y_pred = pred_dt, y_true = y_test)
    acc_dt.append(acc)

    ## Confusion Matrix

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred_dt)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"\nConfusion matrix_Decision_Tree:\n\n{disp.confusion_matrix}\n")
    plt.show()
    

## calculate the mean and standard deviations of both the classifier's performances

acc_svc.append(np.mean(acc_svc))
acc_dt.append(np.mean(acc_dt))
acc_svc.append(np.std(acc_svc))
acc_dt.append(np.std(acc_dt))
acc_df[''] = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5', "mean", "std"]
acc_df["SVC"] = acc_svc
acc_df["DT"] = acc_dt

##
print("\nBest hyper-params for 5 different splits of Train/test/Val using SVM and Decision Tree classifier are:\n")
print(best_h_params_svc, best_h_params_dt, sep='\n')
print("\nAccuracy of Test set for 5 different splits of Train/test/Val using SVM and Decision Tree classifier are:\n")
print(acc_df)
print("\n")

## Write the performance metrics in the readme.md

acc_df.to_markdown("readme.md")
