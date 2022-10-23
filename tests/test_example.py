import sys, os
import numpy as np
from joblib import load
from sklearn import datasets, svm
from sklearn import datasets
from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
)
sys.path.append(".")
from plot_graph import y_train,predicted

def test_gorund_truth_labels():
    digits = datasets.load_digits()
    data_viz(digits)
    data, label = preprocess_digits(digits)
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, 0.8, 0.1)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    predicted = clf(x_test)
    set_1 = set(y_train)
    set_2 = set(predicted)
    assert set_1==set_2  # both set of train and predicted classes must be same


def test_classifier_bias():
    digits = datasets.load_digits()
    data_viz(digits)
    data, label = preprocess_digits(digits)
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, 0.8, 0.1)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    predicted = clf(x_test)
    assert len(set(predicted))>1
