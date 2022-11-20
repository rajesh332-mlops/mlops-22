# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import pdb
import glob


from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb_svm,
    get_all_h_param_comb_tree,
    tune_and_save,
    save_result,
)
from joblib import dump, load
from sklearn import tree

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

import argparse

# Initialize the Parser
parser = argparse.ArgumentParser()

# Adding Arguments
parser.add_argument('-clf','--clf_name',
                    type=str,
                    )

parser.add_argument('-r','--random_state',
                    type=int,
                    )

args = parser.parse_args()



# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits


if args.clf_name == 'svm':
    clf = svm.SVC()
    # 1. set the ranges of hyper parameters
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb_svm(params)

if args.clf_name == 'dt':
    clf = tree.DecisionTreeClassifier()
    max_depth = [2, 10, 20, 30, 100]
    min_samples_leaf = [1, 2, 3, 4, 5]
    max_features = ['auto', 'sqrt', 'log2']
    params = {}
    params["max_depth"] = max_depth
    params["min_samples_leaf"] = min_samples_leaf
    params["max_features"] = max_features
    h_param_comb = get_all_h_param_comb_tree(params)
random_state = args.random_state
x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac,random_state
)

metric = metrics.accuracy_score
actual_model_path = tune_and_save(
    clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path=None
)

# best_model = load(glob.glob(".\models\svm_*.joblib")[0])
save_result(actual_model_path,x_test,y_test,args.clf_name,random_state)
