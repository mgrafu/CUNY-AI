import logging
import time
import numpy as np

import preprocess
import plots

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def get_best_params(i, X_train, y_train):
    st = time.time()
    regs = np.arange(0.001, 1.002, 0.01)
    parameters = {'alpha': regs}
    clf = GridSearchCV(MLPClassifier(verbose="False", activation="relu"), parameters)
    clf.fit(X_train, y_train)
    all_scores = clf.cv_results_
    plots.err_per_lambda(i, [[0 for i in range(len(regs))],
                             [1 - i for i in all_scores["mean_test_score"]]],
                         regs, "MLP")
    
    best_alpha = clf.best_params_["alpha"]
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\Best alpha for MLP obtained...")
    
    return best_alpha
