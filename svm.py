import logging
import time
import numpy as np

import preprocess
import plots

from sklearn.svm import SVC


def get_error(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    return (np.sum((y_hat - y) ** 2) / len(y))


def get_best_reg(i, X_train, X_test, y_train, y_test):
    regs = np.arange(0.05,1.05,.05)
    all_errors_in = []
    all_errors_out = []
    for reg in regs:
        # SVM classifier - https://scikit-learn.org/
        st = time.time()
        svm_model = SVC(C=reg, kernel="linear").fit(X_train, y_train)

        r_train_pred = svm_model.predict(X_train)
        all_errors_in.append(get_error(y_train, r_train_pred))

        r_test_pred = svm_model.predict(X_test)
        all_errors_out.append(get_error(y_test, r_test_pred))
        logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tSVM classifier with l={reg} completed...")

    plots.err_per_lambda(i, [all_errors_in, all_errors_out], regs, "SVC")

    best_eval = min(all_errors_out)
    best_index = all_errors_out.index(best_eval)
    corr_ein = all_errors_in[best_index]
    best_reg = regs[best_index]
    
    return best_reg
