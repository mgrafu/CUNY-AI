#!/usr/bin/env python

import logging
import time
import numpy as np
import sklearn.model_selection as ms

import preprocess
import svm
import mlp

from collections import defaultdict
from sklearn.svm import SVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier


def predict_clust(clust, dict_of_models, X, y):
    while X.shape[1] < 1000:
        complement = np.zeros([X.shape[0]])[None,:]
        X = np.concatenate((X, complement.T), axis=1)
        
    path = f"cluster{clust}_results.txt"
    with open(path, "w") as sink:
        for label, model in dict_of_models.items():
            y_pred = model.predict(X)
            stats = precision_recall_fscore_support(y, y_pred, average="macro")
            precision, recall, fscore, _ = stats
            print(f"{label} CLASSIFIER", file=sink)
            print(f"Precision: {precision * 100:.2f}", file=sink)
            print(f"Recall: {recall * 100:.2f}", file=sink)
            print(f"F-Score: {fscore * 100:.2f}", file=sink)
            print("", file=sink)            


def main():
    logging.info(f"{'':15}Loading data...")
    st = time.time()
    _, texts, _, labels = preprocess.get_all_texts()
    clusters = preprocess.get_clusters()
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tData loaded...")
    
    st = time.time()
    clustered_texts = defaultdict(list)
    for text, label, cluster in zip(texts, labels, clusters):
        clustered_texts[cluster].append((text, label))
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tClusters separated...")
    
    cluster_scores_svm = defaultdict(tuple)
    cluster_scores_mlp = defaultdict(tuple)
    all_models = {}
    
    for clust, content in clustered_texts.items():
        logging.info(f"{'':15}Training cluster {clust}...")
        texts_list = [text for text, _ in content]
        y = np.array([label for _, label in content])

        st = time.time()
        bow = preprocess.get_bow(texts_list, sw="english")
        logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tBoW extracted...")
        logging.info(f"{'':23}\t{bow.shape}")
        
        st = time.time()
        X = MinMaxScaler().fit_transform(bow)
        if X.shape[1] > 1000:
            X = TruncatedSVD(n_components=1000).fit_transform(X)
            logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tDimensions reduced...")
            logging.info(f"{'':23}\t{X.shape}")

        st = time.time()
        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=.2)
        logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tTrain and dev sets created...")
        
        best_reg = svm.get_best_reg(clust, X_train, X_test, y_train, y_test)
        st = time.time()
        svm_model = SVC(C=best_reg, kernel="linear").fit(X_train, y_train)
        if clust != 2:
            all_models[f"SVM - cluster {clust}"] = svm_model
        
        r_test_pred = svm_model.predict(X_test)
        test_stats = precision_recall_fscore_support(y_test, r_test_pred, average="macro")
        cluster_scores_svm[clust] = test_stats
        logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tSVM stats calculated with best reg...")
        logging.info(f"{'':>23}\t{test_stats}")

        mlp_alpha = mlp.get_best_params(clust + 100, X_train, y_train)
        st = time.time()
        mlp_model = MLPClassifier(alpha=mlp_alpha, verbose="False", activation="relu").fit(X_train, y_train)
        if clust != 2:
            all_models[f"MLP - cluster {clust}"] = mlp_model
        
        m_test_pred = mlp_model.predict(X_test)
        test_stats_2 = precision_recall_fscore_support(y_test, m_test_pred, average="macro")
        cluster_scores_mlp[clust] = test_stats_2
        logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tMLP stats calculated with best reg...")
        logging.info(f"{'':>23}\t{test_stats_2}")
        
        if clust == 2:
            st = time.time()
            predict_clust(clust, all_models, X, y)
            logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tSupplementary predictions recorded...")


    with open("results.txt", "w") as sink:
        for clust, stats in cluster_scores_svm.items():
            precision, recall, fscore, _ = stats
            print(f"CLUSTER {clust} RESULTS - SVM classifier", file=sink)
            print(f"Precision: {precision * 100:.2f}", file=sink)
            print(f"Recall: {recall * 100:.2f}", file=sink)
            print(f"F-Score: {fscore * 100:.2f}", file=sink)
            print("", file=sink)

        for clust, stats in cluster_scores_mlp.items():
            precision, recall, fscore, _ = stats
            print(f"CLUSTER {clust} RESULTS - MLP classifier", file=sink)
            print(f"Precision: {precision * 100:.2f}", file=sink)
            print(f"Recall: {recall * 100:.2f}", file=sink)
            print(f"F-Score: {fscore * 100:.2f}", file=sink)
            print("", file=sink)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    main()
