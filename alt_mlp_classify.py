#!/usr/bin/env python

import logging
import time
import numpy as np
import sklearn.model_selection as ms

from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import preprocess
import mlp


def main():
    logging.info(f"Loading data...")
    st = time.time()
    _, _, proc_texts, labels = preprocess.get_all_texts()
    clusters = preprocess.get_clusters()
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tData loaded...")
    
    st = time.time()
    clustered_texts = defaultdict(list)
    for text, label, cluster in zip(proc_texts, labels, clusters):
        clustered_texts[cluster].append((text, label))
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tClusters separated...")

    st = time.time()
    model = Word2Vec.load("data/unigrams_15_min-10.w2v")
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tModel loaded...")
    
    for clust, content in clustered_texts.items():
        if clust != 2:
            continue
        logging.info(f"{'':15}Training cluster {clust}...")
        texts_list = [text for text, _ in content]
        y = np.array([label for _, label in content])

        st = time.time()
        text_embeddings = [preprocess.get_text_embedding(tokens, model) for _, tokens in texts_list]
        logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tText embeddings extracted...")

        st = time.time()
        X = MinMaxScaler().fit_transform(text_embeddings)
        X = PCA(n_components=.99).fit_transform(X)
        logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tDimensionality reduced from ({len(text_embeddings)}, {len(text_embeddings[0])}) to {X.shape}...")

        st = time.time()
        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=.2)
        logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tTrain and dev sets created...")

        supra_st = time.time()

        mlp_alpha = mlp.get_best_params(888, X_train, y_train)
        st = time.time()
        mlp_model = MLPClassifier(alpha=mlp_alpha, verbose="False", activation="relu").fit(X_train, y_train)
        m_test_pred = mlp_model.predict(X_test)
        test_stats = precision_recall_fscore_support(y_test, m_test_pred, average="macro")
        logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tMLP stats calculated with best reg...")
        logging.info(f"{'':>23}\t{test_stats}")
        logging.info(f"{preprocess.get_duration(supra_st, time.time()):>20}\tTotal MLP time")


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    main()
