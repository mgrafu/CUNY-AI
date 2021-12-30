#!/usr/bin/env python

import logging
import time
import numpy as np

from gensim.models import Word2Vec
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import plots
import preprocess


# Modified from https://gdcoder.com/implementation-of-k-means-from-scratch-in-python-9-lines/
def kmeans(X, k, max_iter=1000, dist="cityblock", plusplus=False):
    # Initialize centroids
    if plusplus:
        centroids = get_kplusplus_centroids(X, k, dist)
    else:
        starter_cent = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[starter_cent,:]
    # Get starter clusters
    P = np.argmin(distance.cdist(X, centroids, dist), axis=1)
    # Optimize
    for _ in range(max_iter):
        centroids = np.vstack([X[P==i,:].mean(axis=0) for i in range(k)])
        tmp = np.argmin(distance.cdist(X, centroids, dist), axis=1)
        if np.array_equal(P, tmp):
            break
        else:
            P = tmp
    return centroids, P

# Modified from https://www.geeksforgeeks.org/ml-k-means-algorithm/
def get_kplusplus_centroids(X, k, dist):
    # Pick one random first centroid
    centroids = [X[np.random.choice(X.shape[0], 1, replace=False)]]
    # Select the remaining k - 1
    for c_id in range(k - 1):
        # Compute the distance between all points and the selected centroids
        dist_matrix = np.vstack([distance.cdist(X, centroid,
                                                dist).T for centroid in centroids]).T
        distances = dist_matrix.min(axis=1)
        # Pick the farthest point as the next selected centroid
        centroids.append(X[np.argmax(distances),:][None,:])
    centroids = np.vstack(centroids)
    return centroids


def kmeans_plusplus(X, k, max_iter=1000, dist="cityblock"):
    return kmeans(X, k, max_iter=max_iter, dist=dist, plusplus=True)

# Modified from https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
def calculate_WSS(X, kmax, max_iter=1000, dist="cityblock", plusplus=False):
    sse = []
    for k in range(1, kmax+1):
        centroids, pred_clusters = kmeans(X, k, max_iter=max_iter,
                                          dist=dist, plusplus=plusplus)
        curr_sse = 0
        # Calculate squared distance of each point from its cluster center and add to current WSS
        for i, centroid in enumerate(centroids):
            curr_sse += np.sum(distance.cdist(X[np.where(pred_clusters == i)],
                                       centroid[None,:], dist) ** 2)
        sse.append(curr_sse)
    return sse

# Modified from https://towardsdatascience.com/k-means-clustering-and-the-gap-statistics-4c5d414acd29
def calculate_gap(X, kmax, max_iter=1000, dist="cityblock", plusplus=False):
    all_wcd = []
    for k in range(1, kmax+1):
        wcd = []
        centroids, pred_clusters = kmeans(X, k, max_iter=max_iter,
                                          dist=dist, plusplus=plusplus)
        current_wcd = 0
        for i, centroid in enumerate(centroids):
            curr_wcd = np.sum(distance.cdist(X[np.where(pred_clusters == i)],
                                             centroid[None,:], dist))
            wcd.append(curr_wcd)
        wcd = np.array(wcd) - np.max(np.array(wcd))
        all_wcd.append(np.mean(wcd))
    return all_wcd


def plot_elbow_method(i, X, k_max=5, max_iter=1000, dist="cityblock", plusplus=True):
    st = time.time()
    plus_sse = calculate_WSS(X, k_max, max_iter=max_iter,
                                dist=dist, plusplus=plusplus)
    plots.elbow_method(i, [plus_sse], range(1, k_max + 1), [dist], "K++")
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tElbow Method for K++ computed...")


def plot_gap(i, X, k_max=5, max_iter=1000, dist="cityblock", plusplus=True):
    st = time.time()
    plus_wcd = calculate_gap(X, k_max, max_iter=max_iter,
                             dist=dist, plusplus=plusplus)
    plots.gap(i, [plus_wcd], range(1, k_max + 1), [dist], "K++")
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tGap Method for K++ computed...")


def save_clusters(clusters):
    with open("data/clusters.txt", "w") as sink:
        for clust in clusters:
            print(clust, file=sink)


def main():
    logging.info(f"Loading data...")
    st = time.time()
    _, _, proc_texts, _ = preprocess.get_all_texts()
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tData loaded...")

    st = time.time()
    model = Word2Vec.load("data/unigrams_15_min-10.w2v")
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tModel loaded...")
    
    st = time.time()
    text_embeddings = [preprocess.get_text_embedding(tokens, model) for _, tokens in proc_texts]
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tText embeddings extracted...")
    
    st = time.time()
    X = MinMaxScaler().fit_transform(text_embeddings)
    X = PCA(n_components=.95).fit_transform(X)
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tDimensionality reduced from ({len(text_embeddings)}, {len(text_embeddings[0])}) to {X.shape}...")
    
    dist = "cityblock"
    
    plot_elbow_method(1, X, k_max=20, dist=dist, plusplus=False)
    print(f"{'':15}Enter optimal K (Elbow method)", end=": ")
    k_1 = input()
    plot_gap(2, X, k_max=20, dist=dist, plusplus=False)
    print(f"{'':15}Enter optimal K (Gap method)", end=": ")
    k_2 = input()
    
    k = int((int(k_1) + int(k_2)) / 2)
    
    st = time.time()
    _, p_plus = kmeans(X, k, dist=dist)
    plots.scatter_plot(3, X, p_plus, dist, "KMeans")
    save_clusters(p_plus)
    logging.info(f"{preprocess.get_duration(st, time.time()):>20}\tKMeans clusters extracted...")


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    main()
