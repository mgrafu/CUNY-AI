import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def elbow_method(i, sse_list, all_k, dists, title):
    plt.figure(i)
    plt.xlabel("k")
    plt.ylabel("WSS")
    for sse, dist in zip(sse_list, dists):
        plt.plot(all_k, sse, label="Dist: " + dist)
    plt.legend()
    plt.savefig("charts/" + title + "_elbow.pdf")
    plt.show()


def gap(i, wcd_list, all_k, dists, title):
    plt.figure(i)
    plt.xlabel("k")
    plt.ylabel("Avg. WCD")
    for wcd, dist in zip(wcd_list, dists):
        plt.plot(all_k, wcd, label="Dist: " + dist)
    plt.legend()
    plt.savefig("charts/" + title + "_gap.pdf")
    plt.show()


def scatter_plot(i, X, p, dist, title):
    X_small = PCA(n_components=2).fit_transform(X)
    name = "Clusters with " + title + "+" + dist
    plt.figure(i)
    plt.title(name)
    plt.scatter(X_small[:,0], X_small[:,1], c=p)
    plt.savefig("charts/scatter-" + title + "_" + dist + ".pdf")
    plt.show()


def err_per_lambda(i, err_lists, regs, reg_type):
    err_list_in, err_list_out = err_lists
    plt.figure(i)
    plt.xlabel("regularizer")
    plt.ylabel("error")
    plt.plot(regs, err_list_in, label="training")
    plt.plot(regs, err_list_out, label="validation")
    plt.legend()
    plt.savefig(f"charts/Cluster{i}_{reg_type}-ErrorComparison.pdf")
