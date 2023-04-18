#!/usr/bin/env python
"""Draw a plot of the community-urls usage."""

import json

import networkx as nx
import numpy as np
from matplotlib import colors, pyplot
from scipy import sparse
from sklearn import cluster

from conf import TAU

pyplot.style.use("mplrc")


def get_keywords(topic):
    """Load filtering words.

    Get all possible keywords plus the minimum set of keywords contained in the full set.

    Parameters
    ----------
    topic : str
        the topic (one of `covid` or `vaccine`)

    Returns
    -------
    topics : dict
        a dictionary with a list of self excluding tags (`short`) and all tags (`long`)
    """
    filepath = f"../data/keys-{topic}.txt"
    print(filepath)
    with open(filepath, "rt") as fin:
        long = set([line.lower().strip() for line in fin])

    return long


def load_mats(kind):
    basename = f"../data/stats_comm_tau_{TAU}_"
    adj = sparse.load_npz(basename + kind + ".npz")
    with open(basename + kind + "_map.json", "rt") as fin:
        names = json.load(fin)

    if kind == "hashtags":
        keywords = get_keywords("covid") | get_keywords("vaccine")
        keywords |= {"covid__19", "covid19france", "covid_19fr", "covidー19"}
        indexes = [v for k, v in names["feat2indx"].items() if k not in keywords]

        adj = adj[:, indexes]
        names["feat2indx"] = {
            k: i for i, k in enumerate([n for n in names["feat2indx"].keys() if n not in keywords])
        }

    inv_names = {}
    for kname, name in names.items():
        inv_names[kname] = {int(i): k for k, i in name.items()}

    return adj, inv_names


def load_data(kinds):
    """Load data."""
    if isinstance(kinds, str):
        # load data
        adj, names = load_mats(kinds)
    else:
        adj, names = load_mats(kinds[0])
        print(adj.sum())
        for kind in kinds:
            _adj, _names = load_mats(kind)
            adj += _adj

    adj = adj.toarray()

    # remove under represented communities and urls
    threshold = np.sort(adj.sum(1))[-30]
    c = np.argwhere(adj.sum(1) >= threshold).flatten()
    threshold = np.sort(adj.sum(0))[-50]
    u = np.argwhere(adj.sum(0) >= threshold).flatten()

    _names = {}
    adj = adj[c, :]
    _names["communities"] = [names["comm2indx"][i] for i in c]
    adj = adj[:, u]
    _names["features"] = [names["feat2indx"][i] for i in u]

    print(adj.shape)

    return adj, _names


def projection(labels):
    """Return the projector to the label space."""
    proj = np.zeros((len(np.unique(labels)), len(labels)))
    proj[(labels, np.arange(len(labels)))] = 1
    return proj


def arg_label_sort(labels, weights):
    """Return the indices that sort the labels.

    Use the weights to sort the labels,
    integrating over all entries with the same label.
    """
    proj = projection(labels)
    comm_weights = proj @ weights
    indx = np.argsort(comm_weights[labels], kind="stable")[::-1]
    return indx


def cluster_labels(matrix, use="agglomerative"):
    """Return labels."""
    if use == "agglomerative":
        kmeans = cluster.AgglomerativeClustering(n_clusters=7, affinity="cosine", linkage="average")
        xlabels = kmeans.fit(matrix).labels_
        kmeans = cluster.FeatureAgglomeration(n_clusters=7, affinity="cosine", linkage="average")
        ylabels = kmeans.fit(matrix).labels_
    else:
        adj1 = matrix @ matrix.T
        adj1[adj1 < 0] = 0
        xlabels = covid.directed_partition(nx.from_numpy_array(adj1))
        xlabels = np.array(list(xlabels.values()))
        adj2 = matrix.T @ matrix
        adj2[adj2 < 0] = 0
        ylabels = covid.directed_partition(nx.from_numpy_array(adj2))
        ylabels = np.array(list(ylabels.values()))

    return xlabels, ylabels


def sort_adj(adj, freq, names):
    """Sort the matrix."""

    def sort_all(indx, labels, adj, freq, names, axis="communities"):
        if axis == "communities":
            adj[:, :] = adj[indx, :]
            freq[:, :] = freq[indx, :]
        else:
            adj[:, :] = adj[:, indx]
            freq[:, :] = freq[:, indx]
        names[axis] = [names[axis][i] for i in indx]

        rlabels = labels[indx]
        return [i for i, (l1, l2) in enumerate(zip(rlabels, rlabels[1:])) if l1 != l2]

    ncomm, nurl = adj.shape

    # sort communities based on retweets
    xlabels = np.arange(ncomm)
    indx = arg_label_sort(xlabels, freq.sum(1))
    sort_all(indx, xlabels, adj, freq, names, axis="communities")

    # sort urls based on retweets
    ylabels = np.arange(nurl)
    indx = arg_label_sort(ylabels, freq.sum(0))
    sort_all(indx, ylabels, adj, freq, names, axis="features")

    xlabels, ylabels = cluster_labels(adj, use="agglomerative")

    # sort communities based on similarity
    indx = arg_label_sort(xlabels, freq.sum(1))
    sep_comms = sort_all(indx, xlabels, adj, freq, names, axis="communities")

    # sort urls based on similarity
    indx = arg_label_sort(ylabels, freq.sum(0))
    sep_urls = sort_all(indx, ylabels, adj, freq, names, axis="features")

    return sep_urls, sep_comms


def main():
    """Do main."""
    adj_freq, names = load_data(["urls_pre", "urls_post"])
    __plot_adj__(adj_freq, names, "plot_url_crit_heatmap.pdf")
    adj_freq, names = load_data(["urls_media"])
    __plot_adj__(adj_freq, names, "plot_url_media_heatmap.pdf")
    adj_freq, names = load_data(["hashtags"])
    __plot_adj__(adj_freq, names, "plot_url_hash_heatmap.pdf")


def __plot_adj__(matrix, names, filename):
    marginal0 = matrix.sum(0) / matrix.sum()
    marginal1 = matrix.sum(1) / matrix.sum()

    adj_null = np.outer(marginal1, marginal0)
    adj = matrix / matrix.sum() - adj_null

    print(names.keys())
    sep_urls, sep_comms = sort_adj(adj, matrix, names)

    fig, axes = pyplot.subplots(
        nrows=2,
        ncols=2,
        sharex="col",
        sharey="row",
        figsize=(8, 8),
        gridspec_kw={
            "top": 0.75,
            "bottom": 0,
            "right": 1,
            "hspace": 0,
            "wspace": 0,
            "width_ratios": [3, 1],
            "height_ratios": [4, 1],
        },
    )

    # plot main adj
    n_comm, n_urls = adj.shape
    ax = axes[0, 0]
    ax2 = ax.twiny()
    vmax = max(np.abs(adj.min()), np.abs(adj.max()))
    cadj = ax.matshow(
        adj,
        aspect="auto",
        cmap="PiYG",
        norm=colors.SymLogNorm(linthresh=0.00001, vmin=-vmax, vmax=vmax),
    )

    cb_ax = pyplot.axes([0.77, 0.05, 0.2, 0.2], frame_on=False, xticks=[], yticks=[])
    fig.colorbar(
        cadj,
        ax=cb_ax,
        orientation="horizontal",
        aspect=12,
    )
    ax.set(
        xticks=[],
        yticks=range(n_comm),
        yticklabels=names["communities"],
        ylabel="Communities",
    )
    ax2.set(
        xlim=(0, n_urls),
        xticks=np.arange(n_urls) + 0.5,
    )

    def truncate(string, lenght=30):
        if len(string) <= 30:
            return string
        return string[:29] + "…"

    ax2.set_xticklabels([truncate(n) for n in names["features"]], rotation=90, fontsize="x-small")

    for y in sep_comms:
        ax.axhline(y + 0.5)
    for x in sep_urls:
        ax.axvline(x + 0.5)

    # plot url density
    ax = axes[1, 0]
    ax.bar(np.arange(n_urls), -matrix.sum(0))
    ax.set(
        xticks=[],
        yticks=[],
        ylabel="Number of\nretweets",
        frame_on=False,
    )

    # plot comm density
    ax = axes[0, 1]
    ax2 = ax.twiny()
    ax.barh(np.arange(n_comm), matrix.sum(1), log=True)
    ax.set(
        frame_on=False,
    )
    ax2.set(
        frame_on=False,
        xlabel="Number of retweets\n(log scale)",
        xticks=[],
    )
    ax.axis("off")

    ax = axes[1, 1]
    ax.axis("off")

    fig.align_ylabels(axes)
    pyplot.savefig(filename)


if __name__ == "__main__":
    main()
