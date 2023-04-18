#!/usr/bin/env python3
"""Plot community reach.

depends on:
- ../run_10_make_adj.py
- ../
"""


import json
from collections import Counter

import numpy as np
from adjustText import adjust_text
from matplotlib import pyplot

import cov_utils
from conf import TAU

pyplot.style.use("mplrc")
grey = "#999999"


def load_comm_retweets(kinds="PRE"):
    if isinstance(kinds, str):
        kinds = [
            kinds,
        ]

    with open(f"../data/community_reach_edgelist-tau_{TAU}.json", "rt") as fin:
        data = json.load(fin)

    base = {"out": 0, "in": 0, "self": 0}
    degrees = {}
    for kind in kinds:
        for edge in data[kind.lower()]:
            degrees.setdefault(edge["from"], base.copy())
            degrees.setdefault(edge["to"], base.copy())
            if edge["from"] == edge["to"]:
                degrees[edge["from"]]["self"] += edge["norm_weight"]
            else:
                degrees[edge["from"]]["out"] += edge["norm_weight"]
                degrees[edge["to"]]["in"] += edge["norm_weight"]

    return {
        "part": list(degrees.keys()),
    } | {k: np.array([d[k] for d in degrees.values()]) for k in base}


def comm_size():
    """Return the size of each partition.

    Returns
    -------
    count : dict
        a dict of partiton -> number of nodes.
    """
    with open(f"../data/adjacency-tau_{TAU}_nodes.json", "rt") as fin:
        data = json.load(fin)["nodes"]

    return Counter([d["part"] for d in data])


def draw_labels(ax, *args):
    texts = []
    for x, y, t in args:
        texts.append(
            ax.text(
                x,
                y,
                t,
                va="center",
                ha="center",
                transform=ax.transAxes,
                color=t,
            )
        )

    adjust_text(
        texts,
        ax=ax,
        autoalign="y",
        arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.5),
        transform=ax.transAxes,
    )


def plot_balls(ax, xes, yes, sizes, colors, parts):
    ax.scatter(xes, yes, sizes / 100, color=[colors[p] for p in parts], alpha=0.8)
    text = [x for x in zip(xes, yes, parts) if x[2] in set([f"C{i}" for i in range(7)])]
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e-4)
    draw_labels(ax, *text)


def main():
    """Do the main."""
    fig, axes = pyplot.subplots(
        ncols=3,
        nrows=1,
        figsize=(8, 3.3),
        sharey="row",
        sharex=True,
        squeeze=False,
        gridspec_kw={
            "wspace": 0,
            "top": 0.92,
            "right": 0.99,
            "left": 0.1,
            "bottom": 0.18,
        },
    )
    csize = comm_size()
    data = cov_utils.load_csv(f"../data/community_outreach-tau_{TAU}.tsv", transpose=True)
    print(data.keys())
    for k, v in data.items():
        if k != "part":
            data[k] = np.array(v)

    rng = np.random.default_rng()
    # colors = {p: cmap(r) for p, r in zip(csize, rng.random(len(csize)))}
    colors = {p: "#666666" for p, r in zip(csize, rng.random(len(csize)))}
    for i in range(7):
        colors[f"C{i}"] = f"C{i}"

    # Critic probs

    ax = axes[0, 0]
    size = np.array([csize[p] for p in data["part"]])
    xes = data["crit_prob_in_out"] / data["crit_prob_steady"]
    yes = data["crit_prob_steady"] / size
    plot_balls(ax, xes, yes, size, colors, data["part"])
    ax.set_title("Vaccine critical URLs")
    ax.set_ylabel("Average visit probability")

    # Media probs

    ax = axes[0, 1]
    xes = data["media_prob_in_out"] / data["media_prob_steady"]
    yes = data["media_prob_steady"] / size
    plot_balls(ax, xes, yes, size, colors, data["part"])
    ax.tick_params(left=False, which="both")
    ax.set_title("Media")
    ax.set_xlabel("Escape probability", labelpad=10)

    # full probs

    ax = axes[0, 2]
    xes = data["full_prob_in_out"] / data["full_prob_steady"]
    yes = data["full_prob_steady"] / size
    plot_balls(ax, xes, yes, size, colors, data["part"])
    ax.tick_params(left=False, which="both")
    ax.set(title="Full hyper-graph")

    pyplot.savefig("plot_community_reach_onlyprob.pdf")


if __name__ == "__main__":
    main()
