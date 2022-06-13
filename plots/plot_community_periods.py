#!/usr/bin/env python3
"""Plot the temporal behaviour of the communities on the three periods."""

import json
from collections import Counter
from datetime import date, timedelta

import numpy as np
from matplotlib import pyplot

import cov_utils

pyplot.style.use("mplrc")
grey = "#999999"


class NumpyEncoder(json.JSONEncoder):
    """A class to encode ndarray as list."""

    def default(self, obj):
        """Encode as list."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_data():
    """Just load data."""
    d = {}
    for kind in ["crit", "media", "full"]:
        print(kind)
        np_data = np.load(f"../data/community_temporal_periods_{kind}_-1.npz")
        d[kind] = {"in": dict(zip(np_data["comms"], np_data["steady"]))}
        probs = np_data["probs"]
        for i in range(probs.shape[-1]):
            np.fill_diagonal(probs[:, :, i], 0)
        d[kind]["out"] = dict(zip(np_data["comms"], probs.sum(0) / np_data["steady"]))
        d[kind]["active"] = dict(zip(np_data["comms"], np_data["active"]))
        d["periods"] = np_data["periods"]

    d["days"] = [date.fromisoformat(p[0]) + timedelta(days=15) for p in d["periods"]]

    d["partsize"] = Counter(cov_utils.get_communities(-1).values())

    return d


def draw_balls(ax: pyplot.Axes, data, psize):
    """Draw balls."""
    for i in range(7):
        community = f"C{i}"
        xes = data["out"][community]
        yes = np.array(data["in"][community]) / psize[community]

        for x1, y1, x2, y2 in zip(xes, yes, xes[1:], yes[1:]):
            arrow_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if arrow_length < 0.05:
                continue
            ax.annotate(
                "",
                (x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    facecolor=community,
                    linewidth=0,
                    shrink=0.0,
                    width=5,
                    headwidth=5,
                    headlength=arrow_length * 100,
                    alpha=0.4,
                    connectionstyle="arc3,rad=0.2",
                ),
            )
        ax.scatter(xes, yes, label=community, s=psize[f"C{i}"] / 1000, alpha=[0.8, 0.5, 0.3])

    ax.semilogy()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1e-9, 1e-4)


def main():
    """Do the main."""
    dataset = load_data()

    fig, axes = pyplot.subplots(
        ncols=3,
        nrows=1,
        sharey=True,
        sharex=True,
        gridspec_kw={
            "wspace": 0,
            "top": 0.92,
            "right": 0.99,
            "left": 0.1,
            "bottom": 0.18,
        },
        figsize=(8, 3.3),
    )

    draw_balls(axes[0], dataset["crit"], dataset["partsize"])
    axes[0].set_title("Vaccine critical URLs")
    axes[0].set_ylabel("Average visiting probability")
    draw_balls(axes[1], dataset["media"], dataset["partsize"])
    axes[1].set_title("Media")
    axes[1].set_xlabel("Escape probability")
    axes[1].tick_params(left=False, which="both")
    draw_balls(axes[2], dataset["full"], dataset["partsize"])
    axes[2].set_title("Full hyper-graph")
    axes[2].legend(fancybox=False, frameon=False)
    axes[2].tick_params(left=False, which="both")

    # fig.suptitle("Evolution of community reach in the three periods")
    pyplot.tight_layout(h_pad=0, w_pad=0)

    pyplot.savefig("plot_community_periods.pdf")


if __name__ == "__main__":
    main()
