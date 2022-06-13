#!/usr/bin/env python3
"""Plot tweets and retweets fraction of critic and media content."""

from datetime import date

import numpy as np
from matplotlib import pyplot

import cov_utils

pyplot.style.use("mplrc")
grey = "#999999"


def main():
    """Do your stuff."""
    data = cov_utils.load_csv("../data/daily_stats.tsv", transpose=True)
    print(*data.keys())
    data["day"] = [date.fromisoformat(day) for day in data["day"]]
    for key in data:
        if key[:3] == "twt" or key[:4] == "rtwt":
            data[key] = np.array(data[key])
    fig, axes = pyplot.subplots(
        nrows=2,
        figsize=(7, 3),
        sharex=True,
        gridspec_kw={
            "hspace": 0,
            "top": 0.9,
            "right": 0.95,
            "left": 0.15,
            "height_ratios": [0.6, 0.4],
        },
    )

    crit = data["twt_crit"] + data["rtwt_crit"]
    media = data["twt_media"] + data["rtwt_media"]
    tot = crit + media + data["twt_nourls"] + data["rtwt_nourls"]
    tot[tot == 0] = 1
    ax = axes[0]
    ax.plot(data["day"], crit / tot, label="Vaccine critical URLs")
    ax.plot(data["day"], media / tot, label="Media URLs", zorder=-1)
    ax.set_ylabel("Fraction")

    ax = axes[1]
    ax.plot(data["day"], crit)
    ax.plot(data["day"], media)
    ax.plot(data["day"], tot, color=grey, label="Total flow")
    ax.set_ylabel("Count")

    for ax in axes:
        ax.axvline(date(2020, 11, 11), color=grey, zorder=-1, alpha=0.5)
        ax.axvline(date(2021, 6, 1), color=grey, zorder=-1, alpha=0.5)

    fig.legend()
    fig.align_labels()
    pyplot.savefig("plot_simple_stats.pdf")


if __name__ == "__main__":
    main()
