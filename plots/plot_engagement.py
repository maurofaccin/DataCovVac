#!/usr/bin/env python
"""Plots engagement for a given window."""

import csv
from datetime import date, timedelta

import numpy as np
from matplotlib import dates as mdates
from matplotlib import pyplot, ticker

import cov
import cov_utils
from conf import WIN

pyplot.style.use("mplrc")
grey = "#999999"

keywords = []
for topic in ["covid", "vaccine"]:
    with open(f"../data/keys-{topic}.txt", "rt") as fin:
        keywords += [line.rstrip() for line in fin]

keywords += ["covid__19", "covid19france", "covid_19fr", "covidー19"]


def check_event(day: str, win: int, ignore=None):
    """Check tweet content around an event.

    Parameters
    ----------
    day : date or str
        the day of interest
    win : int
        the window duration in days: [day, day + win)
    ignore : list
        the hashtags to ignore
    """
    if isinstance(day, str):
        day = date.fromisoformat(day)

    # data = cov.load_range(day - timedelta(days=win - 1), day + timedelta(days=1))
    # we check the window following the day at which we measures a peak.
    data = cov.load_range(day, day + timedelta(days=win))

    rank = data.hashtags_rank()
    if ignore is not None:
        for d in ignore:
            del rank[d]
    print(", ".join([r[0] for r in rank.most_common(10)]))
    print()


def label_fmt(x, pos):
    if x == 0:
        return "0"
    if x % 1000000 == 0:
        return f"{x//100000}M"
    if x % 1000 == 0:
        return f"{x//1000:.0f}k"
    return f"{x}"


fig, axes = pyplot.subplots(
    nrows=3,
    figsize=(7, 3.8),
    sharex=True,
    gridspec_kw={
        "hspace": 0,
        "top": 0.82,
        "right": 0.94,
        "left": 0.08,
        "bottom": 0.10,
        "height_ratios": [0.2, 0.4, 0.4],
    },
)

data = cov_utils.load_csv(f"../data/engagement-tau_{WIN}.tsv", transpose=True)
data["day"] = [date.fromisoformat(day) for day in data["day"]]

high = [dt for dt, val in zip(data["day"], data["rt"]) if val >= 20]
cases = cov_utils.load_cases("FR", transpose=True)

ax = axes[2]
ax.plot(data["day"], data["rt"])
ax.plot(data["day"], data["rt_media"], zorder=-1, alpha=0.8)
ax.set(ylabel=r"$R_t$")
ax.set_ylim(-5, 110)
# ax.tick_params("x", rotation=30)


ax = axes[1]
ax2 = ax.twinx()
next(ax2._get_lines.prop_cycler)

(full,) = ax2.plot(
    data["day"],
    cov_utils.smooth_convolve(data["media"], window=7),
    linewidth=0.7,
    label="News media",
)
(eng,) = ax.plot(
    data["day"],
    cov_utils.smooth_convolve(data["eng"], window=7),
    linewidth=1,
    label="Vaccine-critical",
)

ax2.tick_params("y", color=full.get_color(), labelcolor=full.get_color())
ax.tick_params("y", color=eng.get_color(), labelcolor=eng.get_color())
ax.legend(handles=[eng, full], fontsize="small", labelcolor="#555555", frameon=False)
ax.set_ylabel("Engaged", color=eng.get_color())
ax.yaxis.set_major_formatter(ticker.FuncFormatter(label_fmt))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(label_fmt))

ax = axes[0]

yes = np.array(cases["New_cases"])
xes = np.array(cases["day"])
yes = np.array([x if x is not None and not isinstance(x, str) else 0 for x in yes])

ax.fill_between(
    xes,
    cov_utils.smooth_convolve(yes, window=7),
    color=grey,
    alpha=0.5,
    linewidth=0,
)
ax.set_ylabel("New cases")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(label_fmt))
secax = ax.secondary_xaxis("top")
secax.set_xticks(list(high) + [date.fromisoformat("2020-12-30")])
secax.set_xticklabels(
    [d.isoformat() for d in high] + ["first vac."],
    rotation=90,
    ha="center",
    va="bottom",
    fontdict={"fontsize": 7},
)

for locks in cov_utils.load_csv("../data/lockdowns.tsv"):
    ax.axvspan(
        date.fromisoformat(locks["start"]),
        date.fromisoformat(locks["end"]),
        alpha=0.2,
        color="grey",
    )

for dt in high:
    print(dt)
    check_event(dt, WIN, ignore=keywords)
    for ax in axes:
        ax.axvline(dt, color=grey, alpha=0.5, zorder=-1)

for ax in axes:
    ax.axvline(date.fromisoformat("2020-12-30"), color=grey, alpha=0.5, zorder=-1)

for ax in axes[1:]:
    ax.axvspan(date(2020, 1, 1), date(2020, 11, 11), alpha=0.2, color=grey)
    ax.axvspan(date(2021, 6, 1), date(2021, 10, 1), alpha=0.2, color=grey)

fig.align_ylabels(axes)

locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator, formats=["%b\n%Y", "%b", "", "", "", ""])
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

with open(f"../data/engagement-events-tau_{WIN}.tsv", "wt") as fout:
    writer = csv.writer(fout, dialect=csv.excel_tab)
    writer.writerow(["event", "day"])
    writer.writerows([(f"event-{i}", day) for i, day in enumerate(high)])

pyplot.savefig("plot_engagement.pdf")
