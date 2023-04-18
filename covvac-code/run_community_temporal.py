#!/usr/bin/env python3
"""Compute the probability of finding a critic or media tweet in communities.

And they outreach probabilities.
"""

import json
from collections import Counter
from datetime import date, timedelta

import numpy as np
from scipy import sparse

import cov
import cov_utils


def read_communities(tau):
    """Read community structure."""
    with open(f"data/adjacency-tau_{tau}_nodes.json", "rt") as fin:
        udata = {user["id"]: user for user in json.load(fin)["nodes"]}

    comms = {}
    for user in udata.values():
        comms.setdefault(user["part"], set()).add(user["id"])

    return comms, udata


def month_cycle(month1="2020-01", month2="2021-10"):
    """Month by month date ranges."""
    year, month = map(int, month1.split("-"))

    def formt(year, month):
        return f"{year}-{month:02d}"

    def increment(year, month):
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        return year, month

    while formt(year, month) != month2:
        yield formt(year, month) + "-01", formt(*increment(year, month)) + "-01"
        year, month = increment(year, month)


def win_cycle(win=30, step=7):
    """Sliding window."""
    for day in cov_utils.daterange(
        "2020-01-01", date.fromisoformat("2021-10-01") - timedelta(days=win), step=step
    ):
        yield day.isoformat(), (day + timedelta(days=win)).isoformat()


def period_cycle():
    """Return the three main periods."""
    dates = ['2020-01-01', '2020-11-08', '2021-06-01', '2021-10-01']
    return zip(dates, dates[1:])


def main(tau=-1):
    """Do the main."""
    print(f"============\nTAU {tau}\n============")
    communities, ucomms = read_communities(tau)

    best_comms = Counter({k: len(v) for k, v in communities.items()})
    best_comms = {k: i for i, (k, _) in enumerate(best_comms.most_common(50))}

    periods = list(month_cycle())
    periods = list(win_cycle())
    periods = list(period_cycle())

    # extract data that only contains given urls
    urls = cov.Urls()
    output = {
        "probs": {
            kind: np.zeros((len(best_comms) + 1, len(best_comms) + 1, len(periods)))
            #                           ^^^^^ fake node to collect all other communities
            for kind in ["full", "crit", "media"]
        },
        "steady": {
            kind: np.zeros((len(best_comms) + 1, len(periods)))
            #                           ^^^^^ fake node to collect all other communities
            for kind in ["full", "crit", "media"]
        },
        "active": {
            kind: np.zeros((len(best_comms) + 1, len(periods)))
            for kind in ["full", "crit", "media"]
        },
    }
    for it, (day1, day2) in enumerate(periods):
        print(day1, day2)

        # load data
        data = {}
        data["full"] = cov.load_range(day1, day2)
        data["crit"] = data["full"].extract(
            tweet_filter=lambda x: {"PRE", "POST"} & set(urls.is_coded(x))
        )
        data["media"] = data["full"].extract(
            tweet_filter=lambda x: "MEDIA" in set(urls.is_coded(x))
        )

        # compute probs
        for kind, dt in data.items():
            print(kind)
            transition, steadystate, umap, _ = cov.adjacency(
                dt, tau=tau, fix_sources="symmetrize", return_factors=True
            )
            adj = transition @ sparse.diags(steadystate)
            # map user_index to comm_index
            umap = {
                i: best_comms[ucomms[uid]["part"]]
                if ucomms[uid]["part"] in best_comms
                else len(best_comms)
                for i, uid in umap.items()
            }
            best_comm_prj = sparse.csr_matrix(
                (
                    np.ones(len(umap)),
                    (list(umap.keys()), list(umap.values())),
                ),
                shape=(adj.shape[0], len(best_comms) + 1),
            )
            output["probs"][kind][:, :, it] = (
                best_comm_prj.T @ adj @ best_comm_prj
            ).toarray()
            output["steady"][kind][:, it] = best_comm_prj.T @ steadystate
            output["active"][kind][:, it] = best_comm_prj.sum(0)

    for kind, tensor in output["probs"].items():
        np.savez_compressed(
            f"data/community_temporal_periods_{kind}_{tau}.npz",
            probs=tensor,
            steady=output["steady"][kind],
            active=output["steady"][kind],
            comms=list(best_comms.keys()),
            periods=periods,
        )


if __name__ == "__main__":
    for tau in [-1, 0, 1]:
        main(tau)
        exit()
