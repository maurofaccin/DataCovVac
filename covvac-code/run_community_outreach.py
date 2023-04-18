#!/usr/bin/env python3
"""Compute the probability of finding a critic or media tweet in communities.

And they outreach probabilities.
"""

import json
from collections import Counter

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


def projector(data: cov.Data, ucomms: dict, tau: int = -1):
    """Project the adjacency matrix to the community structure.

    Parameters
    ----------
    data : cov.Data
        data to project, may be a tuple (transition, steadystate, nmap)
    ucomms : dict
        dict of users, IDs as keys, dicts as values.
        In the latter the partition is in the 'part' value

    Returns
    -------
    adj :  sparse.csr_matrix
        adjacency matrix of the projected network
    map : dict
        index to community name map
    """
    if isinstance(data, tuple):
        transition, steadystate, nmap = data
    else:
        transition, steadystate, nmap, _ = cov.adjacency(
            data, tau=tau, fix_sources="symmetrize", return_factors=True
        )

    adjacency = transition @ sparse.diags(steadystate)

    # map user indx to comm
    umap = {indx: ucomms[uid]["part"] for indx, uid in nmap.items()}
    # communities
    comms = set([u["part"] for u in ucomms.values()])
    csize = Counter([u["part"] for u in ucomms.values()])
    # comm to comm indes
    comm_indx = {
        comm: icomm
        for icomm, comm in enumerate(
            # sorted(comms, key=lambda x: (x[0], int(x.split("_")[0][1:])))
            sorted(comms, key=lambda x: csize[x])
        )
    }
    prj = sparse.csr_matrix(
        (
            np.ones(len(umap)),
            (list(umap.keys()), [comm_indx[c] for c in umap.values()]),
        )
    )

    p_ss = prj.T @ steadystate
    p_adj = prj.T @ adjacency @ prj

    # entry wise multiplication with identity
    p_in_in = p_adj.multiply(sparse.eye(p_adj.shape[0]))
    p_out = p_adj - p_in_in
    p_in_out = p_out.sum(0).A1
    p_out_in = p_out.sum(1).A1

    return {
        "part": comm_indx,
        "prob_steady": p_ss,
        "prob_in_in": p_in_in @ np.ones(p_adj.shape[0]),
        "prob_in_out": p_in_out,
        "prob_out_in": p_out_in,
    }


def main(tau=-1):
    """Do the main."""
    print(f"============\nTAU {tau}\n============")
    communities, ucomms = read_communities(tau)

    # extract data that only contains given urls
    urls = cov.Urls()
    data = {}
    data["full"] = cov.load_range("2020-01-01", "2021-10-01")
    # data["full"] = cov.load_range("2020-01-01", "2020-04-01")
    data["crit"] = data["full"].extract(
        tweet_filter=lambda x: {"PRE", "POST"} & set(urls.is_coded(x))
    )
    data["media"] = data["full"].extract(
        tweet_filter=lambda x: "MEDIA" in set(urls.is_coded(x))
    )

    out = {}
    for kind, dt in data.items():
        print(kind)
        output = projector(dt, ucomms, tau)
        for prob, vector in output.items():
            if prob == "part":
                key = "part"
            else:
                key = "_".join([kind, prob])

            out[key] = vector

    keys = list(out.keys())
    cov_utils.dump_csv(
        f"data/community_outreach-tau_{tau}.tsv",
        [dict(zip(keys, vals)) for vals in zip(*out.values())],
    )


if __name__ == "__main__":
    for tau in [-1, 0, 1]:
        main(tau)
