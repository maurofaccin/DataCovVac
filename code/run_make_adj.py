#!/usr/bin/env python
"""Load and save and adjacency matrix.

Load adjacency matrix and define partitions.

Compute:
-------
- adjacency matrix
- community structure

Todo:
----
- country discovery
"""

import csv
import json
import pathlib

import networkx as nx
import numpy as np
from scipy import sparse

import cov


def main():
    """Do the MAIN."""
    data = cov.load_range("2020-01-01", "2021-10-01")
    out_folder = pathlib.Path("data")

    for tau in [-1, 0, 1]:
        print(r": ", tau)
        # check vaccine critic adjacency
        # adjacency, nodes, discarted = cov.adjacency(data, tau=tau, fix_sources=True)
        transition, sstate, nodes, discarted = cov.adjacency(
            data, tau=tau, fix_sources="symmetrize", return_factors=True
        )
        txt = ""
        adjacency = transition @ sparse.diags(sstate)
        sparse.save_npz(out_folder / f"adjacency{txt}-tau_{tau}_matrix.npz", adjacency)
        sparse.save_npz(
            out_folder / f"adjacency{txt}-tau_{tau}_transition.npz", transition
        )
        np.savez_compressed(
            out_folder / f"adjacency-tau_{tau}_steadystate.npz", steadystate=sstate
        )

        # check full graph
        graph = nx.from_scipy_sparse_matrix(adjacency, create_using=nx.DiGraph)
        community = cov.directed_partition(graph)
        nodes = [
            {"id": nid, "part": f"C{community[nindx]}", "indx": nindx}
            for nindx, nid in nodes.items()
        ] + [
            {"id": nid, "part": f"X{i}", "indx": None}
            for i, comp in enumerate(discarted)
            for nid in comp
        ]

        with open(out_folder / f"adjacency{txt}-tau_{tau}_nodes.json", "wt") as fout:
            json.dump({"nodes": nodes, "discarted": discarted}, fout)

        inv_nodes = {
            node["indx"]: node["id"] for node in nodes if node["indx"] is not None
        }
        edgelist = [
            (inv_nodes[e1], inv_nodes[e2], w if w is not None else 1.0)
            for e1, e2, w in graph.edges(data="weight")
        ]

        with open(out_folder / f"adjacency{txt}-tau_{tau}_edgelist.tsv", "wt") as fout:
            writer = csv.writer(fout, dialect=csv.excel_tab)
            writer.writerows(edgelist)


if __name__ == "__main__":
    main()
