#!/usr/bin/env python
"""Utility functions."""

import csv
import gzip
import pathlib
import subprocess as sp
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

import networkx as nx
import numpy as np
import pycountry
import pygenstability as stability
from scipy import sparse
from tqdm import tqdm

import cov_utils as utils

COUNTRIES = {
    "CM",
    "FR",
    "CD",
    "CA",
    "MG",
    "CI",
    "NE",
    "BF",
    "ML",
    "SN",
    "TD",
    "GN",
    "RW",
    "BE",
}
COUNTRIES = {c: pycountry.countries.get(alpha_2=c) for c in COUNTRIES}


class Tweets(dict):
    """A dictionary of tweets."""

    def __init__(self):
        """Initialize."""
        super().__init__(self)

        self.__keys__ = []
        self.__retweets__ = {}
        self.__retweet_keys__ = {}

    def update_from(self, fields, tweet_data):
        """Import tweets.

        Parameters
        ----------
        fields : list
            list of keys
        tweet_data : iterable
            iterable of dicts (each representing a tweet)

        Returns
        -------
        users : dict
            users IDs involved with corresponding tweet ids
        """
        self.__keys__ = list(fields)
        users = {}
        for tweet in tweet_data:
            tweet = utils.values2number(tweet)
            tid = tweet["id"]
            self.setdefault(tid, tweet)

            uid = tweet["from_user_id"]
            users.setdefault(uid, []).append(tid)

        return users

    def update_retweets_from(self, fields, retweet_data):
        """Import tweets.

        Parameters
        ----------
        fields : list
            list of keys
        retweet_data : iterable
            iterable of dicts (each representing a retweet)

        Returns
        -------
        users : dict
            users IDs involved with corresponding retweet IDs
        """
        self.__retweet_keys__ = list(fields)
        users = {}
        for retweet in retweet_data:
            retweet = utils.values2number(retweet)

            # get the user that tweeted
            tid = retweet["retweeted_id"]
            if tid not in self:
                # ignore retweets which original tweet is absent
                continue
            self.__retweets__.setdefault(tid, []).append(tuple(retweet.values()))

            uid = retweet["from_user_id"]
            users.setdefault(uid, []).append((tid, retweet["id"]))

        return users

    def iter(self):
        """Iterate over tweets.

        Yields
        ------
        tweets : tuple
            tuple of (Tweet ID, Tweet data, retweet list)
        """
        yield from [(tid, t, self.retweets(tid)) for tid, t in self.items()]

    def retweets(self, tid=None):
        """Return a list of retweets as dicts."""
        if tid is None:
            return [
                dict(zip(self.__retweet_keys__, rts))
                for rtss in self.__retweets__.values()
                for rts in rtss
            ]

        return [dict(zip(self.__retweet_keys__, rts)) for rts in self.__retweets__.get(tid, [])]

    def n_retweets(self, tid=None):
        """Return the number of retweets."""
        if tid is None:
            return {tid: len(self.__retweets__.get(tid, [])) for tid in self}
        return len(self.__retweets__.get(tid, []))

    def urls(self, retweets=False):
        """Return the number of times each urls has been used.

        Parameters
        ----------
        retweets : bool
             (Default value = False)
             If set to True, add the retweets in the count

        Returns
        -------
        urls : collections.Counter
            the counts.
        """
        urls = Counter()
        for tid, tweet in self.items():
            url = Urls.netloc(tweet.get("links", ""))
            if url != "":
                if retweets:
                    urls[url] += self.n_retweets(tid=tid) + 1
                else:
                    urls[url] += 1

        return urls

    @property
    def nt(self):
        """Return the number of tweets.

        Returns
        -------
        ntweets : int
            number of tweets
        """
        return len(self)

    @property
    def nrt(self):
        """Return the number of retweets.

        Returns
        -------
        nretweets : int
            number of retweets
        """
        return sum([len(rts) for _, _, rts in self.iter()])


class Users(dict):
    """A dictionary of users."""

    def update_from(self, users: dict, fields: list, user_data):
        """Import tweets.

        Parameters
        ----------
        users : dict
            set of user IDs to restrict to.
        fields : list
            list of keys
        user_data : iterable
            iterable of dicts (each representing a tweet)
        """
        self.__keys__ = list(fields)
        for user in user_data:
            user = utils.values2number(user)
            uid = user["from_user_id"]
            if uid in users["tweets"] or uid in users["retweets"]:
                # store tweets and retweets done by user
                self.setdefault(uid, user)
                self[uid].setdefault("data_tweets", []).extend(users["tweets"].get(uid, []))
                self[uid].setdefault("data_retweets", []).extend(users["retweets"].get(uid, []))

        for uid in users["tweets"]:
            assert uid in self, f"Some users are not present. {uid}"
        for uid in users["retweets"]:
            assert uid in self, f"Some users are not present. {uid}"

    def h_index(self, tweets: Tweets):
        """Return the H-index of users.

        Number of tweets with at least the same number of retweets.

        Parameters
        ----------
        tweets : Tweets

        Returns
        -------
        rank : Counter
            the rank of the users
        """
        rank = Counter(
            {
                uid: hindex([tweets.n_retweets(tid=tid) for tid in u["data_tweets"]])
                for uid, u in self.items()
            }
        )
        return +rank


@dataclass
class Data:
    """Database of all tweets, retweets and users."""

    users: Users = field(default_factory=Users)
    tweets: Tweets = field(default_factory=Tweets)

    def add_day(self, day):
        """Load a day from the database.

        Parameters
        ----------
        day : str or date
            the day
        """
        if isinstance(day, (date,)):
            day = day.isoformat()

        user_ids = {}

        # add tweets
        fname = utils.CORPUS / f"{day}-tweets.csv.gz"
        user_ids["tweets"] = self.__tweets_from_file__(fname)

        # add retweets
        fname = utils.CORPUS / f"{day}-retweets.csv.gz"
        user_ids["retweets"] = self.__retweets_from_file__(fname)

        # add users
        fname = utils.CORPUS / f"{day}-users.csv.gz"
        self.__users_from_file__(fname, user_ids)

    def hashtags(self):
        """Return hashtags with corresponding tweets."""
        hts = [
            (ht, tid)
            for tid, tweet in self.tweets.items()
            for ht in tweet.get("hashtags", "").split("|")
        ]
        hts_dict = {}
        for ht, tid in hts:
            hts_dict.setdefault(ht, []).append(tid)
        return hts_dict

    def hashtags_rank(self):
        """Return the h-index of hashtags."""
        t_len = self.tweets.n_retweets()
        hts = {ht: [t_len[tid] for tid in tids] for ht, tids in self.hashtags().items()}

        rank = Counter({ht: hindex(tlen) for ht, tlen in hts.items()})
        del rank[""]
        return rank

    def extract(self, tweet_filter=None, user_filter=None):
        """Extract a subset of tweets.

        Parameters
        ----------
        tweet_filter :
             (Default value = None)
        user_filter :
             (Default value = None)

        Returns
        -------
        data : Data
            a database with extracted tweets and users.
        """
        if tweet_filter is None:

            def tweet_filter(x):
                return True

        if user_filter is None:

            def user_filter(x):
                return True

        data = Data()

        users = {}
        users["tweets"] = data.tweets.update_from(
            self.tweets.__keys__,
            [
                tweet
                for tweet in self.tweets.values()
                if tweet_filter(tweet) and user_filter(tweet["from_user_id"])
            ],
        )
        tids = list(data.tweets.keys())
        users["retweets"] = data.tweets.update_retweets_from(
            self.tweets.__retweet_keys__,
            [
                rtw
                for tid in tids
                for rtw in self.tweets.retweets(tid)
                if user_filter(rtw["from_user_id"])
            ],
        )
        data.users.update_from(
            users,
            self.users.__keys__,
            [{ku: vu for ku, vu in u.items() if ku[:4] != "data"} for u in self.users.values()],
        )
        return data

    def __tweets_from_file__(self, fname: pathlib.Path):
        users = {}
        if fname.is_file():
            with gzip.open(fname, "rt") as fin:
                reader = csv.DictReader(fin)
                users = self.tweets.update_from(reader.fieldnames, reader)
        return users

    def __retweets_from_file__(self, fname: pathlib.Path):
        users = {}
        if fname.is_file():
            with gzip.open(fname, "rt") as fin:
                reader = csv.DictReader(fin)
                users = self.tweets.update_retweets_from(reader.fieldnames, reader)
        return users

    def __users_from_file__(self, fname: pathlib.Path, user_ids: dict):
        if fname.is_file():
            with gzip.open(fname, "rt") as fin:
                reader = csv.DictReader(fin)
                self.users.update_from(user_ids, reader.fieldnames, reader)

    def __str__(self):
        return f"""Tweet data
        tweets:   {len(self.tweets)} / {len(self.tweets.__retweets__)}
        retweets: {self.tweets.nrt}
        users:    {len(self.users)}
"""


@dataclass
class Urls:
    """Load coded URLs."""

    pre: set = field(default_factory=set)
    post: set = field(default_factory=set)
    media: set = field(default_factory=set)
    codes: list = field(default_factory=list)

    def __post_init__(self):
        """Initialize Urls bag."""
        # load from file
        urls = utils.load_csv(utils.BASENAME / "data/coding_urls.csv")
        self.pre |= {url["Label"].lower() for url in urls}

        urls = utils.load_csv(utils.BASENAME / "data/coding_last.tsv")
        for row in urls:
            if row["code"] == "COVID":
                self.post.add(row["url"].lower())
            else:
                self.pre.add(row["url"].lower())

        urls = utils.load_csv(utils.BASENAME / "data/media-nosocial.txt", headers=False)
        self.media |= {url[0].lower() for url in urls}

        self.codes = ["PRE", "POST", "MEDIA"]

    @staticmethod
    def netloc(url: str):
        """Return the base url.

        Stripped of:
        - 'http[s]://'
        - 'www.'
        - anything after the first '/'
        - anything after the first '?'

        basically a netloc without 'www.'

        Parameters
        ----------
        url : str
            the URL to check

        Returns
        -------
        netloc : str
            the netloc of the given URL
        """
        if url[:4] == "http":
            netloc = url.split("/", maxsplit=3)[2]
        else:
            netloc = url.split("/", maxsplit=1)[0]
        netloc = utils.removeprefix(netloc, "www.")

        return netloc.split("?")[0]

    def is_coded(self, tweet: dict):
        """Return if the tweet contins a coded link.

        Parameters
        ----------
        tweet : dict
            the tweet data (should contain `links` as key)

        Returns
        -------
        codes : list
            list of codes (strings with values `PRE`, `POST` or `MEDIA`)
        """
        if tweet.get("links", "") == "":
            return []

        coded = set()
        for url in tweet["links"].split("|"):
            netloc = self.netloc(url.lower())
            if netloc in self.pre:
                coded.add("PRE")
            elif netloc in self.post:
                coded.add("POST")
            elif netloc in self.media:
                coded.add("MEDIA")
        return coded


def load_day(day, data=None, strict=False):
    """Load a day from the database.

    Parameters
    ----------
    day : str or date
        the day
    data : Data
        if provided, append to this data (Default value = None)
    strict : bool
        whether to load also retweets from a different day. (Default value = False)

    Returns
    -------
    data : Data
    """
    if data is None:
        data = Data()

    data.add_day(day)

    return data


def load_range(day1, day2=date.today(), strict=False):
    """Load a range of days.

    Parameters
    ----------
    day1 : str or date
        First day
    day2 : str or date, default: today
        Last day (excluded) (Default value = date.today())
    strict : bool
        NotImplemented (Default value = False)

    Returns
    -------
    data : Data
        data
    """
    data = Data()

    ranger = tqdm(utils.daterange(day1, day2), leave=False)
    for day in ranger:
        ranger.set_description(desc=str(day))
        data.add_day(day)

    return data


def directed_partition(graph):
    """Wrap Stability at t=1 for pygenstability.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to partition.

    Returns
    -------
    partition : dict
        Dictionary of nodes -> partition.
        Partition tags are sorted by partition size.
    """
    i2n = dict(enumerate(graph.nodes()))
    adj = nx.to_scipy_sparse_matrix(graph, weight="weight", format="csr")

    parts = stability.run(adj, n_workers=8, times=[1.0])
    # returns a list of community tags per node.

    return {i2n[node]: partition for node, partition in enumerate(parts["community_id"][0])}


def directed_partition_louvain(graph):
    """Wrap DirectedLouvain.

    directed Louvain algorithm from: https://github.com/nicolasdugue/DirectedLouvain

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to partition.

    Returns
    -------
    partition : dict
        Dictionary of nodes -> partition.
        Partition tags are sorted by partition size.
    """
    n2i = {n: i for i, n in enumerate(graph.nodes())}
    i2n = dict(enumerate(graph.nodes()))

    louvainbin = pathlib.Path("~/codes/DirectedLouvain/bin").expanduser()

    tmp = tempfile.NamedTemporaryFile(mode="wt", delete=True)

    for e in graph.edges(data="weight"):
        print(n2i[e[0]], n2i[e[1]], e[2] if e[2] is not None else 1.0, file=tmp)

    tmp.flush()
    tmpfile = pathlib.Path(tmp.name)

    binfile = tmpfile.with_suffix(".bin")
    weightfile = tmpfile.with_suffix(".weight")

    cmd = [louvainbin / "convert", "-i", tmp.name, "-o", binfile, "-w", weightfile]
    sp.run(cmd, capture_output=True)
    tmp.close()  # close and remove the temporary file

    cmd = [louvainbin / "community", binfile, "-l", "-1", "-w", weightfile]
    job = sp.run(cmd, capture_output=True)
    binfile.unlink()  # remove binary file
    weightfile.unlink()  # remove weight file

    # resolve last level of the dendrogram.
    new_part = {
        n: {
            n,
        }
        for n in i2n
    }
    part = {}
    for row in job.stdout.decode().splitlines():
        node, partition = map(int, row.split())
        if node == 0:
            assert len(part) == 0
            part = new_part.copy()
            new_part = dict()

        new_part.setdefault(partition, set())
        new_part[partition] |= part.pop(node)

    new_part = sorted(new_part.values(), key=len, reverse=True)
    part = {i2n[node]: partition for partition, nodes in enumerate(new_part) for node in nodes}

    # sometimes I loose one node
    if len(part) < len(graph):
        for node in set(graph.nodes()) - set(part.keys()):
            part[node] = len(part)

    return part


def hyper_edges(data: Data):
    """Compute the hyper_edges of the system.

    Parameters
    ----------
    data : Data
        system database

    Returns
    -------
    usermap : dict
        users that should go to the adjacency. May be more or less than those in data
    tails : sparse
        entering nodes of each hyper-edge
    heads : sparse
        exit nodes of each hyper-edge
    """
    # user id -> index
    # removing users that never get retweeted
    uids = set(t["from_user_id"] for _, t, rts in data.tweets.iter() if len(rts) > 0) | set(
        rt["from_user_id"] for _, _, rts in data.tweets.iter() for rt in rts
    )
    # nodemap = {uid: uindx for uindx, uid in enumerate(data.users)}
    nodemap = {uid: uindx for uindx, uid in enumerate(uids)}

    # each node is a user
    nnodes = len(nodemap)
    print("BUILDING HYPER-GRAPH: node number ->", nnodes)
    # each tweet is an hyper edge
    tweets = [
        (tid, t["from_user_id"], [rt["from_user_id"] for rt in rts])
        for tid, t, rts in data.tweets.iter()
        if len(rts) > 0  # if there are retweets
    ]
    nedges = len(tweets)
    print("BUILDING HYPER-GRAPH: hyper-edge number ->", nedges)

    # entry point of the hyper edges (tail of the arrow)
    _tails = sparse.coo_matrix(
        (np.ones(nedges), ([nodemap[uid] for _, uid, _ in tweets], np.arange(nedges))),
        shape=(nnodes, nedges),
    )

    retweets = [(tindx, rt_uid) for tindx, (_, _, rts) in enumerate(tweets) for rt_uid in rts]
    print("BUILDING HYPER-GRAPH: hyper-edge total size ->", len(retweets))

    # exit point of the hyper edges (head of the arrow)
    _heads = sparse.coo_matrix(
        (
            np.ones(len(retweets)),
            (
                [nodemap[rt_uid] for _, rt_uid in retweets],
                [tindx for tindx, _ in retweets],
            ),
        ),
        shape=(nnodes, nedges),
    )
    return (nodemap, _tails.tocsc(), _heads.tocsc())


def __interaction_adjacency__(tails, heads, tau=-1):
    """Compute the weights of the interaction adjacency matrix.

    Only the corresponding transition matrix should be considered.

    Parameters
    ----------
    tails : sparse matrix

    heads : sparse matrix

    tau : {-1, 0, 1}
        parameter
        tau = 0 -> project each hyper_edge to a clique
        tau = -1 -> each hyper edge is selected with the same prob (independently from cascade size)
        tau = 1 -> hyper edges with larger cascades are more probable.
        (Default value = -1)

    Returns
    -------
    adjacency matrix : sparse
    """
    # B_{\alpha, \alpha}
    # get the exit size of each hyper edge (number of vertices involved)
    hyper_weight = heads.sum(0).A1

    # here we may have zeros entries for tweets that have never been retweeted
    hyper_weight[hyper_weight > 0] = hyper_weight[hyper_weight > 0] ** tau
    # put that on a diagonal matrix
    hyper_weight = sparse.diags(hyper_weight, offsets=0)

    print(tails.shape)
    print(hyper_weight.shape)
    print(heads.shape)
    # compute the tails -> heads weighted entries (propto probability if symmetric)
    return tails @ hyper_weight @ heads.T


def __fix_sources__(matrix: sparse.csr_matrix):
    """Add a `Twitter basin` node to keep the dynamics going.

    This add a node (last one).

    Parameters
    ----------
    matrix : sparse.csr_matrix
        matrix to fix

    Returns
    -------
    fixed_matrix : sparse.csr_matrix
        fixed matrix
    newnode : int
        index of the new node (usually last one).
    """
    assert matrix.shape[0] == matrix.shape[1]
    nnodes = matrix.shape[0]

    out_degree = np.asarray(matrix.sum(1)).flatten()

    # find sinks: out_degree == 0
    sinks = np.argwhere(out_degree == 0).flatten()

    # find sources: out_degree != 0
    sources = np.argwhere(out_degree != 0).flatten()

    # add a node at the end
    matrix._shape = (nnodes + 1, nnodes + 1)
    matrix.indptr = np.hstack((matrix.indptr, matrix.indptr[-1]))

    # add link from sinks to basin
    sink2basin = sparse.csr_matrix(
        (
            # it doesn't matter the value, we will compute the transition matrix from thid
            np.ones_like(sinks),
            (sinks, np.full_like(sinks, nnodes)),
        ),
        shape=matrix.shape,
    )
    basin2sources = sparse.csr_matrix(
        (
            # it doesn't matter the value, we will compute the transition matrix from thid
            out_degree[sources],
            (np.full_like(sources, nnodes), sources),
        ),
        shape=matrix.shape,
    )
    return matrix + sink2basin + basin2sources, nnodes


def __strip_basin__(matrix: sparse.csr_matrix, vector: np.array, basin=None):
    """Strip last node (column and row) of the matrix."""
    assert matrix.shape[0] == matrix.shape[1]

    if basin is None:
        # assume the last node
        basin = matrix.shape[0] - 1

    keep = np.arange(matrix.shape[0])
    keep = keep[keep != basin]

    return extract_components(matrix, keep), vector[keep]


def adjacency(
    data: Data,
    tau: int = -1,
    fix_sources: str = "basin",
    return_factors=False,
    symmetrize_weight=0.1,
    return_intraction=False,
):
    r"""Return the adjacency matrix of the data.

    It picks the strongly connected component and return

    .. math::
        \Pi T

    where :math:`\Pi` is the diagonal matrix of the steadystate and
    :math:`T` is the transition matrix.
    In this way all edge weights are the probability of being traversed.

    Parameters
    ----------
    data : DataBase or tuple
        database of tweets or tuple(user2id_map, tails heads)
    tau : int
        parameter.
        (Default value = -1)
    fix_sources : bool, default=True
        if a fix to source and sink nodes need to be applied.
        This will performed adding a fake `basin` node as a bridge between sink and source nodes.
        It will be removed before return. (Default value = True)
    return_factors :
         (Default value = False)

    Returns
    -------
    adjacency : sparse
        the adjacency matrix
    umap : dict
        map of index to user IDs
    other_components : list of lists
        list of components other that the largest.
    """
    # compute hyper_edges (tails and heads)
    if isinstance(data, tuple):
        ui_map, tails, heads = data
    else:
        ui_map, tails, heads = hyper_edges(data)
    iu_map = {i: u for u, i in ui_map.items()}

    # put everything in a matrix (interaction matrix)
    weighted_adj = __interaction_adjacency__(tails, heads, tau=tau)
    if fix_sources == "basin":
        # add a fake node to ensure the ergodicity
        weighted_adj, basin = __fix_sources__(weighted_adj)
    elif fix_sources == "symmetrize":
        weighted_adj += symmetrize_weight * weighted_adj.T

    # extract the largest connected component
    comps = find_components(weighted_adj, kind="strong")
    assert sum([len(c) for c in comps]) == weighted_adj.shape[0]
    if fix_sources == "basin":
        assert basin in comps[0]
    weighted_adj = extract_components(weighted_adj, comps[0])

    if return_intraction:
        return (
            weighted_adj,  # the adjacency matrix
            {i: iu_map[cind] for i, cind in enumerate(comps[0])},  # the node IDs
            [[iu_map[i] for i in comp] for comp in comps[1:]],  # nodes discarted
        )

    # compute the transition matrix and the steady state.
    transition, steadystate = compute_transition_matrix(
        weighted_adj,
        return_steadystate=True,
        niter=10000,
    )
    del weighted_adj

    if fix_sources == "basin":
        # remove the fake `basin` node
        transition, steadystate = __strip_basin__(
            transition, steadystate.A1, basin=comps[0].index(basin)
        )
        # renormalize rows
        marginal = transition.sum(0).A1

        # add a self loop to the nodes without outlinks.
        transition += sparse.diags((marginal == 0).astype(int))
        marginal[marginal == 0] = 1

        transition = transition @ sparse.diags(1 / marginal)
        # renormalize steadystate
        steadystate /= steadystate.sum()

        comps = [[c for c in comp if c != basin] for comp in comps]
        print(f"STRONG COMPONENT: removing {basin} as basin node.")
    else:
        steadystate = steadystate.A1
    assert len(comps[0]) == transition.shape[0]
    transition.eliminate_zeros()

    print(
        f"STRONG COMPONENT: we discarted {len(ui_map) - transition.shape[0]}"
        f" nodes {100 * (len(ui_map) - transition.shape[0]) / len(ui_map):4.2f}%."
    )
    print(f"STRONG COMPONENT: adjacency matrix of shape {transition.shape}")
    if return_factors:
        return (
            transition,
            steadystate,
            {i: iu_map[cind] for i, cind in enumerate(comps[0])},  # the node IDs
            [[iu_map[i] for i in comp] for comp in comps[1:]],  # nodes discarted
        )

    # new adjacency matrix as probability A_{ij} \propto p(i, j)
    new_adj = transition @ sparse.diags(steadystate)
    return (
        new_adj,  # the adjacency matrix
        {i: iu_map[cind] for i, cind in enumerate(comps[0])},  # the node IDs
        [[iu_map[i] for i in comp] for comp in comps[1:]],  # nodes discarted
    )


def compute_transition_matrix(matrix, return_steadystate=False, niter=10000):
    r"""Return the transition matrix.

    Parameters
    ----------
    matrix : sparse.spmatrix
        the adjacency matrix (square shape)
    return_steadystate : bool (default=False)
        return steady state. (Default value = False)
    niter : int (default=10000)
        number of iteration to converge to the steadystate. (Default value = 10000)

    Returns
    -------
    trans : np.spmatrix
        The transition matrix.
    v0 : np.matrix
        the steadystate
    """
    # marginal
    tot = matrix.sum(0).A1
    # fix zero division
    tot_zero = tot == 0
    tot[tot_zero] = 1
    # transition matrix
    trans = matrix @ sparse.diags(1 / tot)

    # fix transition matrix with zero-sum rows
    trans += sparse.spdiags(tot_zero.astype(int), 0, *trans.shape)

    if return_steadystate:
        v0 = matrix.sum(0)
        v0 = v0.reshape(1, matrix.shape[0]) / v0.sum()
        for i in range(niter):
            # evolve v0
            v1 = v0.copy()
            v0 = v0 @ trans.T
            if np.sum(np.abs(v1 - v0)) < 1e-7:
                break
        print(f"TRANS: performed {i} itertions.")

        return trans, v0

    return trans


def find_components(matrix, kind="strong"):
    """Return the components of the graph.

    Parameters
    ----------
    matrix : sparse.spmatrix
        the adjacency square matrix
    kind : str, default=`strong`
        either `strong` or `weak` (Default value = 'strong')

    Returns
    -------
    components : list
        sorted list of components (list of node indexes)
    """
    # check strongly connected component
    ncomp, labels = sparse.csgraph.connected_components(
        csgraph=matrix, directed=True, connection=kind
    )

    components = [[] for _ in range(ncomp)]
    for node, label in enumerate(labels):
        components[label].append(node)

    return sorted(components, key=len, reverse=True)


def extract_components(matrix: sparse.spmatrix, indexes: list):
    r"""Extract the sub matrix.

    Parameters
    ----------
    matrix : sparse.spmatrix
        the matrix (square)
    indexes : list
        list of indeces to retain

    Returns
    -------
    matrix : sparse.csc_matrix
        matrix with rows and columns removed.
    """
    return matrix.tocsr()[indexes, :].tocsc()[:, indexes]


def hindex(citations):
    """Compute the H-index.

    Parameters
    ----------
    citations : list
        the list of number of citations per paper.

    Returns
    -------
    hindex : int
        the H-index
    """
    srt = np.sort(citations)[::-1]
    return (srt >= np.arange(1, len(srt) + 1)).sum()
