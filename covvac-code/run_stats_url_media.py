#!/usr/bin/env python3
"""
File: run_stats_url_media.py
Author: Mauro Faccin
Email: mauro@gmail.com
Description: Fraction of tweets and retweets with urls from the critics and media sets.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import tqdm
from scipy import sparse

import cov
import cov_utils


@dataclass
class CommStats:
    """Class for community stats."""

    ndays: int = 0

    data_comm: list = field(default_factory=list)
    data_feat: list = field(default_factory=list)
    comm2indx: dict = field(default_factory=dict)
    feat2indx: dict = field(default_factory=dict)

    timedata: dict = field(default_factory=dict)

    def update(self, part, feat, index):
        """Update inner state.

        Parameters
        ----------
        part : str
            partition class
        feat : str
            feature
        index : int
            temporal index
        """
        if feat == "":
            return

        self.comm2indx.setdefault(part, len(self.comm2indx))
        self.feat2indx.setdefault(feat, len(self.feat2indx))

        # temp counts of features
        self.timedata.setdefault(feat, np.zeros(self.ndays))
        self.timedata[feat][index] += 1

        # partition to feture link
        self.data_comm.append(part)
        self.data_feat.append(feat)

    def sparse(self):
        """Return a sparse matrix.

        Returns
        -------
        matrix : sparse.csr_matrix
            matrix of links from communities to features.
        """
        return sparse.csr_matrix(
            (
                np.ones_like(self.data_comm),
                (
                    [self.comm2indx[c] for c in self.data_comm],
                    [self.feat2indx[f] for f in self.data_feat],
                ),
            ),
            shape=(len(self.comm2indx), len(self.feat2indx)),
            dtype=int,
        )

    def sort(self, communities, features):
        self.comm2indx = {c: i for i, c in enumerate(communities)}
        self.feat2indx = {f: i for i, f in enumerate(features)}

    def size(self):
        return len(self.data_comm)

    def temporal(self):
        inv_map = {v: k for k, v in self.feat2indx.items()}
        return np.stack([self.timedata[inv_map[i]] for i in sorted(inv_map)])


def iter_daily(day1, day2):
    """Iterate over tweets and retweets sorting by `created_at`.

    Parameters
    ----------
    day1 : str or date
        day
    day2 : str or date
        day

    Returns
    -------
    output : list
        tweets and retweets.
    """
    urls = cov.Urls()
    dates = {d.isoformat(): id for id, d in enumerate(cov_utils.daterange(day1, day2))}
    tweets = np.zeros((3, len(dates)))
    retweets = np.zeros((3, len(dates)))

    comm_stats = [
        {
            k: CommStats(len(dates))
            for k in ["urls_pre", "urls_post", "urls_media", "hashtags"]
        }
        for _ in range(3)
    ]

    partition = []
    for tau in [-1, 0, 1]:
        with open(f"data/adjacency-tau_{tau}_nodes.json", "rt") as fin:
            partition.append({u["id"]: u["part"] for u in json.load(fin)["nodes"]})

    urls_set = set()
    hash_set = set()

    for day in tqdm.tqdm(dates):
        daily = cov.load_day(day)

        for tid, tweet, rtws in daily.tweets.iter():
            for i, (indx, isoday) in enumerate(check_urls(tweet, rtws, urls)):
                if i == 0:
                    tweets[indx, dates[day]] += 1
                else:
                    try:
                        retweets[indx, dates[isoday]] += 1
                    except KeyError:
                        pass

            for kind, url, isoday, uid in check_url_stats(tweet, rtws, urls):
                urls_set.add(url)
                for ipart, part in enumerate(partition):
                    if uid in part and isoday in dates:
                        comm_stats[ipart][kind].update(part[uid], url, dates[isoday])

            for hashtag, isoday, uid in check_hash_stats(tweet, rtws):
                hash_set.add(hashtag)
                for ipart, part in enumerate(partition):
                    if uid in part and isoday in dates:
                        comm_stats[ipart]["hashtags"].update(
                            part[uid], hashtag, dates[isoday]
                        )

    urls_set = sorted(urls_set)
    hash_set = sorted(hash_set)
    for stats, part in zip(comm_stats, partition):
        part_set = sorted(set(part.values()))
        for kind, stat in stats.items():
            if kind[:3] == "url":
                stat.sort(part_set, urls_set)
            else:
                stat.sort(part_set, hash_set)

    return (dates, tweets, retweets), comm_stats


def check_urls(tweet, rtws, urls):
    url = set(urls.is_coded(tweet))

    if len(url) == 0:
        indx = 0
    elif "PRE" in url or "POST" in url:
        indx = 1
    elif "MEDIA" in url:
        indx = 2

    yield indx, tweet["created_at"][:10]
    for rtw in rtws:
        yield indx, rtw["created_at"][:10]


def check_url_stats(tweet, rtws, urls):
    for url in tweet.get("links", "").split("|"):
        url = urls.netloc(url.lower())
        if url in urls.pre:
            kind = "urls_pre"
        elif url in urls.post:
            kind = "urls_post"
        elif url in urls.media:
            kind = "urls_media"
        else:
            kind = None

        if kind is not None:
            yield kind, url, tweet["created_at"][:10], tweet["from_user_id"]
            for rtw in rtws:
                yield kind, url, rtw["created_at"][:10], rtw["from_user_id"]


def check_hash_stats(tweet, rtws):
    for hash in tweet.get("hashtags", "").split("|"):
        yield hash, tweet["created_at"][:10], tweet["from_user_id"]
        for rtw in rtws:
            yield hash, rtw["created_at"][:10], rtw["from_user_id"]


def main():
    """Do the main."""
    (days, tweets, retweets), comm_stats = iter_daily("2020-01-01", "2021-10-01")

    cov_utils.dump_csv(
        "data/daily_stats.tsv",
        [
            {
                "day": day,
                "twt_nourls": t[0],
                "twt_crit": t[1],
                "twt_media": t[2],
                "rtwt_nourls": rt[0],
                "rtwt_crit": rt[1],
                "rtwt_media": rt[2],
            }
            for day, t, rt in zip(days, tweets.T, retweets.T)
        ],
    )

    for tau, data in zip([-1, 0, 1], comm_stats):
        for key, dt in data.items():
            print(tau, key)
            sparse.save_npz(f"data/stats_comm_tau_{tau}_{key}.npz", dt.sparse())
            with open(f"data/stats_comm_tau_{tau}_{key}_map.json", "wt") as fout:
                json.dump({"comm2indx": dt.comm2indx, "feat2indx": dt.feat2indx}, fout)

            keys = list(dt.timedata.keys())
            cov_utils.dump_csv(
                f"data/stats_comm_tau_{tau}_{key}_temp.tsv.gz",
                [
                    dict(zip(keys, vals))
                    for vals in np.column_stack(list(dt.timedata.values()))
                ],
            )


if __name__ == "__main__":
    main()
