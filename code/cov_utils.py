#!/usr/bin/env python3
"""Utilities.

File: cov_utils.py
Author: Mauro
Github: https://github.com/maurofaccin
Description: Utility functions for analysis
"""

import csv
import gzip
import pathlib
from datetime import date, timedelta

import networkx as nx
import numpy as np
from community import best_partition

BASENAME = pathlib.Path(__file__)
if BASENAME.is_symlink():
    BASENAME = BASENAME.readlink()
BASENAME = BASENAME.parent
CORPUS = BASENAME / "corpus_merged"

TWEET_FIELDS = [
    ("id",),
    ("time", "timestamp_utc"),
    ("created_at", "local_time"),
    ("text",),
    ("from_user_name", "user_name"),
    ("from_user_id", "user_id"),  # OP of the tweet
    ("to_user_id", "to_userid"),
    ("quoted_id",),
    ("quoted_user_id",),
    ("mentioned_user_ids", "mentioned_ids"),
    ("location", "user_location"),
    ("links",),
    ("hashtags",),
]

RETWEET_FIELDS = [
    ("id",),
    ("retweeted_id",),
    ("time", "timestamp_utc"),
    ("from_user_id", "user_id"),  # OP of the retweet
    ("created_at", "local_time"),
]

USER_FIELDS = [
    ("from_user_name", "user_name"),
    ("from_user_id", "user_id"),
    ("from_user_description", "user_description"),
    ("from_user_url", "user_url"),
    ("from_user_realname", "user_screen_name"),
    ("location", "user_location"),
    ("from_user_followercount", "user_followers"),
]


def translate_row(row) -> tuple:
    """Use the same fields to uniform output.

    Parameters
    ----------
    row : dict
        a raw row from csv

    Returns
    -------
    tweet: dict or None
        the tweet data
    retweet: dict or None
        the retweet data
    user: dict
        the user data
    """
    if "time" in row:
        # old keys
        index = 0
    else:
        # new keys
        index = -1

    if row.get("retweeted_id", "") == "":
        # this is a tweet
        tweet = {k[index]: row[k[index]] for k in TWEET_FIELDS}
        retweet = None
    else:
        # this is a retweet
        tweet = None
        retweet = {k[index]: row[k[index]] for k in RETWEET_FIELDS}
    assert (tweet is None) ^ (retweet is None)

    user = {k[index]: row[k[index]] for k in USER_FIELDS}

    return tweet, retweet, user


def daterange(first, last, step=1):
    """Return a list of dates between first and last (excluded).

    Parameters
    ----------
    first : str or datetime.date
        fisrt date
    last : str or datetime.date
        last date (excluded)
    step : int
        step in days
        (Default value = 1)

    Returns
    -------
    days : list[datetime.date]
        a ordered list of days

    """
    if isinstance(first, str):
        first = date.fromisoformat(first)
    if isinstance(last, str):
        last = date.fromisoformat(last)
    return [first + timedelta(days=x) for x in range(0, (last - first).days, step)]


def filename(day, corpus="coronavirus"):
    """Return a filename.

    Parameters
    ----------
    day : date
        day

    corpus :
        (Default value = 'coronavirus')

    Returns
    -------
    filename : string
        a filename
    """
    return f"tweets-{corpus}_{day.isoformat()}.csv.gz"


def tonumber(string):
    """Convert to number if possible.

    Parameters
    ----------
    string :
        a string

    Returns
    -------
    number :
        a number
    """
    if string is None:
        return None

    if isinstance(string, list):
        print(string)

    try:
        val = int(string)
    except ValueError:
        try:
            val = float(string)
        except ValueError:
            val = str(string)
    except OverflowError:
        val = str(string)

    return val


def values2number(dictionary):
    """Convert values to number if possible.

    This is a shallow map (it won't act recursively)

    Parameters
    ----------
    dictionary : a dict
        a dictionary


    Returns
    -------
    intdict : dict
        same dictionary with numbers as numbers
    """
    return {k: tonumber(v) for k, v in dictionary.items() if v != ""}


def is_float(var):
    if isinstance(var, (list, np.ndarray)):
        return [is_float(x) for x in var]
    try:
        float(var)
    except ValueError:
        return False
    except TypeError:
        return False
    return True


def removesuffix(text, suffix):
    """

    Parameters
    ----------
    text :

    suffix :


    Returns
    -------


    """
    if text.endswith(suffix):
        return text[len(suffix) :]
    return text


def removeprefix(text, prefix):
    """

    Parameters
    ----------
    text :

    prefix :


    Returns
    -------


    """
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def entropy(iterable):
    """Return the entropy.

    Parameters
    ----------
    iterable : iterable
        a list of numbers

    Returns
    -------
    entropy : float
        the entropy of the normalized distribution
    """
    p = np.array([v for v in iterable if v > 0.0], dtype=np.float32)
    p /= p.sum()

    return -(p * np.log2(p)).sum()


def smooth_convolve(line, window=3):
    """Return a smoothed line computed with the hann kernel.

    Parameters
    ----------
    line : iterable
        values to be smoothed (1d array)
    window : int, default=3
        lenght of the sliding window

    Returns
    -------
    line : np.ndarray
        the smoothed line
    """
    assert window % 2 == 1, "Give me an odd window"

    half_window = window // 2
    extended_line = np.concatenate(
        (np.full((half_window,), line[0]), line, np.full((half_window,), line[-1]))
    )
    kernel = np.hanning(window)

    return np.convolve(extended_line, kernel, mode="valid") / np.sum(kernel)


def load_csv(filepath, transpose=False, headers=True):
    """Load from csv.

    Parameters
    ----------
    filepath : str or pathlib.Path
        the file path
    transpose : bool (default=False)
        if True the output will be dict of lists
    headers : bool (default=True)
        if the file has a header row

    Returns
    -------
    data : dict or list
        return the data in the csv file as list of dicts.
        If transpose is `True` the return data will be a dict of lists.
    """
    if isinstance(filepath, str):
        filepath = pathlib.Path(filepath)

    suffixes = filepath.suffixes
    # use gzip if needed
    if suffixes[-1] == ".gz":
        open_with = gzip.open
        suffixes = suffixes[:-1]
    else:
        open_with = open

    # use tsv if suffix
    if suffixes[-1] == ".tsv":
        dialect = csv.excel_tab
    else:
        dialect = csv.excel

    with open_with(filepath, "rt") as fin:
        if headers:
            reader = csv.DictReader(fin, dialect=dialect)
            fields = reader.fieldnames
            data = [values2number(row) for row in reader]
        else:
            reader = csv.reader(fin, dialect=dialect)
            data = [values2number({k: v for k, v in enumerate(row)}) for row in reader]
            fields = list(data[0].keys())

    if transpose:
        return {k: [d.get(k, None) for d in data] for k in fields}
    return data


def dump_csv(filepath, data):
    """Write to csv.

    Parameters
    ----------
    filepath : str or pathlib.Path
        the file path
    data : list
        an iterable over dictionaries
    """
    if isinstance(filepath, str):
        filepath = pathlib.Path(filepath)

    suffixes = filepath.suffixes
    # use gzip if needed
    if suffixes[-1] == ".gz":
        open_with = gzip.open
        suffixes = suffixes[:-1]
    else:
        open_with = open

    # use tsv if suffix
    if suffixes[-1] == ".tsv":
        dialect = csv.excel_tab
    else:
        dialect = csv.excel

    with open_with(filepath, "wt") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(data[0].keys()), dialect=dialect)
        writer.writeheader()
        writer.writerows(data)


def cluster_tseries(matrix, array=None, temp_sort=False):
    """Cluster temporal series.

    Parameters
    ----------
    matrix :
        A mantrix of NxT nodes by time steps
    array :
        Array to sort with the matrix
         (Default value = None)
    temp_sort :
        If True, sort communities based on time.
         (Default value = False)

    Returns
    -------

    """
    nn, nt = matrix.shape

    # compute covariance
    cov = np.cov(matrix)
    cov[cov < 0] = 0

    # build a graph from positive values
    graph = nx.from_numpy_array(cov)

    # partition with modularity
    part = best_partition(graph)
    parts = {}
    for n, p in part.items():
        parts.setdefault(p, []).append(n)

    # sort partitions depending on maximum values happening earlier or later
    maxes = matrix.argmax(axis=1)
    parts = {
        x[0]: x[1][1] for x in enumerate(sorted(parts.items(), key=lambda x: maxes[x[1]].mean()))
    }
    part = {node: p for p, nodes in parts.items() for node in nodes}

    # label nodes
    labels = [part[n] for n in range(nn)]
    # indx = np.argsort(labels, kind='stable')
    indx = [node for nodes in parts.values() for node in nodes]

    # sort labels by indx
    labels = [labels[i] for i in indx]
    # pick the first indexes of each class
    sep = [
        0,
    ] + [i + 1 for i, (l1, l2) in enumerate(zip(labels, labels[1:])) if l1 != l2]

    if array is None:
        return matrix[indx, :], sep
    return matrix[indx, :], np.asarray(array)[indx], sep


def cluster_first_axis(matrix, temp_sort=False):
    """Cluster temporal series.

    Parameters
    ----------
    matrix :
        A mantrix of NxT nodes by time steps
    array :
        Array to sort with the matrix
         (Default value = None)
    temp_sort :
        If True, sort communities based on time.
         (Default value = False)

    Returns
    -------
    partition : list
        list of lists of indices
    """
    nn, nt = matrix.shape

    # compute covariance
    cov = np.cov(matrix)
    cov[cov < 0] = 0

    # build a graph from positive values
    graph = nx.from_numpy_array(cov)

    # partition with modularity
    part = best_partition(graph)
    parts = {}
    for n, p in part.items():
        parts.setdefault(p, []).append(n)

    return [sorted(nodes) for nodes in parts.values()]


def get_keywords(topic: str, get_minimal: bool = False, min_len: int = 0):
    """Load filtering words.

    Get all possible keywords plus the minimum set of keywords contained in the full set.

    Parameters
    ----------
    topic : str
        the topic (one of `covid` or `vaccine`)
    get_minimal : bool
         (Default value = False)
         return long and short sets
         (long has all keyworkds, all keys in long include at least one key in short)
    min_len : int
         (Default value = 0)
         Minimum keyword lenght to consider.

    Returns
    -------
    topics : dict
        a dictionary with a list of self excluding tags (`short`) and all tags (`long`)
    """
    filepath = BASENAME / f"data/keys-{topic}.txt"
    with open(filepath, "rt") as fin:
        long = [line.lower().strip() for line in fin]
    # remove shorter keys
    long = set([k for k in long if len(k) >= min_len])

    if not get_minimal:
        return long

    short = set()
    for flt_long in long:
        to_remove = []
        for flt in short:
            if flt in flt_long:
                break

            if flt_long in flt:
                to_remove.append(flt)
        else:
            for flt2rm in to_remove:
                short.remove(flt2rm)
            short.add(flt_long)

    return {"long": long, "short": short}


def daily_tweets(day: date):
    """List all tweets from a given day.

    Warning: it has to be run in one go.

    Parameters
    ----------
    day : date
        the day of interest

    Yields
    ------
    row : dict
        a row of data (one tweet or retweet)
    """
    for fname in BASENAME.parent.parent.glob(day.strftime("corpus_*/**/*%Y-%m-%d.csv.gz")):
        with gzip.open(fname, "rt") as fin:
            # yield fin.readline()
            for row in csv.DictReader(
                [
                    line.replace("\0", " ").replace("\n", " ").replace("\r", " ").strip()
                    for line in fin
                ]
            ):
                # sanitize null chars
                yield dict(row)


def filter_row(row: list, keywords: dict, fmt="all", check_lang=True):
    """Return True if any of the keywords is in the row.

    Warning: only French tweets are used.

    Parameters
    ----------
    row : list
        list of rows
    keywords : dict
        dict of keywords
    fmt : str, (default = 'all')
        any of 'all' or 'any'
    check_lang : bool
        If to check lang.

    Returns
    -------
    contains : bool
        True is contains the keywords either in row['text'] or in row['hashtags']
    """
    if check_lang and row["lang"] != "fr":
        return False

    for topic, kwds in keywords.items():
        if fmt == "all" and not __keyword_in_row__(kwds["long"], kwds["short"], row):
            return False
        elif fmt == "any" and __keyword_in_row__(kwds["long"], kwds["short"], row):
            return True

    if fmt == "all":
        return True
    return False


def __keyword_in_row__(key_long, key_short, row):
    hts = set(row.get("hashtags", "").split("|"))
    if hts & key_long or __intext__(row["text"].lower(), key_short):
        return True
    return False


def __intext__(text, kwds):
    for word in kwds:
        if word in text:
            return True
    return False


def load_cases(country: str, day1="2020-01-01", day2="2021-10-01", transpose=False):
    with open(BASENAME / "data/WHO-COVID-19-global-data.csv", "rt") as fin:
        reader = csv.DictReader(fin, dialect=csv.excel)
        headers = list(reader.fieldnames)
        data = {
            x["Date_reported"]: values2number(x) | dict(day=date.fromisoformat(x["Date_reported"]))
            for x in reader
            if x["Country_code"] == country.upper()
        }
        headers.append("day")

    with open(BASENAME / "data/table-indicateurs-open-data-france.csv", "rt") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            day = date.fromisoformat(row["extract_date"])
            if day.isoformat() in data:
                data[day.isoformat()].update(values2number(row))

        headers += list(reader.fieldnames)

    data = [data[dt.isoformat()] for dt in daterange(day1, day2) if dt.isoformat() in data]

    if transpose:
        return {k: [d.get(k, None) for d in data] for k in headers}
    return data


def get_communities(tau=-1):
    """Return a dictionary of uids and communities."""
    cache = BASENAME / "data/partitions_tau_{tau}.tsv"

    if cache.is_file():
        print("loading_cache")
        part = load_csv(cache, transpose=False)
        return dict([u.values() for u in part])
    else:
        users = load_csv(f"../data/users-tau_{tau}.tsv.gz", transpose=False)
        part = [{"id": u["id"], "part": u["part"]} for u in users]
        dump_csv(cache, part)
        return dict([u.values() for u in part])
