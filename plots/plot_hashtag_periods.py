#!/usr/bin/env python
"""Most important hashtags per period."""


import json
from collections import Counter
from datetime import date

import numpy as np
import squarify
from matplotlib import colors, pyplot
from matplotlib.transforms import Bbox

import cov
import cov_text
import cov_utils

pyplot.style.use("mplrc")
grey = "#999999"


class Hashtags:
    """Collecting data.

    Collect count of hashtags per day in __data__[hashtag][day]
    Collect best 3 hashtags of each day in __best_hashtags__
    Collect all Hashtags in __hashtags__
    Tot number of hashtag per day in __day_tot__
    """

    def __init__(self):
        self.__days__ = []
        self.__hashtags__ = set()
        self.__best_hashtags__ = set()

        self.__data__ = {}
        self.__day_tot__ = {}

    def update(self, day, hashtags):
        """Update, adding new hashtags."""
        try:
            iday = self.__days__.index(day)
        except ValueError:
            iday = None

        self.__hashtags__ |= set(hashtags.keys())
        if iday is None:
            self.__days__.append(day)

        for hashtag, count in hashtags.items():
            self.__data__.setdefault(day, Counter())
            self.__data__[day][hashtag] += count

        self.__day_tot__.setdefault(day, 0)
        self.__day_tot__[day] += sum(hashtags.values())

    def timeseries(self, use=None):
        """Return the timeseries of each hashtag.

        Parameters
        ----------
            use: list or None
                list of hashtags to use (if None use the best 3 hashtags per day).
        """
        if use is None:
            use = self.__best_hashtags__
            use = {
                k for dhash in self.__data__.values() for k, v in dhash.most_common(3)
            }

        data = np.zeros((len(use), len(self.__days__)))

        bht = {ht: i for i, ht in enumerate(use)}

        for iday, day in enumerate(self.__days__):
            for ht, iht in bht.items():
                data[iht, iday] += self.__data__[day].get(ht, 0)

        # tots = [val for day, val in sorted(self.__day_tot__.items())]
        tots = [self.__day_tot__[day] for day in self.__days__]
        return list(bht.keys()), self.__days__, data, np.array(tots)


def get_daily_hashtags(day: date, keywords=set()):
    """Extract all daily hashtags with count.

    Parameters
    ----------
    day : date
        day of interest
    keywords :
         (Default value = set())
         keywords to remove

    Returns
    -------
    Hashtags : dict
        map of day: Counter of hashtags
    """
    daily = cov.load_day(day)

    hashtags = {day.isoformat(): Counter()}
    for tid, tweet, rtws in daily.tweets.iter():
        for hashtag in tweet.get("hashtags", "").split("|"):
            if hashtag == "":
                continue
            hashtags[day.isoformat()][hashtag] += 1
            for rtw in rtws:
                rtw_day = rtw["created_at"][:10]
                hashtags.setdefault(rtw_day, Counter())
                hashtags[rtw_day][hashtag] += 1

    if len(keywords) > 0:
        for kwd in keywords | {"covidー19", "covid19fr", "covid19france", ""}:
            for hash in hashtags.values():
                hash.pop(kwd, None)
    return hashtags


def filter_entropy(data, hashtags, min_count_rate=None):
    nht, ntime = data.shape
    thresh = np.log2(ntime) * 0.8

    filtered = [i for i in range(nht) if cov_utils.entropy(data[i, :]) < thresh]

    if min_count_rate is not None:
        filtered = [i for i in filtered if data[i, :].sum() > min_count_rate * ntime]
    return data[filtered, :], [hashtags[ht] for ht in filtered]


def sort_part(matrix: np.array, hashtags: list):

    _matrix = matrix.copy()
    for i in range(_matrix.shape[0]):
        nsum = _matrix[i, :].sum()
        _matrix[i, :] = cov_utils.smooth_convolve(_matrix[i, :], window=7)
        _matrix[i, :] *= nsum / _matrix[i, :].sum()

    parts = cov_utils.cluster_first_axis(_matrix)

    maxes = _matrix.argmax(axis=1)
    parts = [p for p in sorted(parts, key=lambda x: np.mean([maxes[i] for i in x]))]

    sep = []
    for p in parts:
        sep.append(len(p))

    indices = [i for nodes in parts for i in nodes]

    return indices, np.cumsum(sep)


def auto_fit_fontsize(text, width, height, fig=None, ax=None):
    """Auto-decrease the fontsize of a text object.

    Args:
        text (matplotlib.text.Text)
        width (float): allowed width in data coordinates
        height (float): allowed height in data coordinates
    """
    fig = fig or pyplot.gcf()
    ax = ax or pyplot.gca()

    # get text bounding box in figure coordinates
    renderer = fig.canvas.get_renderer()
    bbox_text = text.get_window_extent(renderer=renderer)

    # transform bounding box to data coordinates
    bbox_text = Bbox(ax.transData.inverted().transform(bbox_text))

    # evaluate fit and recursively decrease fontsize until text fits
    fits_width = bbox_text.width < width if width else True
    fits_height = bbox_text.height < height if height else True
    if not all((fits_width, fits_height)) and text.get_fontsize() >= 5:
        text.set_fontsize(text.get_fontsize() - 1)
        auto_fit_fontsize(text, width, height, fig, ax)


def add_text(labels: list, squares: list, ax: pyplot.Axes, fig: pyplot.Figure):
    for text, rect in zip(labels, squares):
        ratio = rect["dx"] / rect["dy"]
        if ratio > 1.5:
            angle = 0
        elif ratio < 0.75:
            angle = 90
        else:
            angle = 45
        axtext = ax.text(
            rect["x"] + 0.5 * rect["dx"],
            rect["y"] + 0.5 * rect["dy"],
            text,
            fontsize=24,
            ha="center",
            va="center",
            rotation=angle,
        )

        auto_fit_fontsize(axtext, rect["dx"], rect["dy"], fig=fig, ax=ax)


def load_data():
    dt_ranges = [
        list(cov_utils.daterange("2020-01-01", "2020-11-11")),
        list(cov_utils.daterange("2020-11-11", "2021-06-01")),
        list(cov_utils.daterange("2021-06-01", "2021-10-01")),
    ]
    dt_map = {
        dt.isoformat(): indx for indx, dtrange in enumerate(dt_ranges) for dt in dtrange
    }

    cache = cov_utils.BASENAME / "data/hashtags_period_count.json"
    if cache.is_file():
        with open(cache, "rt") as fin:
            hts = json.load(fin)

        return [Counter(x) for x in hts]

    hts = [Counter() for _ in dt_ranges]
    keywords = cov_utils.get_keywords("covid") | cov_utils.get_keywords("vaccine")

    for dtrange in dt_ranges:
        for day in dtrange:
            print(day, end="\r")
            hashtags = get_daily_hashtags(day, keywords)
            for day, hashs in hashtags.items():
                try:
                    indx = dt_map[day]
                except KeyError:
                    continue

                hts[indx] += hashs

    with open(cov_utils.BASENAME / "data/hashtags_period_count.json", "wt") as fout:
        json.dump([dict(x) for x in hts], fout)
    return hts


def more_than_expected(corpus: list):
    hts = set([ht for doc in corpus for ht in doc])
    hts = {ht: indx for indx, ht in enumerate(hts)}

    def csum(c):
        return np.sum(list(c.values()))

    cum = Counter()
    for doc in corpus:
        cum += doc

    hts = [
        sorted(
            list(doc.keys()),
            # key=lambda x: doc[x] * csum(cum) / cum[x] / csum(doc),
            key=lambda x: doc[x] / cum[x],
            reverse=True,
        )[:20]
        for doc in corpus
    ]
    return hts


def main():
    """Do the main."""
    hts = load_data()

    for ht in hts:
        print(ht.most_common(5))

    # out = cov_text.tfidf(
    #     [" ".join(ht.elements()) for ht in hts],
    #     stop_words=["../data/stopwords-en.json", "../data/stopwords-fr.json"],
    # )
    # best_words = cov_text.tfidf_best_words(*out, nwords=20)
    best_words = [[x[0] for x in ht.most_common(20)] for ht in hts]
    # best_words = more_than_expected(hts)
    print(best_words)

    fig, axes = pyplot.subplots(
        ncols=3,
        figsize=(12, 4),
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,
            "top": 0.9,
            "right": 0.98,
            "left": 0.02,
            "bottom": 0.02,
        },
    )

    for hashtags, words, ax in zip(hts, best_words, axes):
        sizes = [hashtags[w] for w in words]
        rects = squarify.squarify(
            squarify.normalize_sizes(sizes, 100, 100), 0, 0, 100, 100
        )
        squarify.plot(sizes, ax=ax, norm_y=100, norm_x=100, pad=False, alpha=0.5)
        add_text(words, rects, ax, fig)
        ax.axis("off")

    axes[0].set_title("First Period")
    axes[1].set_title("Second Period")
    axes[2].set_title("Third Period")

    pyplot.savefig("plot_hashtag_periods.pdf")


if __name__ == "__main__":
    main()
