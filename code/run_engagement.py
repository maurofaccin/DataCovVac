#!/usr/bin/env pyhton
"""Compute users engagement as an SIS model."""

from datetime import date, timedelta

import cov
import cov_utils as utils
from tqdm import tqdm

# classes


class Engaged:
    """Collect user engagement."""

    def __init__(self, window=3):
        self.__data__ = []
        self.__cache__ = {
            "uids": {"eng": set(), "tot": set(), "media": set()},
            "days": {},
        }

        self.__window__ = window
        self.__urls__ = cov.Urls()

    def add(self, data: cov.Data, day: date):
        """Import data from day.

        Parameters
        ----------
        data : cov.Data
            daily dataset
        day : date
            day
        """
        isoday = day.isoformat()
        self.__cache__["days"][isoday] = {
            "engaged": set(),
            "engaged_media": set(),
            "tot_users": set(),
        }

        for tid, tweet in data.tweets.items():
            # OP ID
            uid = tweet["from_user_id"]
            # Retweeters IDs
            rtw_uids = {rt["from_user_id"] for rt in data.tweets.retweets(tweet["id"])}

            self.__cache__["days"][isoday]["tot_users"].add(uid)
            self.__cache__["days"][isoday]["tot_users"] |= rtw_uids

            if "links" not in tweet:
                continue

            codes = self.__urls__.is_coded(tweet)
            if "PRE" in codes or "POST" in codes:
                self.__cache__["days"][isoday]["engaged"].add(uid)
                self.__cache__["days"][isoday]["engaged"] |= rtw_uids
            if "MEDIA" in codes:
                self.__cache__["days"][isoday]["engaged_media"].add(uid)
                self.__cache__["days"][isoday]["engaged_media"] |= rtw_uids

        self.__update__(day)

    def __update__(self, day: date):
        """Update and compute paramenters."""
        isoday = day.isoformat()

        days = utils.daterange(day - timedelta(days=self.__window__ - 1), day + timedelta(days=1))

        eng = self.__get_engaged__(days)
        tot = self.__get_engaged__(days, kind="tot_users")
        media = self.__get_engaged__(days, kind="engaged_media")

        # people becoming engaged
        new_eng = len(eng - self.__cache__["uids"]["eng"])
        new_media = len(media - self.__cache__["uids"]["media"])
        # people becoming disengaged
        dis_eng = len(self.__cache__["uids"]["eng"] - eng)
        dis_media = len(self.__cache__["uids"]["eng"] - media)

        alpha, beta = sis_params(
            [new_eng, dis_eng], len(eng), len(tot | self.__cache__["uids"]["tot"])
        )

        data = {
            "eng": len(eng),
            "tot": len(tot),
            "new_eng": new_eng,
            "dis_eng": dis_eng,
            "alpha": alpha,
            "beta": beta,
            "rt": alpha / beta if beta > 0 else 0.0,
            "day": isoday,
        }

        alpha, beta = sis_params(
            [new_media, dis_media], len(media), len(tot | self.__cache__["uids"]["tot"])
        )

        data |= {
            "media": len(media),
            "new_media": new_media,
            "dis_media": dis_media,
            "alpha_media": alpha,
            "beta_media": beta,
            "rt_media": alpha / beta if beta > 0 else 0.0,
        }

        self.__data__.append(data)
        self.__cache__["uids"]["eng"] = eng
        self.__cache__["uids"]["media"] = media
        self.__cache__["uids"]["tot"] = tot

    def __get_engaged__(self, days, kind="engaged"):
        eng = set()
        for day in days:
            if day.isoformat() in self.__cache__["days"]:
                eng.update(self.__cache__["days"][day.isoformat()][kind])
        return eng

    def data(self):
        """Return all data.

        Yields
        ------
        data : dicts
            all data as dicts
        """
        yield from self.__data__


def sis_params(di, i, n):
    """Compute alpha and beta for SIS.

    Returns
    -------
    alpha : float
        infection rate
    beta : float
        recovery rate
    """
    # compute SI/N
    sin = (n - i) * i / n

    if i == 0 or i == n:
        alpha = 0
    else:
        alpha = di[0] / sin

    if i == 0:
        beta = 0
    else:
        beta = max(di[1], 1) / i

    return alpha, beta


def main(window):
    """Do the main."""
    print("Window", window)
    engaged = Engaged(window=window)

    for day in tqdm(utils.daterange("2020-01-01", "2021-10-01")):
        daily = cov.load_day(day)
        engaged.add(daily, day)

    data = list(engaged.data())
    utils.dump_csv(f"data/engagement-tau_{window}.tsv", data)


if __name__ == "__main__":
    for win in [3, 5, 7]:
        main(win)
