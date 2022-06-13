# Vaccine-critical and news media URLs

In this repository we provide dataset used and produced in:

> **Assessing the influence of French vaccine critics during the two first years of the COVID-19 pandemic**
> *Faccin, Gargiulo, Atlani-Duault and Ward*
> [arxiv:2202.10952](http://arxiv.org/abs/2202.10952), (2022)

The cited article contains a detailed description of the datasets' construction processes.
If you use this in your project, please add a reference to the above article.

## Datasets

### Tweet IDs

The file `data/tids.txt.gz` contains the list of tweets and retweets IDs used in the above publication.

### Vaccine-critical and News media URLs

The dataset contains 382 French websites and blogs that contains vaccine-critical postures (some of those URLs may not be working anymore), and 383 French websites of news media.

The dataset is saved in `./data/urls.json` in the following format:

```json
{
  "vaccine-critical: [...]",
  "news-media": [...]
}
```

### Twitter APIs keywords

This dataset includes all the keywords used to extract tweets from Twitter using its streaming and search APIs.
The dataset `./data/keywords.json` is divided into three sets that corresponds to the three tweet datasets (DataVac, DataCov, DataHC) mentioned in the paper above.

The dataset format is as follows:

```json
{
    "DataVac": [...],
    "DataCov": [...],
    "DataHC": [...]
}
```

## Codes

In the `code` folder we provide a number of python scripts that extract the data needed to plot.
The code requires that the tweets (once hydrated) will be separated in daily chunks of:

```
YYYY-MM-DD-tweets.csv.gz
YYYY-MM-DD-retweets.csv.gz
YYYY-MM-DD-users.csv.gz
```

Users may be repeated on different days.

### Plotting

In the `plots` folder one can find the python script used to reproduce the plots of the above paper.
