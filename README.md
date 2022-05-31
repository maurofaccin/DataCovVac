# Vaccine-critical and news media URLs

In this repository we provide the manually classified list of URLs used in:

> **Assessing the influence of French vaccine critics during the two first years of the COVID-19 pandemic**
> *Faccin, Gargiulo, Atlani-Duault and Ward*
> [arxiv:2202.10952](http://arxiv.org/abs/2202.10952), (2022)

The cited article contains a detailed description of the database construction process.
If you use this in your project, please add a reference to the above article.

## Dataset

The dataset contains 382 French websites and blogs that contains vaccine-critical postures (some of those URLs may not be working anymore), and 383 French websites of news media.

The dataset is saved in `./data/urls.json` in the following format:

```json
{
  "vaccine-critical: [...]",
  "news-media": [...]
}
```
