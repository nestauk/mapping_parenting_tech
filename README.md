# :baby: Mapping parenting tech

**_Analysing data on children and parenting apps_**

## :wave: Welcome!

Nesta’s fairer start mission wants to close the outcome gap for disadvantaged children in the early years. In a collaboration between the A Fairer Start mission and Discovery Hub, we are exploring what role digital technologies could play in helping parents to enhance their children’s development at home. [Read more about the project and its findings.](https://www.nesta.org.uk/project/mapping-parenting-technology/)

In this repo, you will find our code for collecting and analysing children and parenting app data, as well as links to the collected datasets.

_NB: This codebase is at a prototyping stage, where some of the code is still living in Jupyter notebooks whereas some utilities have already been neatly factored out into modules. Do [contact us](mailto:karlis.kanders@nesta.org.uk) if something appears confusing and we'll be happy to help._

## :hammer_and_wrench: Code

Once you've installed the requirements, also install the repo itself as a package by running the following command in the terminal:

```shell
pip install -e .
```

We've focused on Google Play Store and relied upon [google-play-scraper](https://pypi.org/project/google-play-scraper/) package. Around this package, we've written a module `mapping_parenting_tech.utils.play_store_utils` with useful helper functions that allow easy fetching and housekeeping of the app and apps' review data.

An example of using this module to fetch information about Play Store apps can be found in `mapping_parenting_tech/analysis/prototyping/play_store/get_play_store_apps.py`

Code for clustering the apps into data-driven categories, performing descriptive analysis, and carrying out topic modelling of app reviews can be found in the notebooks in the `mapping_parenting_tech/analysis/prototyping` folder.

We will update this section with more details and instructions as we further refactor the code.

## :floppy_disk: Datasets

The collected datasets can be found here:

- [play_store.zip](https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data/play_store.zip): HTML files fetched from Play Store website (used to extract initial set of app identifiers), and a table with apps and their categories that were rated relevant by manual inspection
- [all_app_details.json](https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data/all_app_details.json): Data collected using google-play-scraper about all apps considered in this study
- [app_reviews.zip](https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data/app_reviews.zip): App reviews for all apps that were rated relevant by manual inspection

## :handshake: Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
