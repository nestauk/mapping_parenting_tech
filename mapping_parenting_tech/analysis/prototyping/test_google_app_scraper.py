# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from google_play_scraper import app

app_id = "com.teachyourmonstertoread.tmapp"
app_id = "com.phonicshero.phonicshero"
# app_id = "com.learnandgo.kaligo.homemena"
# app_id = "com.fishinabottle.navigo"
# app_id = ""÷

result = app(
    # 'com.nianticlabs.pokemongo',
    app_id,
    country="uk",
    lang="en",  # defaults to 'en'
)

result_us = app(
    # 'com.nianticlabs.pokemongo',
    app_id,
    lang="en",  # defaults to 'en'
    country="us",  # defaults to 'us'
)

result_lv = app(
    # 'com.nianticlabs.pokemongo',
    app_id,
    lang="en",  # defaults to 'en'
    country="lv",  # defaults to 'us'
)
# result

# %%
list(result.keys())

# %%
result

# %%
field = "installs"
result[field], result_us[field], result_lv[field]

# %%
field = "score"
result[field], result_us[field], result_lv[field]

# %%
from google_play_scraper import Sort, reviews

result, _ = reviews(
    app_id,
    lang="en",  # defaults to 'en'
    country="uk",  # defaults to 'us'
    sort=Sort.NEWEST,  # defaults to Sort.MOST_RELEVANT
    count=100,  # defaults to 100
    filter_score_with=None,  # defaults to None(means all score)
)

result_us, _ = reviews(
    app_id,
    lang="en",  # defaults to 'en'
    country="us",  # defaults to 'us'
    sort=Sort.NEWEST,  # defaults to Sort.MOST_RELEVANT
    count=100,  # defaults to 100
    filter_score_with=None,  # defaults to None(means all score)
)

result_lv, _ = reviews(
    app_id,
    lang="en",  # defaults to 'en'
    country="lv",  # defaults to 'us'
    sort=Sort.NEWEST,  # defaults to Sort.MOST_RELEVANT
    count=100,  # defaults to 100
    filter_score_with=None,  # defaults to None(means all score)
)
