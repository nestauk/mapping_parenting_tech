# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Testing playstore API
# https://pypi.org/project/google-play-scraper/

# %%
from google_play_scraper import app, Sort, reviews_all

result = app(
    "com.easypeasyapp.epappns",
    lang="en",  # defaults to 'en'
    country="us",  # defaults to 'us'
)

# %%
print(result)

# %%
targets = ["com.mushuk.mushapp", "com.easypeasyapp.epappns", "com.hp.pregnancy.lite"]

# %%
results = []
for target in targets:
    results.append(app(f"{target}", lang="en", country="us"))

print(len(results))

# %%
reviews = []
for target in targets:
    reviews.append(
        reviews_all(
            f"{target}",
            sleep_milliseconds=0,  # defaults to 0
            lang="en",  # defaults to 'en'
            country="us",  # defaults to 'us'
            sort=Sort.MOST_RELEVANT,  # defaults to Sort.MOST_RELEVANT
            filter_score_with=1,  # defaults to None(means all score)
        )
    )

for review in reviews:
    print(len(review))
# %%
import pandas as pd

review_df = pd.DataFrame(reviews[2])


# %%
review_df.head(5)
