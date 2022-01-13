# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simple analyses of apps

# %%
from mapping_parenting_tech import PROJECT_DIR
from google_play_scraper import Sort, app, reviews_all, reviews
import json
import pandas as pd
import pickle

# %% [markdown]
# Load app details from saved JSON into `all_app_details`. Format is `{app_id: {app_details}}`. `app_details` is a further JSON structure, currently a replication of the Google PLay Scraper format, described here https://pypi.org/project/google-play-scraper/

# %%
app_details_file = PROJECT_DIR / "outputs/data/play_store_details.json"

with open(app_details_file, "r") as input_file:
    all_app_details = json.load(input_file)
input_file.close()

# %%
app_reviews_file = PROJECT_DIR / "outputs/data/play_store_reviews-21-12-15.json"

with open(app_reviews_file, "r") as input_file:
    all_app_reviews = json.load(input_file)
input_file.close()

# %% [markdown]
# ## Analyse apps
# Look at:
# - number of ratings
# - average rating / rating breakdown
# - number of reviews
# - rate of reviews (#reviews per day/days between reviews?)

# %%
all_app_info = list()
app_info = dict()

for app_id, app_details in all_app_details.items():
    app_info = {
        "app_id": app_id,
        "title": app_details["title"],
        "description": app_details["description"],
        "summary": app_details["summary"],
        "rating_count": app_details["ratings"],
        "mean_rating": app_details["score"],
        "claimed_reviews": app_details["reviews"],
        "installs": app_details["minInstalls"],
    }
    all_app_info.append(app_info)

app_details_df = pd.DataFrame(all_app_info)
app_details_df.set_index("app_id", inplace=True)

app_details_df.to_csv(PROJECT_DIR / "outputs/data/simple_app_descriptions.csv")


# %%
app_details_df = pd.read_csv(PROJECT_DIR / "outputs/data/simple_app_descriptions.csv")

# %%
app_row = 12
random_app = app_details_df.sample()
app_details_df.loc[app_row, "app_id"]

# %%
testapp = app(app_id=app_details_df.loc[app_row, "app_id"])

# %%
print(app_details_df.loc[app_row, "app_id"])
print(testapp["appId"])

print("Number of claimed reviews =", testapp["reviews"])
print("Number of retrieved reviews =", len(all_reviews))

# %% [markdown]
# ## Parse reviews into a single flat list

# %%
all_parsed_reviews = list()
parsed_review = dict()

for app_id, this_app_reviews in all_app_reviews.items():
    for app_review in this_app_reviews:
        parsed_review = {
            "app_id": app_id,
            "review": app_review["content"],
            "date-time": app_review["at"],
            "plus_one": app_review["thumbsUpCount"],
        }
        all_parsed_reviews.append(parsed_review)

app_reviews_df = pd.DataFrame(all_parsed_reviews)
app_reviews_df.set_index("app_id", inplace=True)

app_reviews_df.to_csv(PROJECT_DIR / "outputs/data/simple_reviews.csv")

# %%
app_reviews_df.index.value_counts()[0:10]
