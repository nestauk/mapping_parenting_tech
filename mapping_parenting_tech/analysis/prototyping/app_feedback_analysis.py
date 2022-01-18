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
#     display_name: 'Python 3.8.0 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# Code to analyse app ratings and reviews

# %%
from mapping_parenting_tech import PROJECT_DIR

# from mapping_parenting_tech.utils.altair_save_utils import (
#    google_chrome_driver_setup,
#    save_altair,
# )
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from pathlib import Path
import json
import math
from tqdm import tqdm

DATA_DIR = PROJECT_DIR / "outputs/data"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# %%
apps_of_interest = [
    "com.easypeasyapp.epappns",
    "com.lingumi.lingumiplay",
    "com.learnandgo.kaligo",
    "com.kamicoach",
    "com.storytoys.myveryhungrycaterpillar.free.android.googleplay",
    "uk.co.bbc.cbeebiesgoexplore",
    "tv.alphablocks.numberblocksworld",
    "com.acamar.bingwatchplaylearn",
    "uk.org.bestbeginnings.babybuddymobile",
    "com.mumsnet.talk",
    "com.mushuk.mushapp",
    "com.teampeanut.peanut",
    "com.flipsidegroup.nightfeed.paid",
]

# %%
load_reviews_list = [
    "education_apps_reviews.csv",
    "parenting_apps_reviews.csv",
    "related_to_easypeasy_reviews.csv",
    "kids_under_five_reviews.csv",
]

getter = []
for r_file in tqdm(load_reviews_list):
    print(f"Loading {r_file}")
    load_df = pd.read_csv(DATA_DIR / r_file, index_col=None, header=0)
    getter.append(load_df)
    print(f"Loaded {len(load_df)} reviews.")

print("Consolidating dataframe")
app_reviews = pd.concat(getter, axis=0, ignore_index=True)

print(f"Dropping duplicates (currently {len(app_reviews)} reviews")
app_reviews.drop_duplicates(inplace=True)
print(f"FINISHED. Loaded {len(app_reviews)} unique reviews.")
# apps = app_reviews["appId"].unique().tolist()

# %%
focus_apps = pd.read_csv(DATA_DIR / "focus_apps_22-01-11.csv")
focus_apps.drop("appId", axis=1, inplace=True)
focus_apps.rename(columns={focus_apps.columns[0]: "appId"}, inplace=True)
focus_apps_list = focus_apps["appId"].unique().tolist()
focus_app_reviews = app_reviews[app_reviews["appId"].isin(focus_apps_list)]

# %%
focus_app_reviews["rYear"] = pd.to_datetime(focus_app_reviews["at"]).dt.year
focus_app_reviews["rMonth"] = pd.to_datetime(focus_app_reviews["at"]).dt.month

# %%
review_dates = pd.DataFrame(columns=["appId", "total_reviews"])
min_date = datetime.now().year
for app_id in focus_apps_list:
    date_dict = dict()
    start_year = datetime.now().year
    total_reviews = 0
    this_app_dates = (
        focus_app_reviews[focus_app_reviews["appId"] == app_id]
        .groupby(by=["rYear"])["reviewId"]
        .count()
    )
    for i, v in this_app_dates.items():
        date_dict.update({i: v})
        start_year = i if (i < start_year) and (v > 0) else start_year
        total_reviews += v
    min_date = start_year if start_year < min_date else min_date
    date_dict.update({"appId": app_id, "total_reviews": total_reviews})
    growth_dict = {
        f"{i}_growth": v / date_dict[start_year] for i, v in this_app_dates.items()
    }
    date_dict.update(growth_dict)
    review_dates = review_dates.append(date_dict, ignore_index=True)

# %%
col_order = ["appId", "total_reviews"]
col_order.extend([i for i in range(min_date, datetime.now().year + 1)])
col_order.extend(f"{i}_growth" for i in range(min_date, datetime.now().year + 1))

review_dates = review_dates[col_order]
review_dates.sort_values(by="total_reviews", ascending=False, inplace=True)

dropped_apps = review_dates[review_dates["total_reviews"] == 0]
review_dates = review_dates[review_dates["total_reviews"] > 0]

print(
    f"Dropped {len(dropped_apps)} apps with zero reviews, leaving {len(review_dates)}"
)

# %%
review_dates = review_dates.merge(
    focus_apps[["appId", "genre", "score", "minInstalls"]], on="appId"
)

# %%
df = pd.DataFrame(
    columns=["appId", "total_reviews", "growth", "genre", "minInstalls", "score"],
    data=review_dates[
        ["appId", "total_reviews", "2021_growth", "genre", "minInstalls", "score"]
    ].values,
    index=review_dates.index,
)
df

# %%
xscale = alt.Scale(type="log", base=10)
yscale = alt.Scale(type="log", base=10)
fig = (
    alt.Chart(df, width=750, height=750)
    .mark_circle(filled=True, size=50)
    .encode(
        alt.X("total_reviews", scale=xscale, title="Given* number of reviews"),
        alt.Y(
            "growth",
            scale=yscale,
            title="Growth of reviews (#reviews in 2021 / #reviews in app's first year",
        ),
        alt.Color("genre:N"),
        tooltip=["appId", "genre", "minInstalls"],
    )
).interactive()

fig

# %%
xscale = alt.Scale(base=10)
yscale = alt.Scale(type="log", base=10)
fig2 = (
    alt.Chart(df, width=750, height=750)
    .mark_circle(filled=True, size=50)
    .encode(
        alt.X("score", scale=xscale, title="App score"),
        alt.Y(
            "growth",
            scale=yscale,
            title="Growth of reviews (#reviews in 2021 / #reviews in app's first year",
        ),
        alt.Color("genre:N"),
        # alt.Size("growth"),
        tooltip=["appId", "genre", "minInstalls"],
    )
).interactive()

fig2


# %%
def check_app(needles: list(), haystack: str) -> dict:

    app_id_list = pd.read_csv(DATA_DIR / haystack)
    pd_needles = pd.Series(needles)

    return pd_needles.isin(app_id_list[list(app_id_list.columns)[0]])


check_app(["com.fiftythings.bradford"], "related_to_fiftythings_ids.csv")


# %%
def load_description_set(filename: str, file_path: Path = DATA_DIR) -> pd.DataFrame:
    # Load in the descriptions

    with open(file_path / filename, "rt") as all_data_handle:
        all_data = json.load(all_data_handle)

    all_data = {k: v for d in all_data for k, v in d.items()}
    return pd.DataFrame(all_data).T


# %%
def load_descriptions(file_list: list) -> pd.DataFrame:

    list_pd = [load_description_set(f) for f in file_list]
    return_df = pd.concat(list_pd, axis=0, ignore_index=True)
    return_df.drop_duplicates(subset=["appId"], inplace=True)
    return return_df


# %%
f_list = [
    "related_to_easypeasy_details.json",
    "education_apps_details.json",
    "kids_under_five_details.json",
    "parenting_apps_details.json",
]

app_details = load_descriptions(f_list)

# %%
app_details["ratings"].replace([None], [0], inplace=True)
app_details["compound_score"] = (
    app_details["minInstalls"] * app_details["ratings"] * app_details["reviews"]
) + 0.1
app_details["log_score"] = app_details["compound_score"].apply(math.log10)
app_details.sort_values("compound_score", ascending=False, inplace=True)

# %%
xscale = alt.Scale(type="log", base=10)
yscale = alt.Scale(base=10)
fig = (
    alt.Chart(app_details, width=500, height=500)
    .mark_circle(size=60)
    .encode(
        alt.X("compound_score", scale=xscale),
        alt.Y("score", scale=yscale),
        alt.Color("genre:N"),
        tooltip=["appId", "genre"],
    )
).interactive()

fig

# %%
xscale = alt.Scale(base=10, zero=False)
yscale = alt.Scale(base=10)
fig = (
    alt.Chart(app_details, width=500, height=500)
    .mark_point(filled=True)
    .encode(
        alt.X("reviews", scale=xscale),
        alt.Y("score", scale=yscale),
        alt.Size("minInstalls"),
        alt.Color("genre:N"),
        tooltip=["appId"],
    )
).interactive()

fig

# %%
