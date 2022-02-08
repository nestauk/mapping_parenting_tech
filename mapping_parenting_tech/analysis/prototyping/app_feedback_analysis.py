# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3812jvsc74a57bd09d0629e00499ccf218c6720a848e8111287e8cbf09d1f93118d5865a19869c30
# ---

# %% [markdown]
# Code to analyse app ratings and reviews

# %%
from mapping_parenting_tech import PROJECT_DIR, logging

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
REVIEWS_DIR = DATA_DIR / "app_reviews"
INPUT_DIR = PROJECT_DIR / "inputs/data/play_store"
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
def load_all_app_reviews() -> pd.DataFrame():
    """
    Loads all app reviews into a Pandas dataframe.

    This may take some time for 1,000s of apps (e.g., ~2.5minutes for reviews of 3,500 apps)

    Returns:
        Pandas dataframe containing all reviews for all apps - note, this can be quite large!
    """

    all_reviews = pd.DataFrame()
    review_files = REVIEWS_DIR.iterdir()
    reviews_df_list = list()

    logging.info("Loading files")
    for file in tqdm(review_files):
        reviews_df = pd.read_csv(file, header=0, index_col=None)
        reviews_df_list.append(reviews_df)

    logging.info("Concatenating data")
    all_reviews = pd.concat(reviews_df_list, axis=0, ignore_index=True)

    return all_reviews


# %%
app_reviews = load_all_app_reviews()

# %%
app_details = pd.read_json(DATA_DIR / "all_app_details.json", orient="index")
app_details.reset_index(inplace=True)
app_details.rename(columns={"index": "appId"}, inplace=True)

# %%
focus_apps_list = pd.read_csv(
    INPUT_DIR / "relevant_app_ids.csv", header=0, names=["appId"]
)
focus_apps_list = focus_apps_list["appId"].tolist()

# %%
focus_apps_details = app_details[app_details["appId"].isin(focus_apps_list)]
focus_app_reviews = app_reviews[app_reviews["appId"].isin(focus_apps_list)]

# %%
# delete the very large `app_reviews` dataframe
del app_reviews

# %%
focus_app_reviews["rYear"] = pd.to_datetime(focus_app_reviews["at"]).dt.year
focus_app_reviews["rMonth"] = pd.to_datetime(focus_app_reviews["at"]).dt.month

# %%
# Create a new dataframe, `reviewDates` with the number of reviews for each app per year
review_dates = focus_app_reviews.groupby(["appId", "rYear"])["appId"].count().unstack()
total_reviews_by_app = focus_app_reviews.groupby(["appId"])["appId"].count()
review_dates["total_reviews"] = total_reviews_by_app
review_dates


# %%
def growth_2021(row):
    if (row[2021] > 0) and (row[2020] > 0):
        return row[2021] / row[2020]
    elif row[2021] > 0:
        return 1
    else:
        return 0


review_dates["2021_growth"] = review_dates.apply(growth_2021, axis=1)
review_dates.sample(5)

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
    focus_apps_details[["appId", "genre", "score", "minInstalls"]], on="appId"
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
plot_df = df[df["score"] >= 4]

# %%
xscale = alt.Scale(type="log", base=10)
yscale = alt.Scale(domain=[0, 30])
fig = (
    alt.Chart(plot_df, width=650, height=650)
    .mark_circle(filled=True, size=50)
    .encode(
        alt.X("total_reviews", scale=xscale, title="Given* number of reviews"),
        alt.Y(
            "growth",
            scale=yscale,
            title="Growth of reviews (#reviews in 2021 / #reviews in app's first year",
        ),
        alt.Color("genre:N"),
        # alt.Size("score:N"),
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
