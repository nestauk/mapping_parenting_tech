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
#     name: python3
# ---

# %% [markdown]
# # Analyse app ratings and reviews

# %%
from mapping_parenting_tech import PROJECT_DIR, logging
from mapping_parenting_tech.utils import play_store_utils as psu

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
focus_apps = pd.read_csv(INPUT_DIR / "relevant_app_ids.csv", header=0)
focus_apps_list = focus_apps["appId"].tolist()

# %%
app_reviews = psu.load_some_app_reviews(focus_apps_list)

# %%
# load app details
app_details = pd.DataFrame(psu.load_all_app_details()).T
app_details.reset_index(inplace=True)
app_details.rename(columns={"index": "appId"}, inplace=True)
app_details = app_details[app_details["appId"].isin(focus_apps_list)]
app_details = app_details.merge(focus_apps, on="appId")

# %%
app_reviews["rYear"] = pd.to_datetime(app_reviews["at"]).dt.year
app_reviews["rMonth"] = pd.to_datetime(app_reviews["at"]).dt.month

# %% [markdown]
# ## Look at reviews by date
# Frequency of reviews etc.

# %%
app_details.shape

# %%
# C reate a new dataframe, `reviewDates`, with the number of reviews for each app per year
review_dates = (
    app_reviews.groupby(["appId", "rYear"])["appId"].count().unstack().reset_index()
)
app_total_reviews = app_reviews.groupby(["appId"])["appId"].count()
review_dates["total_reviews"] = review_dates["appId"].map(app_total_reviews)
review_dates = review_dates.merge(focus_apps, on=["appId"])
review_dates.shape


# %%
def growth_2021(row):
    if (row[2021] > 0) and (row[2020] > 0):
        return row[2021] / row[2020]
    elif row[2021] > 0:
        return 1
    else:
        return 0


review_dates["2021_growth"] = review_dates.apply(growth_2021, axis=1)

# %%
review_dates = review_dates.merge(
    app_details[["appId", "genre", "score", "minInstalls"]], on="appId"
)

# %%
df = pd.DataFrame(
    columns=["appId", "total_reviews", "growth", "cluster", "minInstalls", "score"],
    data=review_dates[
        ["appId", "total_reviews", "2021_growth", "cluster", "minInstalls", "score"]
    ].values,
    index=review_dates.index,
)

# %%
plot_df = df[(df["score"] >= 4) & (df["growth"] < 50)]

# %%
xscale = alt.Scale(type="log")
yscale = alt.Scale(base=10)

fig = (
    alt.Chart(plot_df, width=650, height=650)
    .mark_circle(filled=True, size=50)
    .encode(
        alt.X("total_reviews", scale=xscale),
        alt.Y("growth", scale=yscale),
        color="cluster:N",
        tooltip=["appId", "cluster", "minInstalls"],
    )
).interactive()

fig

# %%
xscale = alt.Scale()
yscale = alt.Scale(domain=[0, 85])
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
        alt.Color("cluster:N"),
        # alt.Size("growth"),
        tooltip=["appId", "cluster", "minInstalls"],
    )
).interactive()

fig2

# %%
app_details["ratings"].replace([None], [0], inplace=True)
app_details["compound_score"] = (
    app_details["minInstalls"] * app_details["ratings"] * app_details["reviews"]
) + 0.1
app_details["log_score"] = app_details["compound_score"].apply(math.log10)
app_details.sort_values("compound_score", ascending=False, inplace=True)

# %%
xscale = alt.Scale(type="log")
yscale = alt.Scale(type="pow")
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
xscale = alt.Scale(type="sqrt")
yscale = alt.Scale(domain=[3, 5])
fig = (
    alt.Chart(app_details[app_details["score"] > 3], width=500, height=500)
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
app_details.columns

# %%
plotter = app_details[
    [
        "appId",
        "cluster",
        "minInstalls",
        "score",
        "ratings",
        "reviews",
        "price",
        "free",
        "containsAds",
        "offersIAP",
    ]
]

# %%
plotter.sample(5)

# %%
plotter = (
    plotter.groupby("cluster")
    .agg(
        cluster_size=("cluster", "count"),
        free=("free", "sum"),
        IAPs=("offersIAP", "sum"),
        ads=("containsAds", "sum"),
    )
    .reset_index()
)

turn_to_pc = ["free", "ads", "IAPs"]
for i in turn_to_pc:
    plotter[f"{i}_pc"] = plotter[i] / plotter.cluster_size

plotter

# %%
base = (
    alt.Chart()
    .mark_point()
    .encode(color="cluster:N")
    .properties(width=250, height=250)
    .interactive()
)

# %%

fig = (
    alt.Chart(plotter)
    .mark_bar()
    .encode(
        x=alt.X("cluster:N"),
        y=alt.Y("IAPs_pc:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("cluster"),
    )
)

fig

# %%
data_map = {
    "free_pc": "Number of free apps",
    "IAPs_pc": "Number of apps with in-app purchases",
    "ads_pc": "Number of apps with ads",
}

# %%
df = plotter.sort_values("free_pc", ascending=False)
bar_width = round(1 / len(data_map), 2) - 0.1

fig, ax = plt.subplots(figsize=(15, 10))
plt.setp(
    ax.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize="medium"
)
plt.grid(visible=True, axis="y", which="major")
ax.set_ylabel("Percentage of apps")

x = np.arange(len(df.cluster))
for i, (key, value) in enumerate(data_map.items()):
    ax.bar(x + (i * bar_width), df[key], label=data_map[key], width=bar_width)

ax.set_xticks(x + (len(data_map) * bar_width) / len(data_map))
ax.set_xticklabels(df.cluster.unique())


fig.legend(loc="upper right")

# %%
from scipy import stats

app_costs = app_details[app_details.price > 0].price.to_list()
print(
    len(app_costs),
    np.mean(app_costs),
    np.median(app_costs),
    stats.mode(app_costs)[0],
    np.min(app_costs),
    np.max(app_costs),
)

# %%
app_details[app_details["price"] == 2.99]

# %%
price_distro = (
    app_details.groupby("price")
    .agg(
        appId=("appId", "count"),
    )
    .reset_index()
)
price_distro.drop(price_distro.index[0], inplace=True)

# %%
fig = (
    alt.Chart(price_distro, width=700, height=500)
    .mark_circle(size=50)
    .encode(x="price", y="appId")
)
fig

# %%
needle = "buddy"
focus_apps[focus_apps["appId"].str.contains(needle, regex=False)]
