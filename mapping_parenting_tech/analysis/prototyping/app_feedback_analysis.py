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
from tqdm.notebook import tqdm

DATA_DIR = PROJECT_DIR / "outputs/data"
REVIEWS_DIR = DATA_DIR / "app_reviews"
INPUT_DIR = PROJECT_DIR / "inputs/data/play_store"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# %%
# get list of relevant ('focus') apps, as identified by automatic and then manual review

focus_apps = pd.read_csv(INPUT_DIR / "relevant_app_ids.csv", header=0)
focus_apps_list = focus_apps["appId"].tolist()

# %%
# load app reviews and add the cluster that each app is in to the review

app_reviews = psu.load_some_app_reviews(focus_apps_list)
app_reviews["rYear"] = pd.to_datetime(app_reviews["at"]).dt.year
app_reviews["rMonth"] = pd.to_datetime(app_reviews["at"]).dt.month
app_reviews = app_reviews.merge(focus_apps, on="appId")

# %%
# load app details
# Note that...
# 1. only apps of interest are retained
# 2. the cluster for each app is added
# 3. `score` and `minInstalls` are converted to float and int, respetively
# 4. `score` is rounded to 1 decimal place

app_details = pd.DataFrame(psu.load_all_app_details()).T
logging.info("DataFrame loaded")
app_details.reset_index(inplace=True)
app_details.rename(columns={"index": "appId"}, inplace=True)
app_details = app_details[app_details["appId"].isin(focus_apps_list)]
app_details = app_details.merge(focus_apps, on="appId")
app_details = app_details.astype({"score": "float16", "minInstalls": "int64"})
# app_details.loc[:, "minInstalls"] = app_details["minInstalls"].map('{:,d}'.format)
app_details["score"] = np.around(app_details["score"].to_list(), 1)

# %%
# add `minInstalls` to apps' reviews

app_reviews = app_reviews.merge(app_details[["appId", "minInstalls"]], on="appId")

# %% [markdown]
# ## Some basic insights into the apps

# %% [markdown]
# ### Popularity and scores
# How much are apps downloaded and do highly downloaded apps have better scores?
# The initial figures are derived from apps' details - i.e., their `minInstalls` count, and their score, all grouped by cluster and averaged accordingly.

# %%
# group by  minInstalls and then
# 1. count how many apps have been installed that many (`minInstall`) times
# 2. get the average score for the apps that have been installed that many times

app_installs_df = app_details.groupby("minInstalls").agg(
    installCount=("minInstalls", "count"), av_score=("score", "mean")
)

# %%
base = alt.Chart(app_installs_df.reset_index(), width=700, height=700).encode(
    x=alt.X("minInstalls", scale=alt.Scale(type="log"))
)
counts = base.mark_point(size=60, filled=True).encode(
    alt.Y("installCount", axis=alt.Axis(title="Count of number of installs"))
)
scores = base.mark_line(stroke="red").encode(
    alt.Y("av_score", axis=alt.Axis(title="Average score"))
)
alt.layer(counts, scores).resolve_scale(y="independent")

# %% [markdown]
# Looking at just a single cluster, as defined by `needle`

# %%
needle = "Tracking babies' rhythms"
app_installs_df = (
    app_details[app_details.cluster == needle]
    .groupby("minInstalls")
    .agg(installCount=("minInstalls", "count"), av_score=("score", "mean"))
)

# %%
base = alt.Chart(app_installs_df.reset_index(), width=700, height=700).encode(
    x=alt.X("minInstalls", scale=alt.Scale(type="log"))
)
counts = base.mark_point(size=50).encode(
    alt.Y("installCount", axis=alt.Axis(title="Count of number of installs")),
)
scores = base.mark_line(stroke="red").encode(
    alt.Y("av_score", axis=alt.Axis(title="Average score")),
)
alt.layer(counts, scores).resolve_scale(y="independent")

# %% [markdown]
# ## Comparisons with Play Store

# %%
foo = pd.read_csv("~/Downloads/category_sizes.csv", index_col=0)
foo.head()

# %%
plot_df = (
    app_details.groupby("genre")
    .agg(
        sample_cat_pc=("appId", "count"),
    )
    .divide(len(app_details))
    .multiply(100)
)

plot_df = plot_df.merge(foo, how="left", left_on="genre", right_on="Category")

# %%
base = alt.Chart(plot_df, width=700, height=550).encode(
    y=alt.Y(
        "Category:N", sort=alt.EncodingSortField(field="size_pc", order="descending")
    ),
)
store_fig = base.mark_bar(color="blue", opacity=0.3).encode(x="size_pc")
sample_fig = base.mark_bar(color="red", opacity=0.3).encode(x="sample_cat_pc")

alt.layer(store_fig, sample_fig).resolve_scale(
    x="shared",
)

# %%
alt.Chart(plot_df, width=700, height=600).mark_bar().encode(
    y=alt.Y(
        "Category:N",
        sort=alt.EncodingSortField(field="sample_cat_pc", order="descending"),
    ),
    x=alt.X("sample_cat_pc"),
)

# %%
plot_df = (
    app_details.loc[app_details.score > 0]
    .groupby("cluster")
    .agg(
        cluster_size=("cluster", "count"),
        free=("free", "sum"),
        IAPs=("offersIAP", "sum"),
        ads=("containsAds", "sum"),
        score_mean=("score", "mean"),
        score_median=("score", "median"),
        score_sd=("score", np.std),
        score_iqr=("score", lambda x: np.subtract(*np.percentile(x, [75, 25]))),
        downloaded_mean=("minInstalls", "mean"),
        downloaded_median=("minInstalls", "median"),
    )
    .reset_index()
    .round(2)
    .sort_values("score_mean")
)
plot_df

# %%
plt_column = "score"
bars = (
    alt.Chart(
        app_details[app_details.score > 0].sort_values(plt_column, ascending=True),
        width=700,
        height=400,
    )
    .mark_bar()
    .encode(
        y=alt.Y(f"mean({plt_column}):Q", scale=alt.Scale(zero=True)),
        x=alt.X(
            "cluster:N",
            sort=alt.EncodingSortField(field=f"mean({plt_column})", order="ascending"),
        ),
        color="cluster",
    )
)

error_bars = (
    alt.Chart(app_details[app_details.score > 0])
    .mark_errorbar(extent="stdev")
    .encode(
        y=alt.Y(f"{plt_column}:Q", scale=alt.Scale(zero=False)), x=alt.X("cluster:N")
    )
)
bars + error_bars

# %%
fig = (
    alt.Chart(app_details.loc[(app_details.score > 0)], width=700, height=700)
    .mark_line()
    .encode(
        x=alt.X("score:Q", bin=True),
        y="count(score)",
        color="cluster:N",
        tooltip="cluster",
    )
)
fig

# %% [markdown]
# ## Look at reviews by date
# Frequency of reviews etc.

# %%
# Create a new dataframe, `reviewDates`, with the number of reviews for each app per year
review_dates = (
    app_reviews.groupby(["appId", "rYear"])["appId"].count().unstack().reset_index()
)
app_total_reviews = app_reviews.groupby(["appId"])["appId"].count()
review_dates["total_reviews"] = review_dates["appId"].map(app_total_reviews)
review_dates = review_dates.merge(focus_apps, on=["appId"])
review_dates.shape


# %%
# Add growth column - how much did each app grow by in 2021 compared to 2020?
def growth_2021(row):
    if (row[2021] > 0) and (row[2020] > 0):
        return row[2021] / row[2020]
    elif row[2021] > 0:
        return 1
    else:
        return 0


review_dates["2021_growth"] = review_dates.apply(growth_2021, axis=1)

# %%
# Add extra app details, namely their category, score and number of installs
review_dates = review_dates.merge(
    app_details[["appId", "genre", "score", "minInstalls"]], on="appId"
)

# %%
# Move all this into a new dataframe
df = pd.DataFrame(
    columns=["appId", "total_reviews", "growth", "cluster", "minInstalls", "score"],
    data=review_dates[
        ["appId", "total_reviews", "2021_growth", "cluster", "minInstalls", "score"]
    ].values,
    index=review_dates.index,
)

# %%
# And another dataframe to use for the plot - selecting highly rated apps whose growth isn't 'excessive'
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
basic_app_details = app_details[
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
plotter = (
    basic_app_details.groupby("cluster")
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
foo = app_details.groupby("score")["score"].count()

# %%
foo = foo.drop(index=[0.000000])

# %%
score_fig = plt.scatter(x=range(0, len(foo)), y=foo)
score_fig

# %%
app_details["released"] = pd.to_datetime(app_details["released"])
app_details["releaseYear"] = app_details["released"].dt.year
app_details.head()

# %%
app_growth = app_details.groupby(["releaseYear", "cluster"], as_index=False).agg(
    app_count=("appId", "count")
)
app_growth.sort_values(by=["cluster", "releaseYear"], inplace=True, ignore_index=True)
app_growth["growth"] = app_growth["app_count"].pct_change()
app_growth[app_growth.cluster == "Parental support"]
# app_growth["growth"].mean()
app_growth[app_growth.cluster == "Parental support"].plot.bar(
    x="releaseYear", y="growth", figsize=(10, 7)
)

# %%
app_growth.groupby("cluster").agg(average_growth=("growth", "mean")).sort_values(
    "average_growth"
)

# %%
cluster_growth = (
    app_details.groupby(by=["cluster", "releaseYear"])["appId"]
    .count()
    .cumsum()
    .reset_index()
)
cluster_growth.rename(columns={"appId": "app_count"}, inplace=True)
cluster_growth["normalised"] = cluster_growth["app_count"] / cluster_growth.groupby(
    "cluster"
)["app_count"].transform("max")

# %%
fig = (
    alt.Chart(cluster_growth, width=700, height=700)
    .mark_line()
    .encode(
        x="releaseYear:N",
        y="app_count:Q",
        color="cluster:N",
        tooltip=["cluster"],
    )
).interactive()
fig

# %%
review_growth = (
    app_reviews[app_reviews.rYear < 2022]
    .groupby(["cluster", "rYear"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
)
review_growth["cumulative_count"] = review_growth.groupby(["cluster"], as_index=False)[
    "review_count"
].cumsum()

cluster_sizes = focus_apps.groupby("cluster", as_index=False).agg(
    cluster_size=("cluster", "count")
)

review_growth = review_growth.merge(cluster_sizes, on="cluster")
review_growth["normalised_by_cluster"] = (
    review_growth["cumulative_count"] / review_growth["cluster_size"]
)
review_growth["normalised"] = review_growth[
    "normalised_by_cluster"
] / review_growth.groupby("cluster")["normalised_by_cluster"].transform("min")

# %%
review_growth[review_growth.cluster == "Parental support"]

# %%
fig = (
    alt.Chart(
        review_growth[review_growth.cluster != "General play"], width=700, height=600
    )
    .mark_line()
    .encode(
        x="rYear:N",
        y=alt.Y(
            "normalised:Q",
            axis=alt.Axis(
                title="Cumulative growth as reviews per app, normalised to 100%"
            ),
            scale=alt.Scale(type="linear"),
        ),
        color="cluster",
        tooltip="cluster:N",
    )
)
fig

# %%
start_year = 2018
end_year = 2020
growth_df = app_reviews[
    (app_reviews["rYear"] == start_year) | (app_reviews["rYear"] == end_year)
]
growth_df = growth_df.groupby(["cluster", "rYear"], as_index=False).agg(
    review_count=("reviewId", "count")
)
growth_df["abs_growth"] = growth_df.groupby("cluster")["review_count"].transform(
    "max"
) - growth_df.groupby("cluster")["review_count"].transform("min")
growth_df

# %%
cluster_sizes = (
    app_details[app_details.releaseYear <= end_year]
    .groupby("cluster", as_index=False)
    .agg(cluster_size=("appId", "count"))
)

# %%
growth_df["growth"] = growth_df["review_count"].pct_change()
growth_df = growth_df[growth_df["rYear"] == end_year]
growth_df = growth_df.merge(cluster_sizes, on="cluster")
growth_df["normalised_by_cluster_size"] = (
    growth_df["abs_growth"] / growth_df["cluster_size"]
)

# %%
growth_df

# %%
fig = (
    alt.Chart(growth_df, width=600, height=550)
    .mark_circle()
    .encode(
        x=alt.X(
            "review_count:Q",
            axis=alt.Axis(title=f"Number of reviews in {end_year}"),
            scale=alt.Scale(type="linear"),
        ),
        y=alt.Y(
            "normalised_by_cluster_size:Q",
            axis=alt.Axis(
                title=f"Growth between {start_year} and {end_year} measured by number of reviews"
            ),
            scale=alt.Scale(type="linear"),
        ),
        size="cluster_size:Q",
        color="cluster:N",
        tooltip="cluster:N",
    )
)
fig
