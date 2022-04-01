# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
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
OUTPUT_DIR = PROJECT_DIR / "outputs/data"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# %%
# get list of relevant ('focus') apps, as identified by automatic and then manual review

focus_apps = pd.read_csv(INPUT_DIR / "relevant_app_ids.csv", header=0)
focus_apps_list = focus_apps["appId"].tolist()

# %%
# load app reviews and add the cluster that each app is in to the review

app_reviews = psu.load_some_app_reviews(focus_apps_list)
app_reviews["reviewYear"] = pd.to_datetime(app_reviews["at"]).dt.year
app_reviews["reviewMonth"] = pd.to_datetime(app_reviews["at"]).dt.month
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
focus_apps["cluster"].unique().tolist()

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
plot_df = app_details.groupby("cluster", as_index=False).agg(
    app_count=("appId", np.count_nonzero)
)

plot_df.sort_values("app_count", ascending=False).plot.bar(
    x="cluster", y=["app_count"], figsize=(10, 8)
)

# %%
# group by minInstalls and then
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
    alt.Y("installCount", axis=alt.Axis(title="Number of apps"))
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

# %%
app_installs_by_cluster = app_details.groupby(
    ["cluster", "minInstalls"], as_index=False
).agg(
    appCount=("minInstalls", "count"),
)

app_installs_by_cluster["totalInstalls"] = (
    app_installs_by_cluster["appCount"] * app_installs_by_cluster["minInstalls"]
)


fig = (
    alt.Chart(app_installs_by_cluster)
    .mark_point()
    .encode(
        alt.X("minInstalls:Q", scale=alt.Scale(type="log"), axis=alt.Axis(tickCount=6)),
        y="appCount:Q",
        color="cluster:N",
        facet=alt.Facet(
            "cluster:N", columns=4, sort=alt.EncodingSortField("totalInstalls", "sum")
        ),
        tooltip=["appCount", "minInstalls"],
    )
    .properties(width=200, height=175)
)

fig


# %% [markdown]
# ## Comparisons with Play Store

# %%
# Load the number of apps in each category on the Play Store

cat_sizes = pd.read_csv(INPUT_DIR / "category_sizes.csv", index_col=0)
cat_sizes.head()

# %%
plot_df = (
    app_details.groupby("genre")
    .agg(
        sample_cat_pc=("appId", "count"),
    )
    .divide(len(app_details))
    .multiply(100)
)

plot_df = plot_df.merge(cat_sizes, how="left", left_on="genre", right_on="Category")

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
plot_df = app_details[["appId", "released"]]
plot_df["year_released"] = pd.DatetimeIndex(plot_df["released"]).year
plot_df["month_released"] = pd.DatetimeIndex(plot_df["released"]).month
plot_df = plot_df.groupby("year_released", as_index=False).agg(
    app_count=("appId", "count"),
    months_in_year=("month_released", lambda x: x.nunique()),
)

plot_df["apps_per_month"] = plot_df["app_count"] / plot_df["months_in_year"]
plot_df["growth"] = plot_df["apps_per_month"].pct_change()
plot_df.plot.line(x="year_released", y=["growth"], figsize=(10, 8), ylim=(-0.5, 2.2))

# %%
# Plot the number of apps released each year in each cluster

fig = plt.figure(figsize=(10, 8))

for cluster in [
    "Parental support",
    "Tracking babies' rhythms",
    "Fertility tracking",
    "Pregnancy tracking",
]:
    plot_df = app_details.loc[app_details["cluster"] == cluster][["appId", "released"]]
    plot_df["year_released"] = pd.DatetimeIndex(plot_df["released"]).year
    plot_df["month_released"] = pd.DatetimeIndex(plot_df["released"]).month
    plot_df = plot_df.groupby("year_released", as_index=False).agg(
        app_count=("appId", "count"),
        months_in_year=("month_released", lambda x: x.nunique()),
    )

    plot_df["apps_per_month"] = plot_df["app_count"] / plot_df["months_in_year"]
    plot_df["growth"] = plot_df["apps_per_month"].pct_change()

    plt.plot(plot_df["year_released"], plot_df["growth"], label=cluster)

plt.legend(loc="upper right")
fig.show()

# %%
play_store_growth = pd.read_csv(OUTPUT_DIR / "play_store_growth.csv", index_col=0)

# %%
fig = plt.figure(figsize=(10, 8))

plt_label = ["Play store", "Sample"]

for i, frame in enumerate([play_store_growth, plot_df]):
    plt.plot(frame["year_released"], frame["growth"], label=plt_label[i])

fig.legend(loc="upper right")
plt.ylabel("Year on year growth")
plt.xlabel("Release year")
fig.show()


# %%

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

# %% [markdown]
# ### Apps' average scores by cluster

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
# ## Using reviews over time as proxy for growth
# Frequency of reviews etc.

# %%
# Create a new dataframe, `reviewDates`, with the number of reviews for each app per year

review_dates = (
    app_reviews.groupby(["appId", "reviewYear"])["appId"]
    .count()
    .unstack()
    .reset_index()
)
app_total_reviews = app_reviews.groupby(["appId"])["appId"].count()
review_dates["total_reviews"] = review_dates["appId"].map(app_total_reviews)
review_dates = review_dates.merge(focus_apps, on=["appId"])

# %%
review_dates.total_reviews.max()


# %%
# Add growth column - how much did each app grow by in 2021 compared to 2020?
def growth_2021(row):
    if (row[2021] > 0) and (row[2019] > 0):
        return row[2021] / row[2019]
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
# Extract columns of interest into a new dataframe, `df`

df = pd.DataFrame(
    columns=["appId", "total_reviews", "growth", "cluster", "minInstalls", "score"],
    data=review_dates[
        ["appId", 2021, "2021_growth", "cluster", "minInstalls", "score"]
    ].values,
    index=review_dates.index,
)

# %%
# identify appropriate thresholds for segmenting data

percentile = 90
print(f"Median of `total_reviews`:", np.percentile(df.total_reviews, 50))
print(f"{percentile} percentile of `growth`:", np.percentile(df.growth, percentile))
print(f"{percentile} of `score`:", np.percentile(df.score, percentile))

# %%
growth_threshold = 200
score_threshold = 3.8
hot_growth = 4.23
hot_review_count = 29

hot_apps = df.loc[
    (df["score"] >= score_threshold)
    & (df["growth"] >= hot_growth)
    & (df["total_reviews"] >= hot_review_count)
]
hot_apps = hot_apps.assign(status="Hot")

emerging_apps = df.loc[
    (df["score"] >= score_threshold)
    & (df["growth"] >= hot_growth)
    & (df["total_reviews"] < hot_review_count)
]
emerging_apps = emerging_apps.assign(status="Emerging")

interesting_apps = pd.concat([hot_apps, emerging_apps])

summary_df = interesting_apps.groupby(["status", "cluster"]).agg(
    app_count=("appId", "count"),
    median_score=("score", np.median),
    median_growth=("growth", np.median),
)
summary_df["median_growth"] = np.round(summary_df["median_growth"], 2)
summary_df.sort_values(["status", "median_growth"], ascending=False)

# %%
_top_apps = hot_apps[hot_apps["cluster"] == "Literacy - English / Reading and Stories"][
    "appId"
].to_list()
review_dates[review_dates["appId"].isin(_top_apps)]

# %%
check_df = review_dates[review_dates["cluster"] == "Pregnancy tracking"][
    ["appId", 2019, 2021, "2021_growth"]
]
print("Total number of reviews in 2019:\t\t", check_df[2019].sum())
print("Total number of reviews in 2021:\t\t", check_df[2021].sum())
print("Average change between 2019 and 2021:\t", check_df["2021_growth"].mean())
check_df

# %%
df[df.cluster == "Pregnancy tracking"].growth.mean()

# %%
df["total_reviews"].max()

# %%
# And another dataframe to use for the plot - selecting highly rated apps whose growth isn't 'excessive'
plot_df = df[(df["score"] >= score_threshold) & (df["growth"] < growth_threshold)]

xscale = alt.Scale(type="log")
yscale = alt.Scale(base=10)

fig = (
    alt.Chart(plot_df, width=650, height=650)
    .mark_circle(filled=True, size=60)
    .encode(
        alt.X("total_reviews", scale=xscale, title="Total reviews in 2021"),
        alt.Y("growth", scale=yscale),
        color="cluster:N",
        tooltip=["appId", "cluster", "minInstalls"],
        # size="minInstalls:Q",
    )
)  # .interactive()

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

fig, ax = plt.subplots(figsize=(15, 9))
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
print("Mean free:", df.free_pc.mean())
print("Mean ads:", df.ads_pc.mean())
print("Mean IAPs:", df.IAPs_pc.mean())

# %%
df

# %%
# Extract the year that each app was released and place in a separate column
# app_details["released"] = pd.to_datetime(app_details["released"])
app_details["releaseYear"] = pd.to_datetime(app_details["released"]).dt.year

# %%
app_growth = app_details.groupby(["releaseYear", "cluster"], as_index=False).agg(
    app_count=("appId", "count")
)
app_growth.sort_values(by=["releaseYear"], inplace=True, ignore_index=True)
app_growth["growth"] = app_growth["app_count"].pct_change()
# app_growth[app_growth.cluster == "Parental support"]
# app_growth["growth"].mean()
app_growth.plot.bar(x="releaseYear", y="growth", figsize=(10, 7))

# %%
app_growth["app_count"].sum()

# %%
app_growth.groupby("cluster").agg(average_growth=("growth", "mean")).sort_values(
    "average_growth"
)

# %%
cluster_size_by_year = (
    app_details.groupby(by=["cluster", "releaseYear"])["appId"]
    .count()
    .cumsum()
    .reset_index()
)
cluster_size_by_year.rename(columns={"appId": "app_count"}, inplace=True)
cluster_size_by_year["normalised"] = cluster_size_by_year[
    "app_count"
] / cluster_size_by_year.groupby("cluster")["app_count"].transform("max")
cluster_size_by_year

# %%
fig = (
    alt.Chart(cluster_size_by_year, width=700, height=700)
    .mark_line()
    .encode(
        x="releaseYear:N",
        y="app_count:Q",
        color="cluster:N",
        tooltip=["cluster"],
    )
)
fig

# %%
review_growth = (
    app_reviews[app_reviews.reviewYear < 2022]
    .groupby(["cluster", "reviewYear"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
)
review_growth["cumulative_count"] = review_growth.groupby(["cluster"], as_index=False)[
    "review_count"
].cumsum()

review_growth = review_growth.merge(
    cluster_size_by_year,
    how="left",
    left_on=["cluster", "reviewYear"],
    right_on=["cluster", "releaseYear"],
)
review_growth["app_count"].ffill(inplace=True)
review_growth.drop(columns=["releaseYear"])

review_growth["normalised_by_cluster"] = (
    review_growth["review_count"] / review_growth["app_count"]
)

review_growth["normalised"] = review_growth[
    "normalised_by_cluster"
] / review_growth.groupby("cluster")["normalised_by_cluster"].transform("min")

# %%
review_growth[review_growth.cluster == "Pregnancy tracking"]

# %%
fig = (
    alt.Chart(
        review_growth[review_growth.cluster != "General play"], width=700, height=600
    )
    .mark_line()
    .encode(
        x="reviewYear:N",
        y=alt.Y(
            "normalised:Q",
            axis=alt.Axis(
                title="Number of reviews per app, normalised to number of reviews in clusters' first year"
            ),
            scale=alt.Scale(type="linear"),
        ),
        color="cluster",
        tooltip="cluster:N",
    )
)
fig

# %%
start_year = 2019
end_year = 2021
growth_df = app_reviews[
    (app_reviews["reviewYear"] == start_year) | (app_reviews["reviewYear"] == end_year)
]
growth_df = growth_df.groupby(["cluster", "reviewYear"], as_index=False).agg(
    review_count=("reviewId", "count")
)

growth_df["growth_change"] = growth_df["review_count"].diff()
growth_df

# %%
cluster_sizes = (
    app_details[app_details.releaseYear <= end_year]
    .groupby("cluster", as_index=False)
    .agg(cluster_size=("appId", "count"))
)
growth_df = growth_df.merge(cluster_sizes, on="cluster")

# %%
growth_df["growth"] = growth_df["review_count"].pct_change()

growth_df["normalised_by_cluster_size"] = (
    growth_df["growth"] / growth_df["cluster_size"]
) * 100

growth_df

# %%
growth_df = growth_df[growth_df["reviewYear"] == end_year]

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
            "growth:Q",
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

# %%
review_growth_by_app = review_dates[
    ["appId", 2019, 2021, "2021_growth", "cluster", "minInstalls", "score"]
]
review_growth_by_app[[2019, 2021]].fillna(1, inplace=True)
review_growth_by_app["pc_change"] = (
    review_growth_by_app[2021] / review_growth_by_app[2019]
) - 1

growth_summary = review_growth_by_app.groupby("cluster").agg(
    mean_change=("pc_change", np.mean),
    stdev=("pc_change", np.std),
    median_change=("pc_change", np.median),
    IQR=("pc_change", lambda x: np.subtract(x.quantile(0.75), x.quantile(0.25))),
)

growth_summary