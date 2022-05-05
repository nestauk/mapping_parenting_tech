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
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
import mapping_parenting_tech.utils.plotting_utils as pu
import importlib
import utils

importlib.reload(utils)


# %%
# Functionality for saving charts
import mapping_parenting_tech.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver()

# %% [markdown]
# ### Load app info

# %%
app_details = utils.get_app_details()
app_reviews = utils.get_app_reviews()

# %%
sorted(app_details["cluster"].unique().tolist())

# %% [markdown]
# ## Helper functions

# %%
check_columns = ["title", "description", "summary", "installs", "score"]

# %%
############ REFACTOR THESE OUT INTO utils.py

# cluster_names = [
#     'Drawing and colouring',
#     'Simple puzzles',
#     'Literacy - English / ABCs',
#     'Literacy - English / Reading and Stories',
#     'Literacy - non-English',
#     'Numeracy development',
#     'Learning play',
#     'General play',
#     "Tracking babies' rhythms",
#     'Helping babies to sleep',
#     'Parental support',
#     'Pregnancy tracking',
#     'Fertility tracking',
#     'Baby photos',
# ]

# clusters_children = [
#     'Drawing and colouring',
#     'Simple puzzles',
#     'Literacy - English / ABCs',
#     'Literacy - English / Reading and Stories',
#     'Literacy - non-English',
#     'Numeracy development',
#     'Learning play',
#     'General play',
# ]

# clusters_parents = [
#     "Tracking babies' rhythms",
#     'Helping babies to sleep',
#     'Parental support',
#     'Pregnancy tracking',
#     'Fertility tracking',
#     'Baby photos',
# ]

# clusters_literacy = [
#     'Literacy - English / ABCs',
#     'Literacy - English / Reading and Stories',
#     'Literacy - non-English'
# ]

# clusters_simple = [
#     'Drawing and colouring',
#     'Simple puzzles',
# ]

# clusters_play = [
#     'Learning play',
#     'General play',
# ]

# # Mapping clusters to the type of user (Parents vs Children)
# cluster_to_user_dict = dict(zip(
#     clusters_parents + clusters_children,
#     ['Parents']*len(clusters_parents) + ['Children']*len(clusters_children)))

# def map_cluster_to_user(cluster: str) -> str:
#     return cluster_to_user_dict[cluster]


# %% [markdown]
# ## Basic insights into the apps

# %% [markdown]
# ### DfE apps (reference)

# %%
dfe_apps = [
    "com.phonicshero.phonicshero",
    "com.fishinabottle.navigo",
    "com.auristech.fonetti",
    "com.lingumi.lingumiplay",
    "com.learnandgo.kaligo.homemena",
    "com.teachyourmonstertoread.tmapp",
]

# %%
app_details.query("@dfe_apps in appId")[check_columns]

# %% [markdown]
# ### Popularity and scores
# How much are apps downloaded and do highly downloaded apps have better scores?
# The initial figures are derived from apps' details - i.e., their `minInstalls` count, and their score, all grouped by cluster and averaged accordingly.

# %%
users_to_plot = ["Children"]
labels_title = "Category"
values_title = "Number of apps"

app_counts = (
    app_details.query(f"user in @users_to_plot")
    .groupby("cluster", as_index=False)
    .agg(app_count=("appId", np.count_nonzero))
    .sort_values("app_count", ascending=False)
    .rename(
        columns={
            "cluster": labels_title,
            "app_count": values_title,
        }
    )
)

# %%
app_counts

# %%
tooltip = [labels_title, values_title]
color = pu.NESTA_COLOURS[0]

chart_title = "Number of apps for kids"
chart_subtitle = ""

fig = (
    alt.Chart(
        app_counts,
        width=300,
        height=300,
    )
    .mark_bar(color=color)
    .encode(
        x=alt.X(
            f"{values_title}:Q", title=values_title, scale=alt.Scale(domain=(0, 260))
        ),
        y=alt.Y(
            f"{labels_title}:N",
            title=labels_title,
            sort="-x",
            axis=alt.Axis(labelLimit=200),
        ),
        tooltip=tooltip,
    )
    .properties(
        title={
            "anchor": "start",
            "text": chart_title,
            "subtitle": chart_subtitle,
            "subtitleFont": pu.FONT,
        },
    )
    .configure_axis(
        gridDash=[1, 7],
        gridColor="grey",
    )
    .configure_view(strokeWidth=0)
    .interactive()
)
fig

# %%
importlib.reload(utils)
table_name = "no_of_children_apps"
utils.save_data_table(app_counts, table_name)

# %%
AltairSaver.save(fig, table_name, filetypes=["html", "png"])

# %%
# Check the total number of apps
n_total = len(app_details)
n_total

# %%
# Apps for children
print(utils.percentage_in_cluster(app_details, utils.clusters_children, False))
print(utils.percentage_in_cluster(app_details, utils.clusters_children))

# %%
# Apps for parents
print(utils.percentage_in_cluster(app_details, utils.clusters_parents, False))
print(utils.percentage_in_cluster(app_details, utils.clusters_parents))

# %%
# Ratio children vs parents
percentage_in_cluster(app_details, clusters_children, False) / percentage_in_cluster(
    app_details, clusters_parents, False
)

# %%
# Apps for literacy
p = percentage_in_cluster(app_details, clusters_literacy)
n = percentage_in_cluster(app_details, clusters_literacy, False)
n, p

# %%
p = percentage_in_cluster(app_details, ["Numeracy development"])
n = percentage_in_cluster(app_details, ["Numeracy development"], False)
n, p

# %%
(241 + 79) / 896

# %%
p = percentage_in_cluster(app_details, clusters_play)
n = percentage_in_cluster(app_details, clusters_play, False)
n, p

# %%
506 / 896

# %%
p = percentage_in_cluster(app_details, clusters_simple)
n = percentage_in_cluster(app_details, clusters_simple, False)
n, p

# %%
70 / 896

# %%
get_top_cluster_apps(app_details, "Literacy - English / ABCs", top_n=5)

# %%
get_top_cluster_apps(app_details, "Literacy - English / Reading and Stories", top_n=5)

# %%
get_top_cluster_apps(app_details, "Numeracy development", top_n=5)

# %%
get_top_cluster_apps(app_details, "Learning play", top_n=20)

# %%
# get_top_cluster_apps(app_details, 'Drawing and colouring', 'score', 100)

# %%
app_details["minInstalls"].median()

# %%
len(app_details.query("minInstalls > 1e+6"))

# %%
154 / len(app_details)

# %%
# group by minInstalls and then
# 1. count how many apps have been installed that many (`minInstall`) times
# 2. get the average score for the apps that have been installed that many times

app_installs_df = app_details.groupby("minInstalls").agg(
    installCount=("minInstalls", "count"), av_score=("score", "mean")
)

# %%
app_installs_df_["total_installs"] = (
    app_installs_df_["minInstalls"] * app_installs_df_["minInstalls"]
)

# %%
n = app_details.minInstalls.sort_values()
percentage_of_installs = np.cumsum(n)[1:] / np.sum(n) * 100
percentage_of_apps = (np.array(range(1, len(n), 1)) / len(n)) * 100

df = pd.DataFrame(
    data={
        "percentage_of_apps": percentage_of_apps,
        "percentage_of_installs": percentage_of_installs,
    }
)

(
    alt.Chart(df)
    .mark_line(color="blue")
    .encode(
        alt.Y("percentage_of_apps:Q"),
        alt.X("percentage_of_installs:Q"),
        tooltip=["percentage_of_apps", "percentage_of_installs"],
    )
).interactive()

# %%
df[df.percentage_of_installs >= 50].head(3)

# %%
df[df.percentage_of_installs >= 20].head(3)

# %%
n = app_details.sort_values("minInstalls", ascending=False).head(36).minInstalls.sum()
n / app_details.minInstalls.sum()

# %%
app_details.sort_values("minInstalls", ascending=False).head(10)[
    check_columns + ["cluster"]
]

# %%
app_details.groupby("cluster").agg(total_installs=("minInstalls", "sum")).sort_values(
    "total_installs"
) / 1e6

# %%
app_details.groupby("cluster").agg(total_installs=("minInstalls", "mean")).sort_values(
    "total_installs"
) / 1e6

# %%
app_installs_df_ = app_installs_df.reset_index()

# %%
app_installs_df_.query("minInstalls > 1e+6").installCount.sum()

# %%
app_installs_df_.query("minInstalls > 1e+6").installCount.sum() / len(app_details)

# %%
1 / 0.128

# %%
# importlib.reload(pu)
# pu.bar_chart(
#     data = (
#         app_installs_df.reset_index()
#         .assign(minInstalls=lambda x: x.minInstalls.apply(lambda y: str(y)))
#     ),
#     values_column = 'installCount',
#     labels_column = 'minInstalls',
#     values_title = "Number of apps",
#     labels_title = "Installations",
#     tooltip = None,
#     color = None,
#     chart_title = '',
#     chart_subtitle = '',
# )

# %%
# (
#     alt.Chart(
#         app_installs_df.reset_index().assign(minInstalls=lambda x: x.minInstalls.apply(lambda y: str(y))),
#         width=400,
#         height=300,
#     )
#     .mark_bar(color='blue')
#     .encode(
#         alt.X('installCount:Q'),
#         alt.Y('minInstalls:O')
#     )
# )

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
# ## Example apps

# %%
cluster_name = "Parental support"
# cluster_name = 'Literacy - English / ABCs'

# %%
get_top_cluster_apps(app_details, cluster_name)[check_columns]

# %%
get_top_cluster_apps(app_details, cluster_name, "score")[
    ["title", "description", "summary", "installs", "score"]
]

# %%
get_top_cluster_apps(app_details, "Literacy - English / Reading and Stories")[
    check_columns
]

# %%
app_details.sort_values("minInstalls", ascending=False)[
    check_columns + ["cluster"]
].head(15)

# %% [markdown]
# ## Comparisons with Play Store

# %%
list(app_details.columns)

# %%
# Import Google Play Store data
google_playstore_df = pd.read_csv(INPUT_DIR / "Google-Playstore.csv")

# %%
google_playstore_df = google_playstore_df.rename(
    columns={
        "App Id": "appId",
        "Minimum Installs": "minInstalls",
        "Free": "free",
        "In App Purchases": "offersIAP",
    }
)

# %%
for col in ["free", "Ad Supported", "offersIAP"]:
    print(col, google_playstore_df[col].mean())

# %%
for col in ["free", "Ad Supported", "offersIAP"]:
    print(
        col,
        google_playstore_df[google_playstore_df["minInstalls"] > 100000][col].mean(),
    )

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
app_details.score.plot.hist()

# %%
app_details.query("score > 0").score.median()

# %%
app_details.query("score > 0").score.mean()

# %%
1 - len(app_details.query("score <= 0")) / len(app_details)

# %%
app_details.score.median()

# %%
app_details.query("score > 0").groupby("cluster").agg(
    score=("score", "mean")
).sort_values("score", ascending=False)

# %%
stripplot = (
    alt.Chart(app_details, width=40)
    .mark_circle(size=8)
    .encode(
        x=alt.X(
            "jitter:Q",
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y("score:Q"),
        color=alt.Color("cluster:N", legend=None),
        column=alt.Column(
            "cluster:N",
            header=alt.Header(
                labelAngle=-90,
                titleOrient="top",
                labelOrient="bottom",
                labelAlign="right",
                labelPadding=3,
            ),
        ),
    )
    .transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    )
    .configure_facet(spacing=0)
    .configure_view(stroke=None)
)

stripplot

# %%
importlib.reload(pu)
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
# importlib.reload(pu)
# plt_column = "score"
# bars = (
#     alt.Chart(
#         app_details[app_details.score > 0].sort_values(plt_column, ascending=True),
#         width=700,
#         height=400,
#     )
#     .mark_bar()
#     .encode(
#         y=alt.Y(f"mean({plt_column}):Q", scale=alt.Scale(zero=True)),
#         x=alt.X(
#             "cluster:N",
#             sort=alt.EncodingSortField(field=f"mean({plt_column})", order="ascending"),
#         ),
#         color="cluster",
#     )
# )

# error_bars = (
#     alt.Chart(app_details[app_details.score > 0])
#     .mark_errorbar(extent="stdev")
#     .encode(
#         y=alt.Y(f"{plt_column}:Q", scale=alt.Scale(zero=False)), x=alt.X("cluster:N")
#     )
# )
# bars + error_bars

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
].copy()
basic_app_details["user"] = basic_app_details.cluster.apply(map_cluster_to_user)

# %%
(
    basic_app_details.groupby("user").agg(
        free=("free", "mean"),
        containsAds=("containsAds", "mean"),
        offersIAP=("offersIAP", "mean"),
    )
)

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
df_ads = df.copy().assign(user=lambda x: x.cluster.apply(map_cluster_to_user))
df_ads

# %%
df_ads.groupby("user").mean()

# %%
df_ads.sort_values(["user", "ads_pc"], ascending=False)

# %%
df_ads.sort_values(["user", "IAPs_pc"], ascending=False)

# %% [markdown]
# ### Compare add percentage with a baseline percentage

# %%
app_details.info()

# %%
google_playstore_df.info()

# %%
app_installs_vs_ads_df = (
    app_details.assign(user=lambda x: x.cluster.apply(map_cluster_to_user))
    .query("user == 'Parents'")
    .groupby("minInstalls")
    .agg(
        # installCount=("minInstalls", "count"),
        # av_score=("score", "mean"),
        free=("free", "mean"),
        inapp=("offersIAP", "mean"),
        # ads=("containsAds", "mean"),
    )
)

app_installs_vs_ads_baseline = google_playstore_df.groupby("Minimum Installs").agg(
    # installCount=("Minimum Installs", "count"),
    # av_score=("Rating", "mean"),
    free_baseline=("Free", "mean"),
    inapp_baseline=("In App Purchases", "mean"),
    # ads=("containsAds", "mean"),
)

# %%
app_installs_vs_ads_df

# %%
app_installs_vs_ads_ = pd.concat(
    [app_installs_vs_ads_df, app_installs_vs_ads_baseline], axis=1
)

# %%
app_installs_vs_ads_.reset_index()


# %%
# df_ads.sort_values(['user', 'ads_pc'], ascending=False)

# %% [markdown]
# ## Trends

# %%
### Time series trends
def moving_average(
    timeseries_df: pd.DataFrame, window: int = 3, replace_columns: bool = False
) -> pd.DataFrame:
    """
    Calculates rolling mean of yearly timeseries (not centered)
    Args:
        timeseries_df: Should have a 'year' column and at least one other data column
        window: Window of the rolling mean
        rename_cols: If True, will create new set of columns for the moving average
            values with the name pattern `{column_name}_sma{window}` where sma
            stands for 'simple moving average'; otherwise this will replace the original columns
    Returns:
        Dataframe with moving average values
    """
    # Rolling mean
    df_ma = timeseries_df.rolling(window, min_periods=1).mean().drop("year", axis=1)
    # Create new renamed columns
    if not replace_columns:
        column_names = timeseries_df.drop("year", axis=1).columns
        new_column_names = ["{}_sma{}".format(s, window) for s in column_names]
        df_ma = df_ma.rename(columns=dict(zip(column_names, new_column_names)))
        return pd.concat([timeseries_df, df_ma], axis=1)
    else:
        return pd.concat([timeseries_df[["year"]], df_ma], axis=1)


def magnitude(time_series: pd.DataFrame, year_start: int, year_end: int) -> pd.Series:
    """
    Calculates signals' magnitude (i.e. mean across year_start and year_end)
    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
    Returns:
        Series with magnitude estimates for all data columns
    """
    magnitude = time_series.set_index("year").loc[year_start:year_end, :].mean()
    return magnitude


def percentage_change(initial_value, new_value):
    """Calculates percentage change from first_value to second_value"""
    return (new_value - initial_value) / initial_value * 100


def growth(
    time_series: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.Series:
    """Calculates a growth estimate
    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
    Returns:
        Series with smoothed growth estimates for all data columns
    """
    # Smooth timeseries
    df = time_series.set_index("year")
    # Percentage change
    return percentage_change(
        initial_value=df.loc[year_start, :], new_value=df.loc[year_end, :]
    )


def smoothed_growth(
    time_series: pd.DataFrame, year_start: int, year_end: int, window: int = 3
) -> pd.Series:
    """Calculates a growth estimate by using smoothed (rolling mean) time series
    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
        window: Moving average windows size (in years) for the smoothed growth estimate
    Returns:
        Series with smoothed growth estimates for all data columns
    """
    # Smooth timeseries
    ma_df = moving_average(time_series, window, replace_columns=True).set_index("year")
    # Percentage change
    return percentage_change(
        initial_value=ma_df.loc[year_start, :], new_value=ma_df.loc[year_end, :]
    )


def estimate_magnitude_growth(
    time_series: pd.DataFrame, year_start: int, year_end: int, window: int = 3
) -> pd.DataFrame:
    """
    Calculates signals' magnitude, estimates their growth and returns a combined dataframe
    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
        window: Moving average windows size (in years) for the smoothed growth estimate
    Returns:
        Dataframe with magnitude and growth trend estimates; magnitude is in
        absolute units (e.g. GBP 1000s if analysing research funding) whereas
        growth is expresed as a percentage
    """
    magnitude_df = magnitude(time_series, year_start, year_end)
    growth_df = smoothed_growth(time_series, year_start, year_end, window)
    combined_df = (
        pd.DataFrame([magnitude_df, growth_df], index=["magnitude", "growth"])
        .reset_index()
        .rename(columns={"index": "trend"})
    )
    return combined_df


def impute_empty_periods(
    df_time_period: pd.DataFrame,
    time_period_col: str,
    period: str,
    min_year: int,
    max_year: int,
) -> pd.DataFrame:
    """
    Imputes zero values for time periods without data
    Args:
        df_time_period: A dataframe with a column containing time period data
        time_period_col: Column containing time period data
        period: Time period that the data is grouped by, 'M', 'Q' or 'Y'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for
    Returns:
        A dataframe with imputed 0s for time periods with no data
    """
    max_year_data = np.nan_to_num(df_time_period[time_period_col].max().year)
    max_year = max(max_year_data, max_year)
    full_period_range = (
        pd.period_range(
            f"01/01/{min_year}",
            f"31/12/{max_year}",
            freq=period,
        )
        .to_timestamp()
        .to_frame(index=False, name=time_period_col)
        .reset_index(drop=True)
    )
    return full_period_range.merge(df_time_period, "left").fillna(0)


def minInstalls_coarse_partition(number: float):
    if number < 100e3:
        return "<100K"
    if (number >= 100e3) and (number < 1e6):
        return "100K-1M"
    if number > 1e6:
        return "1M+"


# %%
def result_dict_to_dataframe(
    result_dict: dict, sort_by: str = "counts", category_name: str = "cluster"
) -> pd.DataFrame:
    """Prepares the output dataframe"""
    return (
        pd.DataFrame(result_dict)
        .T.reset_index()
        .sort_values(sort_by)
        .rename(columns={"index": category_name})
    )


def get_category_time_series(
    time_series_df: pd.DataFrame,
    category_of_interest: str,
    time_column: str = "releaseYear",
    category_column: str = "cluster",
) -> dict:
    """Gets cluster or user-specific time series"""
    return (
        time_series_df.query(f"{category_column} == @category_of_interest")
        .drop(category_column, axis=1)
        .sort_values(time_column)
        .rename(columns={time_column: "year"})
        .assign(year=lambda x: pd.to_datetime(x.year.apply(lambda y: str(int(y)))))
        .pipe(
            impute_empty_periods,
            time_period_col="year",
            period="Y",
            min_year=2010,
            max_year=2021,
        )
        .assign(year=lambda x: x.year.dt.year)
    )


def get_estimates(
    time_series_df: pd.DataFrame,
    value_column: str = "counts",
    time_column: str = "releaseYear",
    category_column: str = "cluster",
    estimate_function=growth,
    year_start: int = 2019,
    year_end: int = 2020,
):
    """
    Get growth estimate for each category

    growth_estimate_function - either growth, smoothed_growth, or magnitude
    For growth, use 2019 and 2020 as year_start and year_end
    For smoothed_growth and magnitude, use 2017 and 2021
    """
    time_series_df_ = time_series_df[[time_column, category_column, value_column]]

    result_dict = {
        category: estimate_function(
            get_category_time_series(
                time_series_df_, category, time_column, category_column
            ),
            year_start=year_start,
            year_end=year_end,
        )
        for category in time_series_df[category_column].unique()
    }
    return result_dict_to_dataframe(result_dict, value_column, category_column)


def get_magnitude_vs_growth(
    time_series_df: pd.DataFrame,
    value_column: str = "counts",
    time_column: str = "releaseYear",
    category_column: str = "cluster",
):
    """Get magnitude vs growth esitmates"""
    df_growth = get_estimates(
        time_series_df,
        value_column=value_column,
        time_column=time_column,
        category_column=category_column,
        estimate_function=smoothed_growth,
        year_start=2017,
        year_end=2021,
    ).rename(columns={value_column: "Growth"})

    df_magnitude = get_estimates(
        time_series_df,
        value_column=value_column,
        time_column=time_column,
        category_column=category_column,
        estimate_function=magnitude,
        year_start=2017,
        year_end=2021,
    ).rename(columns={value_column: "Magnitude"})

    return df_growth.merge(df_magnitude, on="cluster")


def get_smoothed_timeseries(
    time_series_df: pd.DataFrame,
    value_column: str = "counts",
    time_column: str = "releaseYear",
    category_column: str = "cluster",
):
    """TODO: Finish this one!"""
    time_series_df_ = time_series_df[[time_column, category_column, value_column]]

    return pd.concat(
        [
            moving_average(
                get_category_time_series(
                    time_series_df_, category, time_column, category_column
                ),
                replace_columns=True,
            )
            for category in time_series_df[category_column].unique()
        ],
        ignore_index=True,
    )


# %%
app_install_categories = ["<100K", "100K-1M", "1M+"]

# %% [markdown]
# ### Trends: App development trends

# %% [markdown]
# #### App dev trends: "Baseline" sample of all playstore

# %%
google_playstore_df.minInstalls.median()

# %%
google_playstore_df.query('Category== "Parenting"').minInstalls.median()

# %%
google_playstore_df.query('Category== "Education"').minInstalls.median()

# %%
google_playstore_df.query('Category== "Educational"').minInstalls.median()

# %%
# Extract the year that each app was released and place in a separate column
# app_details["released"] = pd.to_datetime(app_details["released"])
app_details["releaseYear"] = pd.to_datetime(app_details["released"]).dt.year
app_details["user"] = app_details.cluster.apply(map_cluster_to_user)

# %%
new_apps_per_year_baseline = (
    google_playstore_df[-google_playstore_df.Released.isnull()]
    .assign(releaseYear=lambda x: x.Released.apply(lambda x: int(x[-4:])))
    # .query("Category == 'Education'")
    .groupby("releaseYear")
    .agg(counts=("appId", "count"))
    .reset_index()
    .query("releaseYear < 2021")
)

# %%
# sorted(google_playstore_df.Category.unique())

# %%
(
    alt.Chart(new_apps_per_year_baseline)
    .mark_line()
    .encode(
        x="releaseYear:O",
        y="counts",
    )
)

# %%
new_apps_per_year_baseline.pct_change()

# %%
new_apps_per_year_per_install_baseline = (
    google_playstore_df[-google_playstore_df.Released.isnull()]
    .assign(releaseYear=lambda x: x.Released.apply(lambda x: int(x[-4:])))
    .assign(minInstalls=lambda x: x.minInstalls.apply(minInstalls_coarse_partition))
    # .assign(minInstalls = lambda x: x.minInstalls.apply(lambda y: np.log10(y)))
    .groupby(["releaseYear", "minInstalls"])
    .agg(counts=("appId", "count"))
    .query("releaseYear < 2021")
    .reset_index()
)

# %%
(
    alt.Chart(new_apps_per_year_per_install_baseline)
    .mark_line()
    .encode(
        x="releaseYear",
        y="counts",
        color="minInstalls",
    )
)

# %%
baseline_pct_change = []
for cat in app_install_categories:
    baseline_pct_change.append(
        new_apps_per_year_per_install_baseline.query(f"minInstalls == '{cat}'")
        .rename(columns={"releaseYear": "year"})
        .sort_values("year")
        .drop("minInstalls", axis=1)
        .assign(pct_change=lambda x: (x.pct_change().counts) * 100)
        .assign(minInstalls=cat)
        .assign(sample="Play Store")
    )
baseline_pct_change = pd.concat(baseline_pct_change, ignore_index=True)

# %%
(
    alt.Chart(baseline_pct_change)
    .mark_line()
    .encode(
        x="year",
        y="pct_change",
        color="minInstalls",
    )
)

# %%
# Short term growth (baseline)
result_dict = {}
for i in app_install_categories:
    df_ = (
        new_apps_per_year_per_install_baseline.query(f"minInstalls == '{i}'")
        .sort_values("releaseYear")
        .rename(columns={"releaseYear": "year"})
        .drop("minInstalls", axis=1)
    )
    result_dict[i] = growth(df_, year_start=2019, year_end=2020)


# %%
result_dict

# %%
# Short term growth (baseline)
result_dict = {}
for i in app_install_categories:
    df_ = (
        new_apps_per_year_per_install_baseline.query(f"minInstalls == '{i}'")
        .sort_values("releaseYear")
        .rename(columns={"releaseYear": "year"})
        .drop("minInstalls", axis=1)
    )
    result_dict[i] = smoothed_growth(df_, year_start=2016, year_end=2020)


# %%
result_dict

# %% [markdown]
# #### App development trends: Big picture

# %%
new_apps_per_year_all = (
    app_details.groupby(["releaseYear"]).agg(counts=("appId", "count"))
    # .query('releaseYear < 2021')
    .reset_index()
)

(
    alt.Chart(new_apps_per_year_all)
    .mark_line()
    .encode(
        x="releaseYear:O",
        y="counts",
    )
)

# %%
get_magnitude_vs_growth(
    new_apps_per_year_all.assign(cluster="all"),
)

# %%
new_apps_per_year = (
    app_details.groupby(["releaseYear", "user"])
    .agg(counts=("appId", "count"))
    .reset_index()
)

(
    alt.Chart(new_apps_per_year)
    .mark_line()
    .encode(
        x="releaseYear",
        y="counts",
        color="user",
    )
)

# %%
new_apps_per_year = (
    app_details.assign(
        minInstalls=lambda x: x.minInstalls.apply(minInstalls_coarse_partition)
    )
    .groupby(["releaseYear", "minInstalls"])
    .agg(counts=("appId", "count"))
    .reset_index()
)

(
    alt.Chart(new_apps_per_year)
    .mark_line()
    .encode(
        x="releaseYear",
        y="counts",
        color="minInstalls",
    )
)

# %%
apps_pct_change = []
for cat in app_install_categories:
    apps_pct_change.append(
        new_apps_per_year.query(f"minInstalls == '{cat}'")
        .rename(columns={"releaseYear": "year"})
        .sort_values("year")
        .query("year < 2021")
        .drop("minInstalls", axis=1)
        .assign(pct_change=lambda x: (x.pct_change().counts) * 100)
        .assign(minInstalls=cat)
        .assign(sample="ours")
    )
apps_pct_change = pd.concat(apps_pct_change, ignore_index=True)

# %%
(
    alt.Chart(apps_pct_change)
    .mark_line()
    .encode(
        x="year",
        y="pct_change",
        color="minInstalls",
    )
)

# %%
pct_chage = pd.concat([baseline_pct_change, apps_pct_change], ignore_index=True).query(
    "minInstalls == '100K-1M'"
)

(
    alt.Chart(pct_chage)
    .mark_bar()
    .encode(
        # x='year:O',
        x="sample",
        y="pct_change",
        column="year",
        color="sample",
    )
)

# %%
# Short term growth (baseline)
result_dict = {}
for i in app_install_categories:
    df_ = (
        new_apps_per_year.query(f"minInstalls == '{i}'")
        .sort_values("releaseYear")
        .rename(columns={"releaseYear": "year"})
        .drop("minInstalls", axis=1)
    )
    result_dict[i] = growth(df_, year_start=2019, year_end=2020)


# %%
result_dict

# %%
# Short term growth (baseline)
result_dict = {}
for i in app_install_categories:
    df_ = (
        new_apps_per_year.query(f"minInstalls == '{i}'")
        .sort_values("releaseYear")
        .rename(columns={"releaseYear": "year"})
        .drop("minInstalls", axis=1)
    )
    result_dict[i] = smoothed_growth(df_, year_start=2016, year_end=2020)


# %%
result_dict

# %% [markdown]
# #### App dev trends: Cumulative size by user

# %%
user = "Children"
total_apps_per_yer_per_user = []
for user in ["Children", "Parents"]:
    total_apps_per_yer_per_user.append(
        app_details.groupby(["releaseYear", "user"])
        .agg(counts=("appId", "count"))
        .reset_index()
        .query("user == @user")
        .assign(counts_sum=lambda x: x.cumsum().counts)
    )
total_apps_per_yer_per_user = pd.concat(total_apps_per_yer_per_user, ignore_index=True)

# %%
(
    alt.Chart(total_apps_per_yer_per_user)
    .mark_line()
    .encode(
        x="releaseYear",
        y="counts_sum",
        color="user",
    )
)

# %% [markdown]
# #### App development trends: Clusters dynamics

# %%
# app_growth = app_details.groupby(["releaseYear", "cluster"], as_index=False).agg(
#     app_count=("appId", "count")
# )
# app_growth.sort_values(by=["releaseYear"], inplace=True, ignore_index=True)
# app_growth["growth"] = app_growth["app_count"].pct_change()
# # app_growth[app_growth.cluster == "Parental support"]
# # app_growth["growth"].mean()
# # app_growth.plot.bar(x="releaseYear", y="growth", figsize=(10, 7))

# %%
# app_growth["app_count"].sum()

# %%
# app_growth.groupby("cluster").agg(average_growth=("growth", "mean")).sort_values(
#     "average_growth"
# )

# %%
# New apps per year, for each cluster
new_apps_per_year_per_cluster = (
    app_details.groupby(["releaseYear", "cluster", "user"])
    .agg(counts=("appId", "count"))
    .reset_index()
)

# %%
fig = (
    alt.Chart(new_apps_per_year_per_cluster, width=700, height=700)
    .mark_line()
    .encode(
        x="releaseYear:O",
        y="counts:Q",
        color="cluster:N",
        tooltip=["cluster"],
    )
)
fig

# %%
# Short term trend (2019 -> 2020)
df_short = get_estimates(
    new_apps_per_year_per_cluster,
    value_column="counts",
    time_column="releaseYear",
    category_column="cluster",
    estimate_function=growth,
    year_start=2019,
    year_end=2020,
)
df_short

# %%
# Short term trend (2020 -> 2021)
df_short = get_estimates(
    new_apps_per_year_per_cluster,
    value_column="counts",
    time_column="releaseYear",
    category_column="cluster",
    estimate_function=growth,
    year_start=2020,
    year_end=2021,
)
df_short

# %%
# Longer term, smoothed trend (2017 -> 2021)
df_long = get_estimates(
    new_apps_per_year_per_cluster,
    value_column="counts",
    time_column="releaseYear",
    category_column="cluster",
    estimate_function=smoothed_growth,
    year_start=2017,
    year_end=2021,
)
df_long

# %%
new_apps_magnitude_vs_growth = get_magnitude_vs_growth(
    new_apps_per_year_per_cluster,
    value_column="counts",
    time_column="releaseYear",
    category_column="cluster",
)

# %%
fig = (
    alt.Chart(new_apps_magnitude_vs_growth, width=600, height=550)
    .mark_circle(size=50)
    .encode(
        x=alt.X(
            "Magnitude:Q",
            # axis=alt.Axis(title=f"Number of reviews in {end_year}"),
            # scale=alt.Scale(type="linear"),
        ),
        y=alt.Y(
            "Growth:Q",
            # axis=alt.Axis(
            #     title=f"Growth between {start_year} and {end_year} measured by number of reviews"
            # ),
            # scale=alt.Scale(type="linear"),
        ),
        # size="cluster_size:Q",
        color="cluster:N",
        tooltip="cluster:N",
    )
)

text = fig.mark_text(align="left", baseline="middle", dx=7).encode(text="cluster")

fig + text

# %% [markdown]
# #### App development trends: Cumulative cluster size

# %%
# Cumulative cluster size dynamic
cluster_sizes_per_year = []
for cluster in cluster_names:
    cluster_sizes_per_year.append(
        new_apps_per_year_per_cluster.query(f"cluster == @cluster").assign(
            counts_sum=lambda x: x.cumsum().counts
        )
    )
cluster_sizes_per_year = pd.concat(cluster_sizes_per_year, ignore_index=True)

# %%
fig = (
    alt.Chart(cluster_sizes_per_year, width=700, height=700)
    .mark_line()
    .encode(
        x="releaseYear:O",
        y="counts_sum:Q",
        color="cluster:N",
        tooltip=["cluster"],
    )
)
fig

# %%
get_growth_estimates(
    cluster_sizes_per_year,
    value_column="counts_sum",
    time_column="releaseYear",
    category_column="cluster",
    growth_estimate_function=smoothed_growth,
    year_start=2017,
    year_end=2021,
)

# %%
size_vs_growth = cluster_sizes_long_term.merge(
    app_details.groupby("cluster").agg(app_counts=("appId", "count")).reset_index()
)
size_vs_growth

# %%
# fig = (
#     alt.Chart(size_vs_growth, width=600, height=550)
#     .mark_circle()
#     .encode(
#         x=alt.X(
#             "app_counts:Q",
#             # axis=alt.Axis(title=f"Number of reviews in {end_year}"),
#             # scale=alt.Scale(type="linear"),
#         ),
#         y=alt.Y(
#             "counts_sum:Q",
#             # axis=alt.Axis(
#             #     title=f"Growth between {start_year} and {end_year} measured by number of reviews"
#             # ),
#             # scale=alt.Scale(type="linear"),
#         ),
#         # size="cluster_size:Q",
#         color="cluster:N",
#         tooltip="cluster:N",
#     )
# )
# fig

# %%
# cluster_size_by_year = (
#     app_details.groupby(by=["cluster", "releaseYear"])["appId"]
#     .count()
#     .cumsum()
#     .reset_index()
# )
# cluster_size_by_year.rename(columns={"appId": "app_count"}, inplace=True)
# cluster_size_by_year["normalised"] = cluster_size_by_year[
#     "app_count"
# ] / cluster_size_by_year.groupby("cluster")["app_count"].transform("max")
# cluster_size_by_year

# %%
# fig = (
#     alt.Chart(cluster_size_by_year, width=700, height=700)
#     .mark_line()
#     .encode(
#         x="releaseYear:N",
#         y="app_count:Q",
#         color="cluster:N",
#         tooltip=["cluster"],
#     )
# )
# fig

# %% [markdown]
# ### Review growth by years

# %%
app_reviews_dedup = app_reviews.drop_duplicates("reviewId")

# %%
yearly_app_reviews_all = app_reviews_dedup.groupby("reviewYear", as_index=False).agg(
    review_count=("reviewId", "count")
)

# %%
get_magnitude_vs_growth(
    yearly_app_reviews_all.assign(cluster="all"),
    value_column="review_count",
    time_column="reviewYear",
)

# %%
get_estimates(
    yearly_app_reviews_all.assign(cluster="all"),
    value_column="review_count",
    time_column="reviewYear",
    estimate_function=growth,
    year_start=2020,
    year_end=2021,
)

# %%
review_growth_by_user = (
    app_reviews_dedup[app_reviews_dedup.reviewYear < 2022]
    .assign(user=lambda x: x.cluster.apply(map_cluster_to_user))
    .groupby(["user", "reviewYear"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
)

# %%
fig = (
    alt.Chart(review_growth_by_user)
    .mark_line()
    .encode(
        x="reviewYear:O",
        y="review_count:Q",
        color="user:N",
    )
)
fig

# %%
review_growth_by_cluster = (
    app_reviews_dedup[app_reviews_dedup.reviewYear < 2022]
    .groupby(["cluster", "reviewYear"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
)

# %%
fig = (
    alt.Chart(review_growth_by_cluster, width=700, height=700)
    .mark_line()
    .encode(
        x="reviewYear:O",
        y="review_count:Q",
        color="cluster:N",
        tooltip=["cluster"],
    )
)
fig

# %%
# Short term trend (2019 -> 2020)
df_short = get_estimates(
    review_growth_by_cluster,
    value_column="review_count",
    time_column="reviewYear",
    category_column="cluster",
    estimate_function=growth,
    year_start=2019,
    year_end=2020,
)
df_short

# %%
# Short term trend (2020 -> 2021)
df_short = get_estimates(
    review_growth_by_cluster,
    value_column="review_count",
    time_column="reviewYear",
    category_column="cluster",
    estimate_function=growth,
    year_start=2020,
    year_end=2021,
)
df_short

# %%
reviews_magnitude_vs_growth = get_magnitude_vs_growth(
    review_growth_by_cluster,
    value_column="review_count",
    time_column="reviewYear",
    category_column="cluster",
)
reviews_magnitude_vs_growth

# %%
fig = (
    alt.Chart(reviews_magnitude_vs_growth, width=600, height=550)
    .mark_circle(size=50)
    .encode(
        x=alt.X(
            "Magnitude:Q",
            # axis=alt.Axis(title=f"Number of reviews in {end_year}"),
            # scale=alt.Scale(type="linear"),
        ),
        y=alt.Y(
            "Growth:Q",
            # axis=alt.Axis(
            #     title=f"Growth between {start_year} and {end_year} measured by number of reviews"
            # ),
            # scale=alt.Scale(type="linear"),
        ),
        # size="cluster_size:Q",
        color="cluster:N",
        tooltip="cluster:N",
    )
)

text = fig.mark_text(align="left", baseline="middle", dx=7).encode(text="cluster")

fig + text

# %%
review_growth_smooth = pd.concat(smoothed_dfs, ignore_index=True)

fig = (
    alt.Chart(review_growth_smooth, width=700, height=700)
    .mark_line()
    .encode(
        x="year:O",
        y="review_count:Q",
        color="cluster:N",
        tooltip=["cluster"],
    )
)
fig

# %%
# app_reviews.query("reviewYear == 2021")

# %% [markdown]
# ### Below: Mat's plots

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

# %%
