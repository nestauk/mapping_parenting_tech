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

# %%
app_reviews = utils.get_app_reviews()

# %%
sorted(app_details["cluster"].unique().tolist())

# %% [markdown]
# ## Helper functions

# %%
check_columns = ["title", "description", "summary", "installs", "score"]

# %% [markdown]
# ## Figures

# %% [markdown]
# #### Number of apps for children vs parents

# %%
# Check the total number of apps
n_total = len(app_details)
n_total

# %%
# Apps for children
n_children_apps = utils.percentage_in_cluster(
    app_details, utils.clusters_children, False
)
print(n_children_apps)
print(utils.percentage_in_cluster(app_details, utils.clusters_children))

# %%
# Apps for parents
n_parent_apps = utils.percentage_in_cluster(app_details, utils.clusters_parents, False)
print(n_parent_apps)
print(utils.percentage_in_cluster(app_details, utils.clusters_parents))

# %%
# Ratio children vs parents
n_children_apps / n_parent_apps

# %% [markdown]
# #### Number of apps per category

# %% [markdown]
# <!-- ### Popularity and scores
# How much are apps downloaded and do highly downloaded apps have better scores?
# The initial figures are derived from apps' details - i.e., their `minInstalls` count, and their score, all grouped by cluster and averaged accordingly. -->

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
users_to_plot = ["Parents"]
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
app_counts["Number of apps"].sum()

# %%
len(utils.get_relevant_apps())

# %%
len(app_details)

# %%
tooltip = [labels_title, values_title]
color = pu.NESTA_COLOURS[0]

chart_title = "Number of apps for parents"
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
            f"{values_title}:Q", title=values_title, scale=alt.Scale(domain=(0, 80))
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
table_name = "no_of_parent_apps"
utils.save_data_table(app_counts, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "png"])

# %% [markdown]
# #### Number of apps per release year

# %%
horizontal_title = "Release year"
values_title = "Number of apps"
colour_title = "User"

new_apps_per_year = (
    app_details.groupby(["releaseYear", "user"])
    .agg(counts=("appId", "count"))
    .reset_index()
    .assign(releaseYear=lambda x: x.releaseYear.astype(int))
    .rename(
        columns={
            "releaseYear": horizontal_title,
            "counts": values_title,
            "user": colour_title,
        }
    )
)

new_apps_per_year.head(3)

# %%
chart_title = "Apps for children and parents"
chart_subtitle = "Number of Play Store apps by release year"
tooltip = [colour_title, horizontal_title, values_title]

fig = (
    alt.Chart(
        new_apps_per_year,
        width=300,
        height=250,
    )
    .mark_bar(size=20)
    .encode(
        x=alt.X(f"{horizontal_title}:O"),
        y=alt.Y(f"sum({values_title})", title=values_title),
        color=f"{colour_title}",
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
def n_children_parents_ratio(new_apps_per_year, year):
    n_parents = new_apps_per_year.query(
        f"User == 'Parents' and `Release year` == {year}"
    )["Number of apps"].iloc[0]
    n_children = new_apps_per_year.query(
        f"User == 'Children' and `Release year` == {year}"
    )["Number of apps"].iloc[0]
    return n_children / n_parents


# %%
for year in range(2012, 2022):
    print(year, f"{n_children_parents_ratio(new_apps_per_year, year): .2f}")

# %%
importlib.reload(utils)
table_name = "no_of_apps_per_release_year"
utils.save_data_table(new_apps_per_year, table_name)

# %%
AltairSaver.save(fig, table_name, filetypes=["html", "png"])

# %% [markdown]
# #### Developers of apps

# %%
developer_counts = (
    app_details.groupby(["user", "developer"])
    .agg(counts=("appId", "count"))
    .reset_index()
    .sort_values(["user", "counts"], ascending=False)
)
developer_counts

# %% [markdown]
# - Number of distinct developers
# - Fraction of 'market' covered by top 5% developers

# %%
n_top = 10

# %%
np.round(
    (
        developer_counts.query("user == 'Children'")
        .sort_values("counts", ascending=False)
        .head(n_top)
        .counts.sum()
    )
    / n_children_apps,
    3,
)


# %%
np.round(
    (
        developer_counts.query("user == 'Parents'")
        .sort_values("counts", ascending=False)
        .head(n_top)
        .counts.sum()
    )
    / n_children_apps,
    3,
)


# %%
developing_for_both_users = set(
    developer_counts.query("user == 'Parents'").developer
).intersection(set(developer_counts.query("user == 'Children'").developer))


# %%
labels_title = "Developer name"
values_title = "Number of apps"
colour_title = "User"

top_developers = pd.concat(
    [
        developer_counts.query("user == 'Children'")
        .sort_values("counts", ascending=False)
        .head(5),
        developer_counts.query("user == 'Parents'")
        .sort_values("counts", ascending=False)
        .head(5),
    ],
    ignore_index=True,
).rename(
    columns={"user": colour_title, "developer": labels_title, "counts": values_title}
)

top_developers.head(5)

# %%
tooltip = [colour_title, labels_title, values_title]

chart_title = "Top 5 developers of children apps"
chart_subtitle = ""

# Figure 1
fig_1 = (
    alt.Chart(
        top_developers.query(f"`{colour_title}` == 'Children'"),
        width=300,
        height=300,
    )
    .mark_bar()
    .encode(
        x=alt.X(
            f"{values_title}:Q", title=values_title, scale=alt.Scale(domain=(0, 25))
        ),
        y=alt.Y(
            f"{labels_title}:N",
            title=labels_title,
            sort="-x",
            axis=alt.Axis(labelLimit=200),
        ),
        color=alt.Color(colour_title, legend=None),
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
)

# Figure 2
chart_title = "Top 5 developers of parent apps"
chart_subtitle = ""

fig_2 = (
    alt.Chart(
        top_developers.query(f"`{colour_title}` == 'Parents'"),
        width=300,
        height=300,
    )
    .mark_bar()
    .encode(
        x=alt.X(
            f"{values_title}:Q", title=values_title, scale=alt.Scale(domain=(0, 25))
        ),
        y=alt.Y(
            f"{labels_title}:N",
            title="",
            sort="-x",
            axis=alt.Axis(labelLimit=200),
        ),
        color=alt.Color(colour_title, legend=None),
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
)


fig = (
    (fig_1 | fig_2)
    .configure_axis(
        gridDash=[1, 7],
        gridColor="grey",
    )
    .configure_view(strokeWidth=0)
)
fig

# %%
importlib.reload(utils)
table_name = "top_developers"
utils.save_data_table(top_developers, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "png"])

# %% [markdown]
# #### Top-5 apps in each category

# %%
n_top = 5
top_apps = pd.concat(
    [
        utils.get_top_cluster_apps(app_details, cluster, sort_by="minInstalls", top_n=5)
        for cluster in utils.clusters_children
    ],
    ignore_index=True,
)


# %%
icon_url = "https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data_viz/app_icons"
cluster = utils.clusters_children[0]
t_df = (
    top_apps.query("cluster == @cluster")
    .copy()
    .assign(ordering=list(range(0, n_top * 2, 2)))
    .assign(x=0)
    .assign(
        icon_s3_url=lambda x: x.appId.apply(
            lambda y: f"{icon_url}/{utils.app_name_to_filename(y)}.png"
        )
    )
)

# %%
fig = (
    alt.Chart(
        t_df,
        width=100,
        height=450,
    )
    .mark_image(
        width=10,
        height=10,
    )
    .encode(
        x="x:O",
        y=alt.Y(
            "ordering:O",
            # title='App',
            # axis=alt.Axis(format='~s')
            # sort="-x",
            # axis=alt.Axis(labelLimit=200),
        ),
        url="icon",
    )
)
fig

# %%
table_name = "top_apps_per_category_children"
AltairSaver.save(fig, table_name, filetypes=["html", "png"])

# %%
fig = (
    alt.Chart(top_apps.query("cluster == @cluster"))
    .mark_bar()
    .encode(
        x="minInstalls:Q",
        y=alt.Y(
            "title:N",
            title="App name",
            # axis=alt.Axis(format='~s')
            sort="-x",
            axis=alt.Axis(labelLimit=200),
        ),
    )
)
fig

# %% [markdown]
# #### DfE apps (reference)

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
app_details.query("@dfe_apps in appId")[check_columns + ["cluster", "minInstalls"]]

# %%
# get_top_cluster_apps(app_details, 'Drawing and colouring', 'score', 100)

# %%
app_details.query("user == 'Children'")["minInstalls"].median()

# %%
df = (
    app_details.query("user == 'Children'")
    .copy()
    .assign(title_plot=lambda x: list(range(len(x))))
    .assign(DfE_endorsed=lambda x: x.appId.isin(dfe_apps))
)

# %%
df.query("DfE_endorsed == True")

# %%
labels_title = "title_plot"
values_title = "minInstalls"
colour_title = "DfE_endorsed"
tooltip = ["title", values_title]

color = pu.NESTA_COLOURS[0]

chart_title = "Number of apps for kids"
chart_subtitle = ""

fig = (
    alt.Chart(
        df,
        width=600,
        height=300,
    )
    .mark_bar()
    .encode(
        y=alt.Y(
            f"{values_title}:Q",
            title=values_title,  # scale=alt.Scale(domain=(0, 260))
            # scale=alt.Scale(type="log"),
        ),
        x=alt.X(
            f"{labels_title}:N",
            title=labels_title,
            sort="y",
            axis=alt.Axis(labelLimit=200),
        ),
        tooltip=tooltip,
        color=colour_title,
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
importlib.reload(utils)
utils.get_review_dates(app_reviews)

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
# review_dates.head()

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
