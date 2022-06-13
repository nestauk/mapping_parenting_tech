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


# %%
# Functionality for saving charts
import mapping_parenting_tech.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver()

# %% [markdown]
# ### Load app info

# %%
importlib.reload(psu)
importlib.reload(utils)

# %%
app_details = utils.get_app_details()

# %%
app_reviews = utils.get_app_reviews(test=False)

# %%
sorted(app_details["cluster"].unique().tolist())

# %%
len(utils.cluster_names)

# %%
len(app_details)

# %%
len(app_reviews.appId.unique())

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
importlib.reload(pu)

# %%
tooltip = [labels_title, values_title]
color = pu.NESTA_COLOURS[0]

chart_title = "Number of apps for children"
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
            axis=alt.Axis(labelLimit=300),
        ),
        tooltip=tooltip,
    )
)

fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
importlib.reload(utils)
table_name = "no_of_children_apps"
utils.save_data_table(app_counts, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])

# %%
users_to_plot = ["Parents"]
labels_title = "Category"
values_title = "Number of apps"

app_counts_parents = (
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
app_counts_parents

# %%
app_counts_parents["Number of apps"].sum()

# %%
tooltip = [labels_title, values_title]
color = pu.NESTA_COLOURS[0]

chart_title = "Number of apps for parents"
chart_subtitle = ""

fig = (
    alt.Chart(
        app_counts_parents,
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
)

fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
importlib.reload(utils)
table_name = "no_of_parent_apps"
utils.save_data_table(app_counts_parents, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])

# %%
users_to_plot = ["Children", "Parents"]
labels_title = "Category"
values_title = "Number of apps"
colour_title = "App user"

app_counts_all = (
    app_details.query(f"user in @users_to_plot")
    .groupby(["user", "cluster"], as_index=False)
    .agg(app_count=("appId", np.count_nonzero))
    .sort_values(["app_count"], ascending=False)
    .sort_values("user")
    .rename(
        columns={
            "cluster": labels_title,
            "app_count": values_title,
            "user": colour_title,
        }
    )
)

# %%
app_counts_all

# %%
app_counts_all["Number of apps"].sum()

# %%
tooltip = [labels_title, values_title, colour_title]
color = pu.NESTA_COLOURS[0]

chart_title = "Apps for toddlers and parents"
chart_subtitle = "Number of different types of apps in the UK Google Play Store"

fig = (
    alt.Chart(
        app_counts_all,
        width=300,
        height=400,
    )
    .mark_bar(color=color)
    .encode(
        x=alt.X(
            f"{values_title}:Q", title=values_title, scale=alt.Scale(domain=(0, 260))
        ),
        y=alt.Y(
            f"{labels_title}:N",
            title="",
            # sort="-x",
            sort=app_counts_all[labels_title].to_list(),
            axis=alt.Axis(labelLimit=250),
        ),
        color=colour_title,
        tooltip=tooltip,
    )
)

fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
alt_text = " ".join(
    [
        "This graph shows a bar chart with the number of different types of apps for toddlers and parents in the UK Google Play Store.",
        "The categories of apps aimed at children and the number of apps in each category are the following:",
        "; ".join(
            [
                f"{row['Category']}: {row['Number of apps']}"
                for i, row in app_counts_all.query(
                    "`App user` == 'Children'"
                ).iterrows()
            ]
        )
        + ".",
        "The categories of apps aimed at parents and the number of apps in each category are the following:",
        "; ".join(
            [
                f"{row['Category']}: {row['Number of apps']}"
                for i, row in app_counts_all.query("`App user` == 'Parents'").iterrows()
            ]
        )
        + ".",
    ]
)
alt_text

# %%
importlib.reload(utils)
table_name = "no_of_all_apps"
utils.save_data_table(app_counts_all, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])

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
importlib.reload(pu)

chart_title = "Apps for children and parents"
chart_subtitle = "Number of Play Store apps by release year"
tooltip = [colour_title, horizontal_title, values_title]

fig = (
    alt.Chart(
        new_apps_per_year,
        width=500,
        height=350,
    )
    .mark_bar(size=35)
    .encode(
        x=alt.X(f"{horizontal_title}:O"),
        y=alt.Y(f"sum({values_title}):Q", title=values_title),
        color=f"{colour_title}:N",
        tooltip=tooltip,
    )
)

fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
importlib.reload(utils)
table_name = "no_of_apps_per_release_year"
utils.save_data_table(new_apps_per_year, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])


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
# - Number of developers having more than one app

# %%
def prop_developers(df, user="Parents", more_than_n=1):
    return len(df.query("user == @user").query("counts > @more_than_n")) / len(
        df.query("user == @user")
    )


# %%
prop_developers(developer_counts, "Parents", more_than_n=1)

# %%
prop_developers(developer_counts, "Children", more_than_n=1)

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
from bs4 import BeautifulSoup


def clean_html(html):
    soup = BeautifulSoup(html)
    text = soup.get_text()
    return text


def shorten_text(text, l=250):
    return text[0:l] + "..."


def shorten_and_clean(html):
    return shorten_text(clean_html(html))


# %%
n_top = 5
top_apps = pd.concat(
    [
        utils.get_top_cluster_apps(
            app_details.sort_values("score", ascending=False),
            cluster,
            sort_by="minInstalls",
            top_n=5,
        )
        for cluster in utils.clusters_children
    ],
    ignore_index=True,
)


# %%
icon_url = "https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data_viz/app_icons"

labels_title = "Category"

t_dfs = []
for i, cluster in enumerate(utils.clusters_children):
    t_dfs.append(
        (
            top_apps.query("cluster == @cluster")
            .copy()
            .assign(ordering=list(range(0, n_top * 2, 2)))
            .assign(x=i)
            .assign(
                icon_s3_url=lambda x: x.appId.apply(
                    lambda y: f"{icon_url}/{utils.app_name_to_filename(y)}.png"
                )
            )
        )
    )
t_dfs = pd.concat(t_dfs, ignore_index=True).rename(columns={"cluster": labels_title})

# %%
t_dfs.minInstalls.min() / 1e6

# %%
t_dfs_ = t_dfs.assign(
    Description=lambda df: df.description.apply(shorten_and_clean),
    Installations=lambda df: df.minInstalls,
    Score=lambda df: df.score.apply(lambda x: np.round(x, 2)),
)
t_dfs_.loc[t_dfs_["Score"] == 0, "Score"] = "n/a"

# %%
chart_title = "Five most popular children apps in each category"
chart_subtitle = "Apps with the largest number of installations"

fig = (
    alt.Chart(
        t_dfs_,
        width=450,
        height=500,
    )
    .mark_image(
        width=40,
        height=40,
    )
    .encode(
        x=alt.X("ordering:Q", scale=alt.Scale(domain=(-1, 9)), axis=None),
        y=alt.Y(
            f"{labels_title}:N",
            sort=app_counts.Category.to_list(),
            axis=alt.Axis(labelLimit=300),
        ),
        url="icon",
        href="url",
        tooltip=["title", "Description", "Installations", "Score"],
    )
)

fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
top_app_dict = {}
for cat in t_dfs_.Category.unique():
    top_app_dict[cat] = (
        t_dfs_.query("Category == @cat")
        .sort_values("Installations", ascending=False)
        .iloc[0]
        .title
    )

# %%
alt_text = " ".join(
    [
        "This graph shows a bar chart with the number of different types of apps for toddlers and parents in the UK Google Play Store.",
        "The categories of apps aimed at children and the number of apps in each category are the following:",
        "; ".join(
            [
                f"{row['Category']}: {row['Number of apps']}"
                for i, row in app_counts_all.query(
                    "`App user` == 'Children'"
                ).iterrows()
            ]
        )
        + ".",
        "The categories of apps aimed at parents and the number of apps in each category are the following:",
        "; ".join(
            [
                f"{row['Category']}: {row['Number of apps']}"
                for i, row in app_counts_all.query("`App user` == 'Parents'").iterrows()
            ]
        )
        + ".",
    ]
)
alt_text

# %%
alt_text = "; ".join([f"{cat}: {top_app_dict[cat]}" for cat in top_app_dict])
alt_text

# %%
table_name = "top_apps_per_category_children"
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])

# %%
df = t_dfs_[
    [
        "ordering",
        labels_title,
        "icon",
        "url",
        "title",
        "Description",
        "Installations",
        "Score",
    ]
]
utils.save_data_table(df, table_name)

# %%
chart_title = "Five most popular parenting apps in each category"
chart_subtitle = "Apps with the largest number of installations"
url = "https://raw.githubusercontent.com/beingkk/test/main/top_apps_per_category_children.csv"
fig = (
    alt.Chart(
        url,
        width=450,
        height=500,
    )
    .mark_image(
        width=40,
        height=40,
    )
    .encode(
        x=alt.X(
            "ordering:Q",
            scale=alt.Scale(domain=(-1, 9)),
            axis=None,
        ),
        y=alt.Y(
            f"{labels_title}:N",
            sort=app_counts.Category.to_list(),
            axis=alt.Axis(labelLimit=200),
        ),
        url="icon:N",
        href="url:N",
        tooltip=["title:N", "Description:N", "Installations:Q", "Score:N"],
    )
)

fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
table_name = "top_apps_per_category_children_url"
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])

# %%
n_top = 5
top_apps = pd.concat(
    [
        utils.get_top_cluster_apps(
            app_details.sort_values("score", ascending=False),
            cluster,
            sort_by="minInstalls",
            top_n=5,
        )
        for cluster in utils.clusters_parents
    ],
    ignore_index=True,
)


# %%
icon_url = "https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data_viz/app_icons"

labels_title = "Category"

t_dfs = []
for i, cluster in enumerate(utils.clusters_parents):
    t_dfs.append(
        (
            top_apps.query("cluster == @cluster")
            .copy()
            .assign(ordering=list(range(0, n_top * 2, 2)))
            .assign(x=i)
            .assign(
                icon_s3_url=lambda x: x.appId.apply(
                    lambda y: f"{icon_url}/{utils.app_name_to_filename(y)}.png"
                )
            )
        )
    )
t_dfs = pd.concat(t_dfs, ignore_index=True).rename(columns={"cluster": labels_title})

# %%
t_dfs.minInstalls.min() / 1e6

# %%
t_dfs_ = t_dfs.assign(
    Description=lambda df: df.description.apply(shorten_and_clean),
    Installations=lambda df: df.minInstalls,
    Score=lambda df: df.score.apply(lambda x: np.round(x, 2)),
)
t_dfs_.loc[t_dfs_["Score"] == 0, "Score"] = "n/a"

# %%
chart_title = "Five most popular parenting apps in each category"
chart_subtitle = "Apps with the largest number of installations"

fig = (
    alt.Chart(
        t_dfs_,
        width=450,
        height=500,
    )
    .mark_image(
        width=40,
        height=40,
    )
    .encode(
        x=alt.X(
            "ordering:Q",
            scale=alt.Scale(domain=(-1, 9)),
            axis=None,
        ),
        y=alt.Y(
            f"{labels_title}:N",
            sort=app_counts_parents.Category.to_list(),
            axis=alt.Axis(labelLimit=200),
        ),
        url="icon",
        href="url",
        tooltip=["title", "Description", "Installations", "Score"],
    )
)

fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
table_name = "top_apps_per_category_parents"
AltairSaver.save(fig, table_name, filetypes=["png", "svg", "html"])

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
app_details.query("user == 'Children'")["minInstalls"].median()

# %%
labels_title = "Installations"
values_title = "Number of apps"

importlib.reload(utils)
app_counts_by_installs = (
    app_details.query("user == 'Children'")
    .assign(install_range=lambda df: df.minInstalls.apply(utils.install_labels_range))
    .groupby("install_range")
    .agg(counts=("appId", "count"))
    .reset_index()
    .rename(columns={"counts": values_title, "install_range": labels_title})
)

app_counts_by_installs

# %%
dfe_apps_df = (
    app_details.query("@dfe_apps in appId")[
        check_columns + ["cluster", "minInstalls", "icon"]
    ]
    .assign(install_range=lambda df: df.minInstalls.apply(utils.install_labels_range))
    .assign(values_title=[330, 200, 50, 100, 310, 100])
    .rename(
        columns={
            "install_range": labels_title,
            "values_title": values_title,
        }
    )
)
dfe_apps_df

# %%
alt_text = "; ".join(
    [
        f"{row.title} ({row.installs} installations, score {row.score:.2f})"
        for i, row in dfe_apps_df.iterrows()
    ]
)
alt_text

# %%
dfe_fig = (
    alt.Chart(
        dfe_apps_df,
        width=100,
        height=450,
    )
    .mark_image(
        width=35,
        height=35,
    )
    .encode(
        x=alt.Y(f"{labels_title}:O", sort=utils.app_install_ranges),
        y=alt.Y(
            f"{values_title}:Q",
            # title='App',
            # axis=alt.Axis(format='~s')
            # sort="-x",
            # axis=alt.Axis(labelLimit=200),
        ),
        url="icon",
        tooltip=["title", "description", "score"],
    )
)
dfe_fig

# %%
tooltip = [labels_title, values_title]
color = pu.NESTA_COLOURS[0]

chart_title = "Distribution of of app installations across our sample"
chart_subtitle = "Icons show the six apps endorsed by the Department for Education"

fig = (
    alt.Chart(
        app_counts_by_installs,
        width=450,
        height=330,
    )
    .mark_bar(color=color)
    .encode(
        y=alt.Y(
            f"{values_title}:Q", title=values_title, scale=alt.Scale(domain=(0, 300))
        ),
        x=alt.X(
            f"{labels_title}:N",
            # title=labels_title,
            title="Number of installations per app",
            sort=utils.app_install_ranges,
            axis=alt.Axis(labelLimit=200, labelAngle=0),
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
)

final_fig = pu.configure_plots(
    # (fig + dfe_fig)
    fig,
    "Crowded marketplace for children apps",
    "One third a large number of installations",
)

final_fig

# %%
len(app_details.query('user == "Children" and minInstalls >= 1_000_000'))

# %%
# importlib.reload(utils)
table_name = "children_app_installations"
utils.save_data_table(app_counts_by_installs, table_name)
AltairSaver.save(final_fig, table_name, filetypes=["html", "png", "svg"])

# %% [markdown]
# ##### DfE apps, take 2

# %%
dfe_apps_df = (
    app_details.query("@dfe_apps in appId")[
        check_columns + ["appId", "cluster", "minInstalls", "icon"]
    ]
    .assign(install_range=lambda df: df.minInstalls.apply(utils.install_labels_range))
    .assign(y=np.array([60, 40, 10, 30, 50, 20]) * 2)
    .assign(x=1)
    .assign(x_text=1.5)
    .assign(y_title_text=lambda df: df.y + 10)
    .rename(
        columns={
            "install_range": labels_title,
            "values_title": values_title,
        }
    )
)

# %%
dfe_apps_df[["appId", "minInstalls"]]

# %%
# dfe_apps_df

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
dfe_fig = (
    alt.Chart(
        dfe_apps_df,
        # width=100,
        # height=450,
    )
    .mark_image(
        width=50,
        height=50,
    )
    .encode(
        # x=alt.Y(
        #     f'x:O',
        #     sort=utils.app_install_ranges,
        #     axis=None
        # ),
        y=alt.Y(
            f"y:Q",
            # axis=None,
            axis=alt.Axis(domainOpacity=0, ticks=False)
            # title='App',
            # axis=alt.Axis(format='~s')
            # sort="-x",
            # axis=alt.Axis(labelLimit=200),
        ),
        url="icon",
        tooltip=["title", "description", "score", "y", "appId", "minInstalls"],
    )
)

# flags = alt.Chart(df).mark_image().encode(
# y=alt.Y('country', axis=alt.Axis(domainOpacity=0, ticks=False)),
# url="flag_image"
# )

annotation_text = (
    alt.Chart(dfe_apps_df)
    .mark_text(font=pu.FONT, fontSize=14, align="left", fontStyle="bold", opacity=0.75)
    .encode(
        # x=alt.X('x_text:Q'),
        y=alt.Y("y_title_text:Q", axis=None),
        text=alt.Text("title"),
    )
)

# final_fig = pu.configure_plots(
#     (dfe_fig + annotation_text),
#     "Apps endorsed by the Department for Education",
#     "",
# )
# final_fig
alt.concat(dfe_fig, annotation_text)

# %% [markdown]
# #### Ads and in-app purchases

# %%
(
    app_details.groupby("user").agg(
        free=("free", "mean"),
        containsAds=("containsAds", "mean"),
        offersIAP=("offersIAP", "mean"),
    )
)

# %%
apps_access = (
    app_details.groupby(["user", "cluster"])
    .agg(
        cluster_size=("cluster", "count"),
        free=("free", "sum"),
        IAPs=("offersIAP", "sum"),
        ads=("containsAds", "sum"),
    )
    .assign(
        free_pc=lambda x: x["free"] / x["cluster_size"],
        ads_pc=lambda x: x["ads"] / x["cluster_size"],
        IAPs_pc=lambda x: x["IAPs"] / x["cluster_size"],
    )
    .reset_index()
    .rename(
        columns={
            "user": "User",
            "cluster": "Category",
            "free_pc": "Free",
            "ads_pc": "Ads",
            "IAPs_pc": "In-app purchases",
        }
    )
)

apps_access

# %%
apps_access_plot = (
    pd.melt(
        apps_access.query("User =='Children'"),
        id_vars=["User", "Category"],
        value_vars=["In-app purchases", "Ads", "Free"],
    )
    .rename(
        columns={
            "variable": "Free, ads or purchases",
            "value": "Percentage",
        }
    )
    .assign(variable=lambda x: x["Free, ads or purchases"])
)

# %%
sort_order = (
    apps_access_plot.query("`Free, ads or purchases`=='Free'")
    .sort_values("Percentage")
    .Category.to_list()
)

# %%
# chart_title = "Percentage of apps that are free, feature ads or have in-app purchases"
# chart_subtitle = "While most apps are free, the majority of apps have in-app purchases"
chart_title = ""
chart_subtitle = ""

fig = (
    alt.Chart(
        apps_access_plot.sort_values("Percentage", ascending=False),
        width=250,
    )
    # .transform_calculate(key="datum.variable == 'Free'")
    # .transform_joinaggregate(sort_key="argmax(key)", groupby=["Category"])
    # .transform_calculate(sort_val="datum.sort_key.value")
    .mark_bar().encode(
        row=alt.Row(
            "Category",
            header=alt.Header(
                labelAngle=0,
                labelAlign="left",
                labelFontSize=pu.FONTSIZE_NORMAL,
                labelLimit=163,
            ),
            sort=sort_order
            # sort=alt.SortField("sort_val", order="ascending"),
        ),
        y=alt.Y(
            "Free, ads or purchases",
            axis=alt.Axis(ticks=False, labels=False, title=""),
            sort=["Free", "In-app purchases"],
        ),
        x=alt.X(
            "Percentage",
            axis=alt.Axis(grid=True, format="%", labelAlign="center"),
            title="",
        ),
        color=alt.Color(
            "Free, ads or purchases",
            sort=["Free", "In-app purchases"],
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            "Category",
            "Free, ads or purchases",
            alt.Tooltip("Percentage:Q", title="Percentage", format=".0%"),
        ],
    )
)

fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
# importlib.reload(utils)
table_name = "ads_iap_free_children_apps"
utils.save_data_table(apps_access_plot.drop("variable", axis=1), table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])

# %%
apps_access_plot.head(2)

# %%
p = []
for cat in apps_access_plot.Category.unique():
    p.append(
        cat
        + ": "
        + "; ".join(
            [
                f'{row.Percentage*100:.0f}% {row["Free, ads or purchases"]}'
                for i, row in apps_access_plot.query("Category == @cat")
                .sort_values("Free, ads or purchases")
                .iterrows()
            ]
        )
    )
". ".join(p)

# %%
apps_access.sort_values(["User", "Free"], ascending=False)

# %%
app_details.query("user == 'Children'").free.mean()

# %% [markdown]
# #### Reviews

# %%
len(app_reviews.drop_duplicates("reviewId"))

# %%
horizontal_title = "Year"
values_title = "Number of reviews"
colour_title = "User"

reviews_per_year_by_user = (
    app_reviews.query("reviewYear <= 2022")
    .groupby(["user", "reviewYear"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
    .rename(
        columns={
            "reviewYear": horizontal_title,
            "review_count": values_title,
            "user": colour_title,
        }
    )
)

# %%
reviews_per_year_by_user.head(2)

# %%
tooltip = [colour_title, horizontal_title, values_title]
chart_title = "Number of yearly reviews on Play Store"
chart_subtitle = ""

fig = (
    alt.Chart(
        reviews_per_year_by_user.query("Year >= 2010"),
        width=300,
        height=200,
    )
    .mark_line(size=3)
    .encode(
        x=alt.X(
            f"{horizontal_title}:O",
            scale=alt.Scale(domain=list(range(2010, 2023, 1))),
        ),
        y=alt.Y(f"{values_title}:Q", scale=alt.Scale(domain=(0, 300_000))),
        color=f"{colour_title}:N",
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
    .configure_view(stroke=None, strokeWidth=0)
)
fig

# %%
importlib.reload(utils)
table_name = "no_of_reviews_by_user"
utils.save_data_table(reviews_per_year_by_user, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "png"])

# %%
utils.percentage_change(
    reviews_per_year_by_user.query("User == 'Children' and Year == 2019")[
        "Number of reviews"
    ].iloc[0],
    reviews_per_year_by_user.query("User == 'Children' and Year == 2020")[
        "Number of reviews"
    ].iloc[0],
)


# %%
def transform_dates(date):
    return "\n".join(date.split())


# %%
horizontal_title = "Date"
values_title = "Number of reviews"
colour_title = "User"
years = [2019, 2020, 2021, 2022]

review_per_month_by_user = (
    app_reviews.query("reviewYear in @years")
    .assign(year_month=lambda df: pd.to_datetime(df["at"]).dt.to_period("M"))
    # .astype({'year_month': str})
    .groupby(["user", "year_month"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
    .assign(year_month_=lambda df: df["year_month"].dt.strftime("%b %Y"))
    .astype({"year_month_": str})
    # .assign(year_month_ = lambda df: df.year_month_.apply(transform_dates))
    .rename(
        columns={
            "year_month_": horizontal_title,
            "review_count": values_title,
            "user": colour_title,
        }
    )
    .drop("year_month", axis=1)
)
review_per_month_by_user = review_per_month_by_user[
    review_per_month_by_user.Date != "May 2022"
]

# %%
review_per_month_by_user.tail(2)

# %%
label_expression = ""
for i, row in review_per_month_by_user.iterrows():
    start = " : " if i != 0 else ""
    if (i % 3) == 0:
        if row.Date[0:3] == "Jan":
            label_expression += (
                f"{start}datum.value == '{row.Date}' ? split(datum.value, ' ')"
            )
        else:
            label_expression += (
                f"{start}datum.value == '{row.Date}' ? '{row.Date[0:3]}'"
            )
    else:
        label_expression += f"{start}datum.value == '{row.Date}' ? null"
label_expression += " : 'dunno'"
# "datum.label == 0 ? 'Poor' : datum.label == 5 ? 'Neutral' : 'Great'"

# %%
tooltip = [colour_title, horizontal_title, values_title]
chart_title = "Number of monthly app reviews on Play Store"
chart_subtitle = ""

fig = (
    alt.Chart(
        review_per_month_by_user,
        width=540,
        height=300,
    )
    .mark_line(size=3)
    .encode(
        x=alt.X(
            f"{horizontal_title}:O",
            # 'year_month_:O',
            # scale = alt.Scale(domain=list(range(2010, 2022,1))),
            axis=alt.Axis(
                tickCount=3,
                labelAngle=-0,
                labelAlign="center",
                labelExpr=label_expression,
                title="",
            ),
            sort=review_per_month_by_user.Date.to_list(),
        ),
        y=alt.Y(f"{values_title}:Q", scale=alt.Scale(domain=(0, 35_000))),
        color=f"{colour_title}:N",
        tooltip=tooltip,
    )
)

fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
importlib.reload(utils)
table_name = "no_of_reviews_per_month_by_user"
utils.save_data_table(reviews_per_year_by_user, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])

# %% [markdown]
# ##### Review counts by app Id

# %%
horizontal_title = "Month"
values_title = "Number of reviews"
colour_title = "App"
years = [2019, 2020, 2021]

review_per_month_by_appId = (
    app_reviews.query("reviewYear in @years")
    .query("user == 'Children'")
    .assign(year_month=lambda df: pd.to_datetime(df["at"]).dt.to_period("M"))
    .assign(year=lambda df: pd.to_datetime(df["at"]).dt.year)
    .astype({"year_month": str})
    .groupby(["appId", "year"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
    .rename(
        columns={
            "year_month": horizontal_title,
            "review_count": values_title,
            "appId": colour_title,
        }
    )
)

# %%
# tooltip = [colour_title, horizontal_title, values_title]
# chart_title = "Number of monthly reviews on Play Store, 2019-2021"
# chart_subtitle = ""

# fig = (
#     alt.Chart(
#         review_per_month_by_appId,
#         width=800,
#         height=200,
#     )
#     .mark_line(size=3)
#     .encode(
#         x=alt.X(
#             f"{horizontal_title}:O",
#             # scale = alt.Scale(domain=list(range(2010, 2022,1))),
#         ),
#         y=alt.Y(
#             f"{values_title}:Q",
#             # scale = alt.Scale(domain=(0, 300_000))
#         ),
#         color=f"{colour_title}:N",
#         tooltip=tooltip,
#     )
#     .properties(
#         title={
#             "anchor": "start",
#             "text": chart_title,
#             "subtitle": chart_subtitle,
#             "subtitleFont": pu.FONT,
#         },
#     )
#     .configure_axis(
#         gridDash=[1, 7],
#         gridColor="grey",
#     )
#     .configure_view(stroke=None, strokeWidth=0)
#     .interactive()
# )
# fig

# %%
horizontal_title = "Month"
values_title = "Number of reviews"
colour_title = "Developer"
years = [2019, 2020, 2021]

review_per_month_by_developer = (
    app_reviews.query("reviewYear in @years")
    .query("user == 'Children'")
    .merge(app_details[["appId", "developer"]], how="left", on="appId")
    .assign(year_month=lambda df: pd.to_datetime(df["at"]).dt.to_period("M"))
    .astype({"year_month": str})
    .groupby(["developer", "year_month"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
    .rename(
        columns={
            "year_month": horizontal_title,
            "review_count": values_title,
            "developer": colour_title,
        }
    )
)

# %%
tooltip = [colour_title, horizontal_title, values_title]
chart_title = "Number of monthly reviews on Play Store, 2019-2021"
chart_subtitle = ""

fig = (
    alt.Chart(
        review_per_month_by_developer,
        width=800,
        height=200,
    )
    .mark_line(size=3)
    .encode(
        x=alt.X(
            f"{horizontal_title}:O",
            # scale = alt.Scale(domain=list(range(2010, 2022,1))),
        ),
        y=alt.Y(
            f"{values_title}:Q",
            # scale = alt.Scale(domain=(0, 300_000))
        ),
        color=f"{colour_title}:N",
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
    .configure_view(stroke=None, strokeWidth=0)
    .interactive()
)
fig

# %%
app_details.to_csv(PROJECT_DIR / "outputs/data/finals/app_details.csv", index=False)

# %% [markdown]
# ##### Inspect review content

# %%
from toolz import pipe
import mapping_parenting_tech.utils.text_preprocessing_utils as tpu
import nltk
import mapping_parenting_tech.analysis.cluster_analysis_utils as cau
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#
def remove_stopwords(txt):
    return [t for t in txt if t not in nltk.corpus.stopwords.words("English")]


def process_review_text(txt):
    """'this is text' -> [''text']"""
    return pipe(
        txt,
        tpu.lowercase,
        tpu.remove_non_alphanumeric,
        # tpu.simple_tokenizer,
        # remove_stopwords
    )


# %%
process_review_text(app_reviews_child.content.iloc[2000])

# %%
app_reviews_child = (
    app_reviews[-app_reviews.content.isnull()]  # .sample(50000)
    .query("user == 'Children'")
    # .query("user == 'Parents'")
    .query("reviewYear > 2018 and reviewYear < 2022")
    .assign(year_month=lambda df: pd.to_datetime(df["at"]).dt.to_period("M"))
    .assign(review_len=lambda df: df.content.apply(len))
    .assign(content_processed=lambda df: df.content.apply(process_review_text))
)

# %% [markdown]
# - Divide by year-month
# - Check average review length across year-month (any oddities?)
# - Simple process (lower-string, only alphanumeric), tokenise and count vectorizer / tf-idf for each year-month
# - Inspect top 50 terms for each year-month

# %%
reviews_general_stats = app_reviews_child.groupby("year_month").agg(
    mean_len=("review_len", "mean"),
    med_len=("review_len", np.median),
    counts=("reviewId", "count"),
)
reviews_general_stats

# %%
len(app_reviews_child)

# %%
term = "school"
(
    app_reviews_child[app_reviews_child.content_processed.str.contains(term)]
    .groupby("year_month")
    .agg(term_counts=("appId", "count"))
    .assign(prop=lambda df: df.term_counts / reviews_general_stats.counts)
)

# %%
top_words = cau.cluster_keywords(
    documents=app_reviews_child.content_processed.to_list(),
    cluster_labels=app_reviews_child.year_month.astype(str).to_list(),
    n=100,
    tokenizer=lambda x: x,
    Vectorizer=CountVectorizer,
)

# %%
sorted_keys = sorted(app_reviews_child.year_month.astype(str).unique())
top_words = {i: top_words[i] for i in sorted_keys}
top_words_df = pd.DataFrame(top_words)

# %%
tfidf_words = cau.cluster_keywords(
    documents=app_reviews_child.content_processed.to_list(),
    cluster_labels=app_reviews_child.year_month.astype(str).to_list(),
    n=100,
    tokenizer=lambda x: x,
    max_df=0.90,
    min_df=0.01,
    Vectorizer=TfidfVectorizer,
)
tfidf_words = {i: tfidf_words[i] for i in sorted_keys}
tfidf_words_df = pd.DataFrame(tfidf_words)

# %%
top_words_df.to_csv(
    PROJECT_DIR / "outputs/data/review_analysis/top_words.csv", index=False
)
tfidf_words_df.to_csv(
    PROJECT_DIR / "outputs/data/review_analysis/tfidf_words.csv", index=False
)

# %% [markdown]
# ##### Review growth 2019 > 2020 for children apps

# %%
# len(app_reviews)

# %%
labels_title = "Category"
values_title = "Number of reviews (thousands)"
colour_title = "Year"
years = [2019, 2020]

reviews_per_year_by_cluster = (
    app_reviews.query("reviewYear < 2022")
    .groupby(["user", "cluster", "reviewYear"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
    .assign(review_count=lambda df: df.review_count / 1000)
    .rename(
        columns={
            "cluster": labels_title,
            "review_count": values_title,
            "reviewYear": colour_title,
            "user": "User",
        }
    )
)

# %%
sort_order = (
    reviews_per_year_by_cluster.query("Year == 2020").sort_values(
        values_title, ascending=False
    )
)[labels_title].to_list()

# %%
reviews_per_year_by_cluster.query("Category == @cat and Year == 2019")[
    "Number of reviews (thousands)"
].iloc[0]

# %%
al_text = "; ".join(
    [
        f'{cat}: {reviews_per_year_by_cluster.query("Category == @cat and Year == 2019")["Number of reviews (thousands)"].iloc[0]*1000:.0f} and {reviews_per_year_by_cluster.query("Category == @cat and Year == 2020")["Number of reviews (thousands)"].iloc[0]*1000:.0f}'
        for cat in reviews_per_year_by_cluster.Category.unique()
    ]
)
al_text


# %%
# Short term trend (2019 -> 2020)
growth_title = "Growth"

review_growth_yoy = (
    utils.get_estimates(
        reviews_per_year_by_cluster.query("User == 'Children'"),
        value_column=values_title,
        time_column=colour_title,
        category_column=labels_title,
        estimate_function=utils.growth,
        year_start=2019,
        year_end=2020,
    )
    .sort_values(values_title, ascending=False)
    .assign(values_title=lambda df: df[values_title] / 100)
    .rename(columns={"values_title": growth_title})
    .drop(values_title, axis=1)
)
review_growth_yoy

# %%
tooltip = [labels_title, colour_title, values_title]
# chart_title = "Number of reviews on Play Store in 2019 and 2020"
chart_title = ""
chart_subtitle = ""

fig = (
    alt.Chart(
        reviews_per_year_by_cluster.query("Year in @years and User == 'Children'"),
        width=200,
        height=250,
    )
    .mark_point(filled=False, size=50)
    .encode(
        alt.X(
            f"{values_title}:Q",
            title=values_title,
            # scale=alt.Scale(zero=False, domain=(0, 140)),
            axis=alt.Axis(grid=False, labelAlign="center", tickCount=5),
        ),
        alt.Y(
            f"{labels_title}:N",
            title=labels_title,
            sort=sort_order,
            axis=alt.Axis(grid=True),
        ),
        color=alt.Color(
            f"{colour_title}:N",
            scale=alt.Scale(
                domain=[2019, 2020], range=[pu.NESTA_COLOURS[2], pu.NESTA_COLOURS[3]]
            ),
            legend=alt.Legend(title=colour_title, titleAnchor="middle", orient="top"),
        ),
        tooltip=tooltip,
    )
    .properties(
        title={
            "anchor": "start",
            "text": chart_title,
            "subtitle": chart_subtitle,
            "subtitleFont": pu.FONT,
            "fontSize": 15,
        },
    )
    .configure_axis(
        gridDash=[1, 7],
        gridColor="grey",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_legend(
        titleFontSize=pu.FONTSIZE_NORMAL,
        labelFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_view(strokeWidth=0)
)

fig

# %%
importlib.reload(utils)
table_name = "children_app_reviews_2019_2020"
utils.save_data_table(reviews_per_year_by_cluster, f"{table_name}")
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])

# %%
tooltip = [labels_title, growth_title]
color = pu.NESTA_COLOURS[0]

chart_title = "Growth between 2019 and 2020"
chart_subtitle = ""

fig_growth = (
    alt.Chart(
        review_growth_yoy,
        width=200,
        height=250,
    )
    .mark_bar(color=color)
    .encode(
        x=alt.X(
            f"{growth_title}:Q",
            title=growth_title,
            scale=alt.Scale(domain=(0, 4)),
            axis=alt.Axis(format="%", labelAlign="center", tickCount=2),
        ),
        y=alt.Y(
            f"{labels_title}:N",
            title="Category",
            sort=sort_order,
            # axis=alt.Axis(labelLimit=300, labels=False),
        ),
        tooltip=[
            labels_title,
            # 'Free, ads or purchases',
            alt.Tooltip(f"{growth_title}:Q", title="Growth", format=".0%"),
        ],
    )
    .properties(
        title={
            "anchor": "start",
            "text": "",
            "subtitle": chart_subtitle,
            "subtitleFont": pu.FONT,
            "fontSize": 15,
        },
    )
    .configure_axis(
        gridDash=[1, 7],
        gridColor="grey",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_legend(
        titleFontSize=pu.FONTSIZE_NORMAL,
        labelFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_view(strokeWidth=0)
)

fig_growth

# %%
importlib.reload(utils)
table_name = "children_app_growth_2019_2020"
utils.save_data_table(review_growth_yoy, f"{table_name}")
AltairSaver.save(fig_growth, table_name, filetypes=["html", "svg", "png"])

# %%
"; ".join(
    [
        f'{cat}: {review_growth_yoy.query("Category == @cat").Growth.iloc[0]*100:.0f}%'
        for cat in review_growth_yoy.Category.unique()
    ]
)

# %%
fig_final = (
    (fig | fig_growth)
    .configure_axis(
        gridDash=[1, 7],
        gridColor="grey",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_legend(
        titleFontSize=pu.FONTSIZE_NORMAL,
        labelFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_view(strokeWidth=0)
)
fig_final

# %%
importlib.reload(utils)
table_name = "children_app_growth_2019_2020"
utils.save_data_table(reviews_per_year_by_cluster, f"{table_name}_2019_2020")
utils.save_data_table(review_growth_yoy, f"{table_name}_growth")

AltairSaver.save(fig_final, table_name, filetypes=["html", "svg", "png"])

# %% [markdown]
# ##### Check most popular words for each year

# %%
df = app_reviews.groupby(["reviewYear", "appId"]).count().reset_index()

# %%
# df.query('reviewYear==2021')

# %% [markdown]
# ##### Reviews for parent apps

# %%
utils.percentage_change(
    reviews_per_year_by_user.query("User == 'Parents' and Year == 2019")[
        "Number of reviews"
    ].iloc[0],
    reviews_per_year_by_user.query("User == 'Parents' and Year == 2020")[
        "Number of reviews"
    ].iloc[0],
)

# %%
horizontal_title = "Year"
values_title = "Number of reviews"
colour_title = "Category"

reviews_parent_apps_per_year_by_category = (
    app_reviews.query("reviewYear < 2022")
    .query("user == 'Parents'")
    .groupby(["cluster", "reviewYear"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
    .rename(
        columns={
            "reviewYear": horizontal_title,
            "review_count": values_title,
            "cluster": colour_title,
        }
    )
)

# %%
tooltip = [colour_title, horizontal_title, values_title]
chart_title = "Number of yearly reviews on Play Store"
chart_subtitle = ""

fig = (
    alt.Chart(
        reviews_parent_apps_per_year_by_category.query("Year >= 2010"),
        width=300,
        height=200,
    )
    .mark_line(size=3)
    .encode(
        x=alt.X(
            f"{horizontal_title}:O",
            scale=alt.Scale(domain=list(range(2010, 2022, 1))),
        ),
        y=alt.Y(f"{values_title}:Q", scale=alt.Scale(domain=(0, 100_000))),
        color=f"{colour_title}:N",
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
    .configure_view(stroke=None, strokeWidth=0)
)
fig

# %%
horizontal_title = "Year"
values_title = "Number of reviews"
colour_title = "Category"

app_reviews_ = app_reviews.copy()
app_reviews_.loc[app_reviews.user == "Children", "cluster"] = "Children apps (combined)"
app_reviews_.loc[app_reviews.user == "Children", "user"] = "Parents"

reviews_parent_apps_per_year_by_category = (
    app_reviews_.query("reviewYear < 2022")
    .query("user == 'Parents'")
    .groupby(["cluster", "reviewYear"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
    .rename(
        columns={
            "reviewYear": horizontal_title,
            "review_count": values_title,
            "cluster": colour_title,
        }
    )
)

# %%
reviews_parent_apps_per_year_by_category

# %%
# Longer term trend (2017 -> 2021)
values_title_ = "Number of reviews"
values_title = "Growth"
labels_title = "Category"

df_short = (
    utils.get_estimates(
        reviews_parent_apps_per_year_by_category,
        value_column=values_title_,
        time_column=horizontal_title,
        category_column=colour_title,
        estimate_function=utils.smoothed_growth,
        year_start=2017,
        year_end=2021,
    )
    .assign(growth=lambda df: (df[values_title_] / 100).round(2))
    .rename(columns={"growth": values_title})
)
df_short

# %%
importlib.reload(utils)
reviews_magnitude_vs_growth = utils.get_magnitude_vs_growth(
    reviews_parent_apps_per_year_by_category,
    value_column=values_title_,
    time_column=horizontal_title,
    category_column=colour_title,
)
reviews_magnitude_vs_growth

# %%
fig = (
    alt.Chart(
        (reviews_magnitude_vs_growth.head(6).assign(Growth=lambda df: df.Growth / 100)),
        width=300,
        height=300,
    )
    .mark_circle(size=50)
    .encode(
        x=alt.X(
            "Magnitude:Q",
            axis=alt.Axis(title=f"Average number of reviews per year"),
            # scale=alt.Scale(type="linear"),
        ),
        y=alt.Y(
            "Growth:Q",
            axis=alt.Axis(format="%"),
            # axis=alt.Axis(
            #     title=f"Growth between {start_year} and {end_year} measured by number of reviews"
            # ),
            # scale=alt.Scale(domain=(-.100, .300)),
        ),
        # size="cluster_size:Q",
        color=alt.Color(f"{colour_title}:N", legend=None),
        tooltip=[
            "Category",
            alt.Tooltip("Magnitude", format=",", title="Number of reviews"),
            alt.Tooltip("Growth", format=".0%"),
        ],
    )
    .properties(
        title={
            "anchor": "start",
            "text": "Review trends of parenting app categories",
            "subtitle": "Magnitude and growth of reviews between 2017 and 2021",
            "subtitleFont": pu.FONT,
            "subtitleFontSize": pu.FONTSIZE_SUBTITLE,
        },
    )
)

text = fig.mark_text(align="left", baseline="middle", font=pu.FONT, dx=7).encode(
    text=colour_title
)

fig_final = (
    (fig + text)
    .configure_axis(
        gridDash=[1, 7],
        gridColor="white",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_legend(
        titleFontSize=pu.FONTSIZE_NORMAL,
        labelFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_view(strokeWidth=1)
)
fig_final

# %%
importlib.reload(utils)
table_name = "magnitude_growth_parent_apps_reviews"
# utils.save_data_table(reviews_per_year_by_user, table_name)
AltairSaver.save(fig_final, table_name, filetypes=["html", "svg", "png"])

# %%
# Short term trend (2019 -> 2020)
values_title_ = "Number of reviews"
values_title = "Growth"
labels_title = "Category"

df_short = (
    utils.get_estimates(
        reviews_parent_apps_per_year_by_category,
        value_column=values_title_,
        time_column=horizontal_title,
        category_column=colour_title,
        estimate_function=utils.growth,
        year_start=2019,
        year_end=2020,
    )
    .assign(growth=lambda df: (df[values_title_] / 100).round(2))
    .rename(columns={"growth": values_title})
)
df_short

# %%
chart_title = "Growth of Play Store reviews for parent apps"
chart_subtitle = "Precentage change from 2019 to 2020"
tooltip = [labels_title, values_title]


fig = (
    alt.Chart(
        df_short,
        width=300,
        height=300,
    )
    .mark_bar()
    .encode(
        x=alt.X(
            f"{values_title}:Q",
            title=values_title,
            axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=(-0.25, 2.00)),
        ),
        y=alt.Y(
            f"{labels_title}:N",
            title=labels_title,
            sort="-x",
            axis=alt.Axis(labelLimit=200),
        ),
        color=labels_title,
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
    .configure_view(stroke=None, strokeWidth=0)
)

fig

# %%
importlib.reload(utils)
table_name = "no_of_reviews_growth_parent_apps"
# utils.save_data_table(reviews_per_year_by_user, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "png"])

# %%
tooltip = [colour_title, horizontal_title, values_title]
chart_title = "Reviews on Play Store"
chart_subtitle = ""

fig = (
    alt.Chart(
        reviews_parent_apps_per_year_by_category.query("Year >= 2010"),
        width=300,
        height=200,
    )
    .mark_bar(size=20)
    .encode(
        x=alt.X(
            f"{horizontal_title}:O",
            scale=alt.Scale(domain=list(range(2010, 2022, 1))),
        ),
        y=alt.Y(
            f"sum({values_title}):Q",
            # stack="normalize",
            title="Proportion of user reviews"
            # scale = alt.Scale(domain=(0, 100_000))
        ),
        color=f"{colour_title}:N",
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
    .configure_view(stroke=None, strokeWidth=0)
)
fig

# %%
horizontal_title = "Month"
values_title = "Number of reviews"
colour_title = "Category"
years = [2019, 2020, 2021]

review_parent_apps_per_month_by_category = (
    app_reviews.query("reviewYear in @years")
    .query("user == 'Parents'")
    .assign(year_month=lambda df: pd.to_datetime(df["at"]).dt.to_period("M"))
    .astype({"year_month": str})
    .groupby(["cluster", "year_month"], as_index=False)
    .agg(
        review_count=("reviewId", "count"),
    )
    .rename(
        columns={
            "year_month": horizontal_title,
            "review_count": values_title,
            "cluster": colour_title,
        }
    )
)

# %%
tooltip = [colour_title, horizontal_title, values_title]
chart_title = "Number of monthly reviews on Play Store, 2019-2021"
chart_subtitle = ""

fig = (
    alt.Chart(
        review_parent_apps_per_month_by_category,
        width=450,
        height=200,
    )
    .mark_line(size=3)
    .encode(
        x=alt.X(
            f"{horizontal_title}:O",
            # scale = alt.Scale(domain=list(range(2010, 2022,1))),
        ),
        y=alt.Y(f"{values_title}:Q", scale=alt.Scale(domain=(0, 3_000))),
        color=f"{colour_title}:N",
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
    .configure_view(stroke=None, strokeWidth=0)
    .interactive()
)
fig

# %% [markdown]
# #### Downloads

# %%
installs_by_cluster = (
    app_details.query("user == 'Children'")
    .groupby("cluster")
    .agg(installs=("minInstalls", "mean"))
    .sort_values("installs", ascending=False)
    .reset_index()
)

# %%
installs_by_cluster

# %%
values_title = "Number of installations (thousands)"
labels_title = "Category"

installs_by_cluster = (
    app_details.query("user == 'Parents'")
    .groupby("cluster")
    .agg(installs=("minInstalls", "mean"))
    .sort_values("installs", ascending=False)
    .reset_index()
    .assign(installs=lambda df: (df.installs / 1000).round().astype(int))
    .rename(
        columns={
            "cluster": labels_title,
            "installs": values_title,
        }
    )
)

# %%
installs_by_cluster.head(3)

# %%
chart_title = "Parenting app installations"
chart_subtitle = "Average number of installations from Play Store by app category"
tooltip = [labels_title, values_title]

# Figure 1
fig = (
    alt.Chart(
        installs_by_cluster,
        width=300,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        x=alt.X(
            f"{values_title}:Q",
            title=values_title,
            scale=alt.Scale(domain=(0, 5000)),
            axis=alt.Axis(labelAlign="center"),
        ),
        y=alt.Y(
            f"{labels_title}:N",
            title=labels_title,
            sort="-x",
            axis=alt.Axis(labelLimit=200),
        ),
        tooltip=tooltip,
    )
    # .properties(
    #     title={
    #         "anchor": "start",
    #         "text": chart_title,
    #         "subtitle": chart_subtitle,
    #         "subtitleFont": pu.FONT,
    #     },
    # )
    # .configure_axis(
    #     gridDash=[1, 7],
    #     gridColor="grey",
    # )
    # .configure_view(strokeWidth=0)
)
fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
importlib.reload(utils)
table_name = "app_installations_parent_apps"
utils.save_data_table(installs_by_cluster, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])

# %% [markdown]
# #### Scores

# %%
values_title = "Score"
labels_title = "Category"

score_by_cluster = (
    app_details.query("user == 'Parents'")
    .query("score != 0")
    .groupby("cluster")
    .agg(score=("score", "mean"))
    .sort_values("score", ascending=False)
    .assign(score=lambda df: df.score.astype(float).round(2))
    .reset_index()
    .rename(
        columns={
            "cluster": labels_title,
            "score": values_title,
        }
    )
)

# %%
score_by_cluster.head(3)

# %%
chart_title = "Parenting app scores"
chart_subtitle = "Average scores on Play Store by app category"
tooltip = [labels_title, values_title]

# Figure 1
fig = (
    alt.Chart(
        score_by_cluster,
        width=300,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        x=alt.X(
            f"{values_title}:Q",
            title=values_title,
            scale=alt.Scale(domain=(0, 5)),
            axis=alt.Axis(labelAlign="center", tickCount=4),
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
)
fig = pu.configure_plots(fig, chart_title, chart_subtitle)
fig

# %%
importlib.reload(utils)
table_name = "app_scores_parent_apps"
utils.save_data_table(score_by_cluster, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "svg", "png"])

# %%
# fig =  (
#     alt.Chart(
#         (
#             app_details
#             .query("user == 'Parents'")
#             .query("score != 0")
#         ),
#         width=300,
#         height=50
#     )
#     .mark_circle(size=20)
#     .encode(
#         y=alt.Y(
#             'jitter:Q',
#             title=None,
#             axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
#             scale=alt.Scale(),
#         ),
#         x=alt.X(
#             'score:Q',
#             # scale=alt.Scale(domain=(20, 85))
#         ),
#         color=alt.Color('cluster:N', legend=None),
#         row=alt.Row(
#             'cluster:N',
#             header=alt.Header(
#                 labelAngle=0,
#                 labelFontSize=16,
#                 titleOrient='top',
#                 labelOrient='left',
#                 labelAlign='left',
#             ),
#         ),
#     )
#     .transform_calculate(
#         # Generate Gaussian jitter with a Box-Muller transform
#         jitter='sqrt(-2*log(random()))*cos(2*PI*random())')
#     .configure_facet(
#         spacing=0
#     )
#     .configure_view(
#         stroke=None
#     )
#     .configure_axis(
#         labelFontSize=16,
#         titleFontSize=16
#     )
# )

# fig

# %% [markdown]
# #### App development trends

# %%
# New apps per year, for each cluster
horizontal_title = "Release year"
values_title = "Number of apps"
color_title = "Category"

new_apps_per_year_per_cluster = (
    app_details.groupby(["releaseYear", "cluster", "user"])
    .agg(counts=("appId", "count"))
    .reset_index()
    .query("user == 'Parents'")
)

new_apps_per_year_per_cluster_ = new_apps_per_year_per_cluster.rename(
    columns={
        "releaseYear": horizontal_title,
        "counts": values_title,
        "cluster": color_title,
        "user": "User",
    }
)

# %%
new_apps_per_year_per_cluster.head(2)

# %%
chart_title = "Parenting app development trends"
chart_subtitle = "Number of Play Store apps by release year "
tooltip = [labels_title, values_title, color_title]

fig_ts = (
    alt.Chart(new_apps_per_year_per_cluster_, width=400, height=300)
    .mark_bar()
    .encode(
        x=alt.X(f"{horizontal_title}:O", title=horizontal_title),
        y=alt.Y(f"sum({values_title}):Q", title=values_title),
        color=f"{color_title}:N",
        tooltip=tooltip,
    )
    .properties(
        title={
            "anchor": "start",
            "text": chart_title,
            "subtitle": chart_subtitle,
            "subtitleFont": pu.FONT,
            "subtitleFontSize": pu.FONTSIZE_SUBTITLE,
        },
    )
    # .configure_axis(
    #     gridDash=[1, 7],
    #     gridColor="grey",
    # )
    # .configure_view(strokeWidth=0)
)

fig_ts_ = pu.configure_plots(fig_ts, chart_title, chart_subtitle)
fig_ts_

# %%
AltairSaver.save(
    fig_ts_, "parent_apps_release_number", filetypes=["html", "svg", "png"]
)

# %%
# Longer term, smoothed trend (2017 -> 2021)
labels_title = "Category"
values_title = "Growth"

df_long = (
    utils.get_estimates(
        new_apps_per_year_per_cluster,
        value_column="counts",
        time_column="releaseYear",
        category_column="cluster",
        estimate_function=utils.smoothed_growth,
        year_start=2017,
        year_end=2021,
    ).assign(counts=lambda df: (df.counts / 100).round(2))
    # .assign(percentage = lambda df: (df.counts/100).round(2))
    .rename(
        columns={
            "cluster": labels_title,
            "counts": values_title,
        }
    )
)
df_long

# %%
chart_title = " "
chart_subtitle = "Smoothed growth estimate between 2017 and 2021"
tooltip = [labels_title, values_title]

# Figure 1
fig = (
    alt.Chart(
        df_long,
        width=300,
        height=300,
    )
    .mark_bar()
    .encode(
        x=alt.X(
            f"{values_title}:Q",
            title="",
            axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=(-0.40, 1.00)),
        ),
        y=alt.Y(
            f"{labels_title}:N",
            title=labels_title,
            sort="-x",
            axis=alt.Axis(labelLimit=200),
        ),
        color=labels_title,
        tooltip=[
            labels_title,
            alt.Tooltip(f"{values_title}:Q", title=values_title, format=".0%"),
        ],
    )
    .properties(
        title={
            "anchor": "start",
            "text": chart_title,
            "subtitle": chart_subtitle,
            "subtitleFont": pu.FONT,
            "subtitleFontSize": pu.FONTSIZE_SUBTITLE,
        },
    )
)
fig_ = pu.configure_plots(fig, chart_title, chart_subtitle)
fig_

# %%
AltairSaver.save(fig_, "parent_apps_release_growth", filetypes=["html", "svg", "png"])

# %%
fig_final = (
    alt.hconcat(fig_ts, fig)
    .configure_axis(
        gridDash=[1, 7],
        gridColor="grey",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_legend(
        titleFontSize=pu.FONTSIZE_NORMAL,
        labelFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_view(strokeWidth=0)
)

fig_final

# %%
importlib.reload(utils)
table_name = "no_of_parent_apps_growth"
# utils.save_data_table(score_by_cluster, table_name)
AltairSaver.save(fig_final, table_name, filetypes=["html", "svg", "png"])

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
# #### Downloads

# %%
app_details.appId.iloc[0]

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
