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
# # Topic visualisation
# Visualising topics of app reviews, overlaying them with their clusters

# %%
from mapping_parenting_tech import logging, PROJECT_DIR
from mapping_parenting_tech.utils import lda_modelling_utils as lmu
from mapping_parenting_tech.utils import play_store_utils as psu
from tqdm import tqdm

import pandas as pd
import numpy as np
import altair as alt
import pickle
import umap

OUTPUT_DIR = PROJECT_DIR / "outputs/data"
REVIEWS_DIR = OUTPUT_DIR / "app_reviews"
INPUT_DIR = PROJECT_DIR / "inputs/data/play_store"
TPM_DIR = OUTPUT_DIR / "tpm"
MODEL_NAME = "play_store_reviews"

# %%
import utils

# %%
import mapping_parenting_tech.utils.embeddings_utils as eu
from sentence_transformers import SentenceTransformer
from mapping_parenting_tech.utils import plotting_utils as pu

# %%
# Functionality for saving charts
import mapping_parenting_tech.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver()

# %% [markdown]
# ## Load and process data

# %%
# Load ids for those apps that are relevant
relevant_apps = pd.read_csv(INPUT_DIR / "relevant_app_ids.csv")

# %%
# Load in app details
details = pd.read_json(OUTPUT_DIR / "all_app_details.json", orient="index")
details.reset_index(inplace=True, drop=True)
# details.rename(columns={"index": "appId"}, inplace=True)
details.shape

# %%
# Add apps' cluster to their details
# NB: left join to preserve apps that aren't relevant - they're needed to map onto the embedding...
details = details.merge(
    relevant_apps,
    how="left",
    on="appId",
)

# %%
# get the index of those apps that aren't relevant
remove_apps = details[details["cluster"].isna()].index

# %% [markdown]
# ## Plot embeddings

# %%
app_details = details[-details.cluster.isnull()]

# %%
# Embedding model name
EMBEDDING_MODEL = "all-mpnet-base-v2"
# File names
vector_filename = "app_description_vectors_2022_04_27"
embedding_model = EMBEDDING_MODEL
EMBEDINGS_DIR = PROJECT_DIR / "outputs/data"

# %%
# model = SentenceTransformer(EMBEDDING_MODEL)

# %%
v = eu.Vectors(
    filename=vector_filename, model_name=EMBEDDING_MODEL, folder=EMBEDINGS_DIR
)
v.vectors.shape

# %%
# app_details[app_details.title.str.contains('AppClose')]

# %%
description_embeddings = v.select_vectors(app_details.appId.to_list())
description_embeddings.shape

# %%
# # remove 'irrelevant' apps from the dataframe and the embedding
# description_embeddings = np.delete(description_embeddings, remove_apps, 0)
# details.drop(remove_apps, inplace=True)

# %%
# Reduce the embedding to 2 dimensions
reducer = umap.UMAP(
    n_components=2,
    random_state=1,
    n_neighbors=8,
    min_dist=0.3,
    spread=0.5,
)
embedding = reducer.fit_transform(description_embeddings)

# %%
# Reduce the embedding to 2 dimensions
reducer = umap.UMAP(
    n_components=2,
    random_state=2,
    n_neighbors=20,
    min_dist=0.5,
    spread=0.7,
)
embedding = reducer.fit_transform(description_embeddings)

# %%
# Prepare dataframe for visualisation
df = (
    app_details.copy()
    .assign(x=embedding[:, 0])
    .assign(y=embedding[:, 1])
    .assign(circle_size=lambda x: 0.2 * (x.score + 1))
    .assign(user=lambda x: x.cluster.apply(utils.map_cluster_to_user))
)

# %% [markdown]
# ## Prepare

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
filename = "app_landscape_v2022_05_04"
df_viz = pd.read_csv(AltairSaver.path + f"/tables/{filename}.csv")
df_viz[["x", "y"]] = embedding

df_viz = (
    df_viz.fillna({"score": 0})
    .assign(
        Description=lambda df: df.description.apply(shorten_and_clean),
        Installations=lambda df: df.minInstalls.apply(lambda x: f"{x:,}"),
        Score=lambda df: df.score.apply(lambda x: np.round(x, 2)),
    )
    .astype({"Score": float})
    .rename(
        columns={
            "icon": "image",
            "cluster": "Category",
            "user": "User",
        }
    )
    .reset_index()
    .drop(["score", "description", "minInstalls"], axis=1)
)
df_viz.loc[df_viz["Score"] == 0, "Score"] = "n/a"
df_viz["Score"] = df_viz["Score"].astype(str)

# %%
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
import re
from mapping_parenting_tech.utils.text_preprocessing_utils import (
    remove_non_alphanumeric,
)
from mapping_parenting_tech.utils.text_preprocessing_utils import simple_tokenizer
from mapping_parenting_tech.analysis import cluster_analysis_utils
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

defined_stopwords = [
    "kids",
    "kid",
    "baby",
    "potty",
    "mini",
    "sago",
    "dino",
    "dinosaurs",
    "game",
]

full_stopwords = stopwords.words("english") + defined_stopwords


def preproc(text: str) -> str:
    text = re.sub(r"[^a-zA-Z ]+", "", text).lower()
    text = simple_tokenizer(text)
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [t for t in text if t not in full_stopwords]
    return " ".join(text)


clusterer = KMeans(n_clusters=40, random_state=10)
clusterer.fit(df[["x", "y"]])
soft_clusters = list(clusterer.labels_)
soft_cluster = [np.argmax(x) for x in soft_clusters]

# %%
title_texts = df_viz["title"].apply(preproc)

# %%
cluster_texts = cluster_analysis_utils.cluster_texts(title_texts, soft_clusters)

# %%
cluster_keywords = cluster_analysis_utils.cluster_keywords(
    documents=list(cluster_texts.values()),
    cluster_labels=list(cluster_texts.keys()),
    n=2,
    max_df=0.90,
    min_df=0.01,
    Vectorizer=TfidfVectorizer,
)


df_viz["soft_cluster"] = soft_clusters
df_viz["soft_cluster_"] = [str(x) for x in soft_clusters]

cluster_keywords[15] = ""
cluster_keywords[33] = ""

centroids = (
    df_viz.groupby("soft_cluster")
    .agg(x_c=("x", "mean"), y_c=("y", "mean"))
    .reset_index()
    .assign(
        keywords=lambda x: x.soft_cluster.apply(
            lambda y: ", ".join(cluster_keywords[y])
        )
    )
)

# %%
import importlib

importlib.reload(pu)

# %%
SPEC_COLOURS = [
    # Nesta brand colors:
    # "#0000FF",
    "#6483e8",
    "#FDB633",
    # "#18A48C",
    "#84c493",
    # "#9A1BBE",
    "#c980ed",
    "#EB003B",
    "#FF6E47",
    # "#646363",
    "#ed8add",
    "#0F294A",
    # parent
    "#a8baae",
    # "#dbdac3",
    "#a6a592",
    "#a3a379",
    "#d4d496",
    "#a3a3a3",
    "#d2d4d2",
    #
    "#97D9E3",
    "#A59BEE",
    "#F6A4B7",
    "#D2C9C0",
    "#000000",
    # Extra non-Nesta colors:
    "#4d16c4",
]

# %% [markdown]
# ### Plot

# %%
# Visualise using altair
fig = (
    alt.Chart(df_viz, width=700, height=700)
    .mark_circle(size=60, opacity=0.60)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        tooltip=[
            "image",
            "title",
            "User",
            "Category",
            "Description",
            "Installations",
            "Score",
            # "x", "y",
        ],
        color=alt.Color(
            "Category:N",
            legend=alt.Legend(
                # titleAnchor='left',
                # orient='right',
                labelLimit=200,
            ),
            sort=sorted(utils.clusters_children) + sorted(utils.clusters_parents),
            scale=alt.Scale(range=SPEC_COLOURS),
        ),
        href="url:N",
        # size="circle_size",
    )
)

text = (
    alt.Chart(centroids)
    .mark_text(
        font=pu.FONT,
        fontSize=13.5,
        fontStyle="bold",
        opacity=0.8,
        stroke="white",
        strokeWidth=1,
        strokeOffset=0,
        strokeOpacity=0.4,
    )
    .encode(x=alt.X("x_c:Q"), y=alt.Y("y_c:Q"), text=alt.Text("keywords"))
)

df_txt = pd.DataFrame(
    data={
        "x": [2.4, -1.5],
        "y": [6.5, 13],
        "text": ["Parenting apps", "Children apps"],
    }
)

# df_line_1 = pd.DataFrame(data={
#     'x': [2.2, 2.7],
#     'y': [6.2, 6.8],
# })

annotation_text = (
    alt.Chart(df_txt)
    .mark_text(font=pu.FONT, fontSize=15, align="left", fontStyle="bold")
    .encode(
        x=alt.X("x:Q"),
        y=alt.Y("y:Q"),
        text=alt.Text("text"),
    )
)

# line_1 = (
#     alt.Chart(df_line_1)
#     .mark_line(color="#333333", strokeDash=[1.5, 4])
#     .encode(
#         x=alt.X("x:Q"),
#         y=alt.Y("y:Q"),
#     )
# )

# line_2 = (
#     alt.Chart(df_line_1)
#     .mark_line(color="#333333", strokeDash=[1.5, 4])
#     .encode(
#         x=alt.X("x:Q"),
#         y=alt.Y("y:Q"),
#     )
# )


fig_final = (
    (fig + text + annotation_text)
    .configure_axis(
        # gridDash=[1, 7],
        gridColor="white",
    )
    .configure_view(strokeWidth=0, strokeOpacity=0)
    .properties(
        title={
            "anchor": "start",
            "text": ["Children and parenting app landscape"],
            "subtitle": [
                "Each app is visualised as a circle, with similar apps located closer together",
            ],
            "subtitleFont": pu.FONT,
            "subtitleFontSize": 14,
        },
    )
    .interactive()
)

fig_final

# %%
# filename = "app_landscape_v2022_05_28"
# AltairSaver.save(fig_final, filename, filetypes=["html", "svg", "png"])

# %%
df_viz.to_csv(AltairSaver.path + f"/tables/{filename}.csv", index=False)

# %%
import mapping_parenting_tech.utils.io

importlib.reload(mapping_parenting_tech.utils.io)
mapping_parenting_tech.utils.io.save_json(
    (
        df_viz.drop(
            ["comments", "descriptionHTML", "screenshots", "recentChanges"], axis=1
        )
    ).to_dict(orient="records"),
    AltairSaver.path + f"/{filename}.json",
)

# %%
AltairSaver.path + f"/{filename}.json"

# %%

# %% [markdown]
# ## Figure using URLs

# %%
importlib.reload(mapping_parenting_tech.utils.io)

# %%
mapping_parenting_tech.utils.io.save_json(
    df_viz[
        ["x", "y", "title", "User", "Category", "Description", "Installations", "Score"]
    ].to_dict(orient="records"),
    # df_viz[['x','y']].to_dict(orient='records'),
    AltairSaver.path + f"/test.json",
)

# %%
xx = df_viz[
    [
        "x",
        "y",
        "title",
        "User",
        "Category",
        "Description",
        "Installations",
        "Score",
        "image",
    ]
]
xx.to_csv(AltairSaver.path + f"/test.csv", index=False)

# %%
alt.data_transformers.enable("json")

# %%
import mapping_parenting_tech.utils.io

x = mapping_parenting_tech.utils.io.load_json(AltairSaver.path + f"/test.json")

# %%
pd.DataFrame(x).astype("float32").info()

# %%
# url = "https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data_viz/tables/app_landscape_v2022_05_28.csv"
# url = "https://www.dropbox.com/s/d7n4z5bw2wn7wz7/app_landscape_v2022_05_28.csv?dl=1"
# url = "https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data_viz/app_landscape_v2022_05_04.json"
# url = AltairSaver.path + f"/test.json"
# url = 'https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data_viz/test.json'
url = "https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data_viz/test.json"
url = "https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/mapping_parenting_tech/data_viz/test.csv"
url = "https://raw.githubusercontent.com/beingkk/test/main/app_landscape_v2.csv"
# Visualise using altair
fig = (
    alt.Chart(
        url,
        # df_viz[['x','y']],
        width=700,
        height=500,
    )
    .mark_circle(size=60, opacity=0.60)
    .encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None),
        tooltip=[
            # "image",
            "title:N",
            "User:N",
            "Category:N",
            "Description:N",
            "Installations:N",
            "Score:N",
            # "x", "y",
        ],
        color=alt.Color(
            "Category:N",
            legend=alt.Legend(
                # titleAnchor='left',
                orient="bottom",
                columns=2,
                # labelLimit=200,
            ),
            sort=sorted(utils.clusters_children) + sorted(utils.clusters_parents),
            scale=alt.Scale(range=SPEC_COLOURS),
        ),
        # href="url:N",
        # size="circle_size",
    )
)

text = (
    alt.Chart(centroids)
    .mark_text(
        font=pu.FONT,
        fontSize=13.5,
        fontStyle="bold",
        opacity=0.8,
        stroke="white",
        strokeWidth=1,
        strokeOffset=0,
        strokeOpacity=0.4,
    )
    .encode(x=alt.X("x_c:Q"), y=alt.Y("y_c:Q"), text=alt.Text("keywords"))
)

df_txt = pd.DataFrame(
    data={
        "x": [2.4, -1.5],
        "y": [6.5, 13],
        "text": ["Parenting apps", "Children apps"],
    }
)

annotation_text = (
    alt.Chart(df_txt)
    .mark_text(font=pu.FONT, fontSize=15, align="left", fontStyle="bold")
    .encode(
        x=alt.X("x:Q"),
        y=alt.Y("y:Q"),
        text=alt.Text("text"),
    )
)


fig_final = (
    # (fig + text + annotation_text)
    (fig + text)
    .configure_axis(
        # gridDash=[1, 7],
        gridColor="white",
    )
    .configure_view(strokeWidth=0, strokeOpacity=0)
    # .properties(
    # title={
    # "anchor": "start",
    # "text": ["Children and parenting app landscape"],
    # "subtitle": [
    # "Each app is visualised as a circle, with similar apps located closer together",
    # ],
    # "subtitleFont": pu.FONT,
    # "subtitleFontSize": 14,
    # },
    # )
    .interactive()
)

fig_final

# %%
filename = "app_landscape_test"
AltairSaver.save(fig_final, filename, filetypes=["html", "svg", "png"])

# %%
data.airports.url

# %%
data.cars.url

# %%
import altair as alt
from vega_datasets import data

url = data.cars.url
# url = data.airports.url

alt.Chart(url).mark_point().encode(x="latitude:Q", y="longitude:Q")

# %% [markdown]
# ## Only parenting apps

# %%
# Visualise using altair
fig = (
    alt.Chart(df_viz.query("User == 'Parents'"), width=700, height=600)
    .mark_circle(size=70, opacity=0.60)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        tooltip=[
            "image",
            "title",
            "User",
            "Category",
            "Description",
            "Installations",
            "Score",
            # "x", "y",
        ],
        color=alt.Color(
            "Category:N",
            legend=alt.Legend(
                # titleAnchor='left',
                # orient='right',
                labelLimit=200,
            ),
            sort=sorted(utils.clusters_children) + sorted(utils.clusters_parents),
            scale=alt.Scale(range=SPEC_COLOURS),
        ),
        href="url:N",
        # size="circle_size",
    )
)

text = (
    alt.Chart(centroids.query("y_c < 8.1"))
    .mark_text(
        font=pu.FONT,
        fontSize=13.5,
        fontStyle="bold",
        opacity=0.8,
        stroke="white",
        strokeWidth=1,
        strokeOffset=0,
        strokeOpacity=0.4,
    )
    .encode(x=alt.X("x_c:Q"), y=alt.Y("y_c:Q"), text=alt.Text("keywords"))
)

fig_final = (
    (fig + text)
    .configure_axis(
        # gridDash=[1, 7],
        gridColor="white",
    )
    .configure_view(strokeWidth=0, strokeOpacity=0)
    .properties(
        title={
            "anchor": "start",
            "text": ["Parenting app landscape"],
            "subtitle": [
                "Each app is visualised as a circle, with similar apps located closer together",
            ],
            "subtitleFont": pu.FONT,
            "subtitleFontSize": 14,
        },
    )
    .interactive()
)

fig_final

# %%
filename = "parent_app_landscape_v2022_05_28"
AltairSaver.save(fig_final, filename, filetypes=["html", "svg", "png"])

# %% [markdown]
# ## Grid

# %%
from numpy import array, dstack, float32, float64, linspace, meshgrid, random, sqrt
from scipy.spatial.distance import cdist

from lapjv import lapjv


# %%
def test_1024():
    random.seed(777)
    size = 1024
    dots = random.random((size, 2))
    grid = dstack(
        meshgrid(linspace(0, 1, int(sqrt(size))), linspace(0, 1, int(sqrt(size))))
    ).reshape(-1, 2)
    cost = cdist(dots, grid, "sqeuclidean")
    cost *= 100000 / cost.max()
    row_ind_lapjv32, col_ind_lapjv32, _ = lapjv(cost, verbose=True)
    return row_ind_lapjv32, col_ind_lapjv32
    # row_ind_lapjv64, col_ind_lapjv64, _ = lapjv(cost, verbose=True, force_doubles=True)


# %%
test_1024()

# %%
random.seed(777)
size = 1024
dots = random.random((size, 2))
grid = dstack(
    meshgrid(linspace(0, 1, int(sqrt(size))), linspace(0, 1, int(sqrt(size))))
).reshape(-1, 2)
cost = cdist(dots, grid, "sqeuclidean")
cost *= 100000 / cost.max()
row_ind_lapjv32, col_ind_lapjv32, _ = lapjv(cost, verbose=True)

# %%
cost.shape

# %%
random.seed(777)
dots = df[["x", "y"]].iloc[0:1024].to_numpy()
size = dots.shape[0]
# size = 2048
grid = dstack(
    meshgrid(linspace(0, 1, int(sqrt(size))), linspace(0, 1, int(sqrt(size))))
).reshape(-1, 2)
cost = cdist(dots, grid, "sqeuclidean")
cost *= 100000 / cost.max()
row_ind_lapjv32, col_ind_lapjv32, _ = lapjv(cost, verbose=False)
# row_ind_lapjv64, col_ind_lapjv64, _ = lapjv(cost, verbose=False, force_doubles=True)

# %%
grid_jv = grid[row_ind_lapjv32]
df_coords = pd.DataFrame(data=grid_jv, columns=["lap_x", "lap_y"])

# %%
df_ = df.copy().iloc[0:1024]
df_["x_grid"] = df_coords["lap_x"]
df_["y_grid"] = df_coords["lap_y"]

# %%
# Visualise using altair
fig = (
    alt.Chart(
        df_.reset_index().rename(columns={"icon": "image"}), width=700, height=700
    )
    .mark_square(size=60)
    .encode(
        x=alt.X("x_grid:Q", axis=None),
        y=alt.Y("y_grid:Q", axis=None),
        tooltip=[
            "image",
            "cluster",
            "appId",
            "title",
            "description",
            "score",
            "url",
            "x_grid",
            "y_grid",
        ],
        color="cluster:N",
        href="url:N",
        # url="image",
        # size="circle_size",
    )
    .configure_axis(
        # gridDash=[1, 7],
        gridColor="white",
    )
    .configure_view(strokeWidth=0, strokeOpacity=0)
    .properties(
        title={
            "anchor": "start",
            "text": ["Apps for parents and children"],
            "subtitle": ["Landscape of Play Store apps "],
            "subtitleFont": pu.FONT,
        },
    )
    .interactive()
)

fig

# %%
