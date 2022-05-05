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
# Prepare dataframe for visualisation
df = (
    app_details.copy()
    .assign(x=embedding[:, 0])
    .assign(y=embedding[:, 1])
    .assign(circle_size=lambda x: 0.2(x.score + 1))
    .assign(user=lambda x: x.cluster.apply(utils.map_cluster_to_user))
)

# %%
df.info()

# %%
df

# %%
# Visualise using altair
fig = (
    alt.Chart(df.reset_index(), width=700, height=700)
    .mark_circle(size=60)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        tooltip=["cluster", "appId", "title", "description", "score", "url"],
        color="cluster:N",
        href="url:N",
        size="circle_size",
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
filename = "app_landscape_v2022_05_04"
AltairSaver.save(fig, filename, filetypes=["html"])

# %%
import utils

utils.to_

# %%
df.to_csv(AltairSaver.path + f"/{filename}.csv", index=False)

# %%
