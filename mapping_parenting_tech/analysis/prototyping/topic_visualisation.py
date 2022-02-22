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
#     display_name: Python 3.8.0 ('mapping_parenting_tech')
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Topic visualisation
# Visualising topics of app reviews, overlaying them with their clusters

# %%
from mapping_parenting_tech import logging, PROJECT_DIR
from mapping_parenting_tech.utils import lda_modelling_utils as lmu

import pandas as pd
import numpy as np
import altair as alt
import pickle
import umap

OUTPUT_DIR = PROJECT_DIR / "outputs/data"
INPUT_DIR = PROJECT_DIR / "inputs/data/play_store"
TPM_DIR = OUTPUT_DIR / "tpm"
MODEL_NAME = "play_store_reviews"

# %%
# Load model and associated data - step can take a few minutes
model_data = lmu.load_lda_model_data(model_name=MODEL_NAME, folder=TPM_DIR)
mdl = model_data["model"]

# %%
doc_topic_probabilities = lmu.get_document_topic_distributions(mdl)

topic_probability_table = lmu.create_document_topic_probability_table(
    model_data["document_ids"], model_data["model"], doc_topic_probabilities
)

# %%
# Load in app details
details = pd.read_json(OUTPUT_DIR / "all_app_details.json", orient="index")
details.reset_index(inplace=True)
details.rename(columns={"index": "appId"}, inplace=True)
details.shape

# %%
# Load ids for those apps that are relevant and add the cluster to the dataframe
relevant_apps = pd.read_csv(INPUT_DIR / "relevant_app_ids.csv")
# NB: left join to preserve apps that aren't relevant - they're needed to map onto the embedding...
details = details.merge(
    relevant_apps,
    how="left",
    on="appId",
)

# %%
# get the index of those apps that aren't relevant
remove_apps = details[details["cluster"].isna()].index

# %%
# Load the existing embeddings of the app descriptions
with open(OUTPUT_DIR / "description_embeddings-22-01-21.pickle", "rb") as f:
    description_embeddings = pickle.load(f)

# %%
# remove 'irrelevant' apps from the dataframe and the embedding
description_embeddings = np.delete(description_embeddings, remove_apps, 0)
details.drop(remove_apps, inplace=True)

# %%
# Reduce the embedding to 2 dimensions
reducer = umap.UMAP(n_components=2, random_state=1)
embedding = reducer.fit_transform(description_embeddings)

# %%
embedding.shape

# %%
# Prepare dataframe for visualisation
df = details.copy()
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]

# %%
# Visualise using altair
fig = (
    alt.Chart(df.reset_index(), width=750, height=750)
    .mark_circle(size=60)
    .encode(x="x", y="y", tooltip=["cluster", "appId", "summary"], color="cluster:N")
).interactive()

fig

# %%
