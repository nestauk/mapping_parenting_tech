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
# Functionality for saving charts
import mapping_parenting_tech.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver()

# %% [markdown]
# ## Load and process data

# %%
# Load model and associated data - step can take a few minutes
model_data = lmu.load_lda_model_data(model_name=MODEL_NAME, folder=TPM_DIR)
mdl = model_data["model"]

# %%
# Get the topic probabilities for each document (review)
doc_topic_probabilities = lmu.get_document_topic_distributions(mdl)

topic_probability_table = lmu.create_document_topic_probability_table(
    model_data["document_ids"], model_data["model"], doc_topic_probabilities
)

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
# Load reviews for the relevant apps
relevant_reviews = psu.load_some_app_reviews(relevant_apps["appId"])

# %%
# add the appId to each review in `topic_probability_table`, so we know what it's connected to
topic_probability_table = topic_probability_table.merge(
    relevant_reviews[["reviewId", "appId"]], left_on="id", right_on="reviewId"
).drop(columns=["reviewId"])

# %%
# calculate the topic probability for each app
app_topic_probabilities = dict()

for appId in topic_probability_table["appId"].unique().tolist():
    focus_data = topic_probability_table[topic_probability_table["appId"] == appId]

    _app_topic_probabilities = [focus_data[f"topic_{i}"].sum() for i in range(mdl.k)]

    app_topic_probabilities[appId] = {
        f"topic_{i}": p / sum(_app_topic_probabilities)
        for i, p in enumerate(_app_topic_probabilities)
    }

app_topic_probabilities = (
    pd.DataFrame(app_topic_probabilities)
    .T.reset_index()
    .rename(columns={"index": "appId"})
)

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
import mapping_parenting_tech.utils.embeddings_utils as eu
from sentence_transformers import SentenceTransformer

# Embedding model name
EMBEDDING_MODEL = "all-mpnet-base-v2"
# File names
vector_filename = "app_description_vectors_2022_04_27"
embedding_model = EMBEDDING_MODEL
EMBEDINGS_DIR = PROJECT_DIR / "outputs/data"

# %%
model = SentenceTransformer(EMBEDDING_MODEL)

# %%
# # Load the existing embeddings of the app descriptions
# filename = "description_embeddings-22-01-21.pickle"
# filename = "description_embeddings-22-04-19.pickle"
# with open(OUTPUT_DIR / filename, "rb") as f:
#     description_embeddings = pickle.load(f)

# %%
# description_embeddings.shape

# %%
# Generate sentence embeddings (might take a few minutes for 1000s of sentences)
description_embeddings = np.array(model.encode(app_details.description.to_list()))

# %%
v = eu.Vectors(
    vector_ids=app_details["appId"].to_list(),
    vectors=description_embeddings,
    filename=vector_filename,
    model_name=EMBEDDING_MODEL,
    folder=EMBEDINGS_DIR,
)
v.vectors.shape

# %%
# v.save_vectors(vector_filename, EMBEDINGS_DIR)

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
embedding.shape

# %%
# Prepare dataframe for visualisation
df = app_details.copy()
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]

# %%
from mapping_parenting_tech.utils import plotting_utils as pu

# %%
app_details.info()

# %%
# Visualise using altair
fig = (
    alt.Chart(df.reset_index(), width=700, height=700)
    .mark_circle(size=60)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        tooltip=["cluster", "title", "description", "score", "url"],
        color="cluster:N",
        href="url:N",
        # size='minInstalls',
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
filename = "app_landscape_v2022_04_27"
AltairSaver.save(fig, filename, filetypes=["html"])

# %%
df.to_csv(AltairSaver.path + f"/{filename}.csv", index=False)

# %% [markdown]
# ## Visualising topics

# %%
# merge the topic probabilities with the app details; fill apps that have no reviews with 0 to avoid 'NaN'
df = pd.merge(
    details[["appId", "cluster", "summary"]],
    app_topic_probabilities,
    on="appId",
    how="left",
).fillna(0)

df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]

# %%
topic = "topic_14"
# Visualise using altair
fig = (
    alt.Chart(df.reset_index(), width=700, height=700)
    .mark_circle(
        size=50,
        filled=True,
        opacity=1,
    )
    .encode(
        x="x",
        y="y",
        tooltip=["cluster", "appId", "summary"],
        color=alt.Color(f"{topic}:Q", scale=alt.Scale(scheme="yellowgreenblue")),
    )
).interactive()

fig

# %%
