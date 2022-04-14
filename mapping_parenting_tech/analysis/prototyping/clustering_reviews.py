# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# # Clustering of app reviews
#
# - Creating embeddings of app reviews (at the sentence level)
# - Visualising the embeddings using UMAP
# - Clustering the reviews

# +
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import altair as alt
import numpy as np
import pandas as pd
import pickle
from mapping_parenting_tech import PROJECT_DIR

alt.data_transformers.disable_max_rows()

INPUT_DATA = PROJECT_DIR / "outputs/data/clustering"
# -

# Load an embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# +
# Load in the sentences
reviews_sentences = pickle.load(
    open(INPUT_DATA / "reviews_for_clustering_sentences.pickle", "rb")
)

# Load in the metadata
reviews_metadata = pd.read_csv(INPUT_DATA / "reviews_for_clustering.csv")

# Create one dataframe
reviews_metadata["sentences"] = reviews_sentences
reviews_metadata = reviews_metadata.explode("sentences")

# Create a random sample (for demo purposes)
reviews_random_sample = reviews_metadata.sample(3000, random_state=1)
# -

reviews_random_sample.head(3)

sentence_subsample = reviews_random_sample.sentences.to_list()

sentence_subsample[0:10]

# Generate sentence embeddings (might take a few minutes for 1000s of sentences)
sentence_embeddings = np.array(model.encode(reviews_random_sample.sentences.to_list()))

# Check the shape of the sentence embeddings array
print(sentence_embeddings.shape)

# Create a 2D embedding
reducer = umap.UMAP(n_components=2, random_state=1)
embedding = reducer.fit_transform(sentence_embeddings)

# Check the shape of the reduced embedding array
embedding.shape

# Create another low-dim embedding for clustering
reducer_clustering = umap.UMAP(n_components=50, random_state=1)
embedding_clustering = reducer_clustering.fit_transform(sentence_embeddings)

# Clustering with hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
clusterer.fit(embedding_clustering)

# Prepare dataframe for visualisation
df = reviews_random_sample.copy()
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]
df["cluster"] = [str(x) for x in clusterer.labels_]

# +
# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
fig = (
    alt.Chart(df, width=500, height=500)
    .mark_circle(size=60)
    .encode(x="x", y="y", tooltip=["sentences", "appId"], color="cluster")
).interactive()

fig
