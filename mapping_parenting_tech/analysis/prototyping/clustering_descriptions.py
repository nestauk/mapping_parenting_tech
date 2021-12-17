# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Clustering of app descriptions
# Takes descriptions of Play Store apps and clusters accordingly, producing a visualisation of the clusters.

# %%
from sentence_transformers import SentenceTransformer
from mapping_parenting_tech.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import umap
import hdbscan
import altair as alt
import numpy as np
import pandas as pd
import pickle
from mapping_parenting_tech import PROJECT_DIR

alt.data_transformers.disable_max_rows()

INPUT_DATA = PROJECT_DIR / "outputs/data"

# %%
# Load an embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# %%
# Load in the descriptions
descriptions = pd.read_csv(open(INPUT_DATA / "simple_app_descriptions.csv", "rb"))

# %%
# Generate sentence embeddings (might take a few minutes for 1000s of sentences)
description_embeddings = np.array(model.encode(descriptions.description.to_list()))

# %%
# Check the shape of the sentence embeddings array
print(description_embeddings.shape)

# %%
# Create a 2D embedding
reducer = umap.UMAP(n_components=2, random_state=1)
embedding = reducer.fit_transform(description_embeddings)

# %%
# Check the shape of the reduced embedding array
embedding.shape

# %%
# Create another low-dim embedding for clustering
reducer_clustering = umap.UMAP(n_components=50, random_state=1)
embedding_clustering = reducer_clustering.fit_transform(description_embeddings)

# %%
# Clustering with hdbscan
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=6, min_samples=1, cluster_selection_method="leaf"
)
clusterer.fit(embedding_clustering)

# %%
# Prepare dataframe for visualisation
df = descriptions.copy()
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]
df["cluster"] = [str(x) for x in clusterer.labels_]

# %%
# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
fig = (
    alt.Chart(df, width=500, height=500)
    .mark_circle(size=60)
    .encode(x="x", y="y", tooltip=["cluster", "app_id", "summary"], color="cluster")
).interactive()

fig

# %%
of_interest = ["5", "13"]
df[df.cluster.isin(of_interest)].app_id.to_list()

# %%
driver = google_chrome_driver_setup()

# %%
save_altair(fig, "cluster_descriptions", driver)
