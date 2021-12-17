# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
from mapping_parenting_tech import PROJECT_DIR

alt.data_transformers.disable_max_rows()

INPUT_DATA = PROJECT_DIR / "outputs/data/clustering"
REVIEWS_TABLE = "reviews_for_clustering_processed.csv"
REVIEWS_EMBEDDINGS = "reviews_sentences_all-mpnet-base-v2.npy"
TOKENS_EMBEDDINGS = "reviews_tokens_all-mpnet-base-v2.npy"
# -

# Load in the table
reviews_df = pd.read_csv(INPUT_DATA / REVIEWS_TABLE)

reviews_df.head(1)

reviews_df["length"] = reviews_df["content_sentence"].apply(lambda x: len(str(x)))

plt.hist(reviews_df.length, bins=50)
plt.show()

reviews_df_ = reviews_df[reviews_df.length > 100]

reviews_df_subsample = reviews_df_.sample(len(reviews_df_), random_state=1)

# Load sentence embeddings
sentence_embeddings = np.load(INPUT_DATA / f"embeddings/{REVIEWS_EMBEDDINGS}")
sentence_embeddings_sub = sentence_embeddings[reviews_df_subsample.index.to_list(), :]

# Check the shape of the sentence embeddings array
print(sentence_embeddings.shape)
print(sentence_embeddings_sub.shape)

# Create a 2D embedding
reducer = umap.UMAP(n_components=2, random_state=1)
embedding = reducer.fit_transform(sentence_embeddings_sub)

np.save(INPUT_DATA / "embeddings/reviews_sentences_UMAP_embeddings.npy", embedding)

# Check the shape of the reduced embedding array
embedding.shape

# ## Clustering: hdbscan

# +
# # Create another low-dim embedding for clustering
# reducer_clustering = umap.UMAP(n_components=50, random_state=1)
# embedding_clustering = reducer_clustering.fit_transform(sentence_embeddings_sub)
# -

# Clustering with hdbscan
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100, min_samples=25, cluster_selection_method="leaf"
)
clusterer.fit(embedding)

len(set(clusterer.labels_))

# # Visualisation

# Prepare dataframe for visualisation
df = reviews_df_subsample.copy()
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]
df["cluster"] = [str(x) for x in clusterer.labels_]

# +
# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
fig = (
    alt.Chart(df[df.cluster != "-1"], width=500, height=500)
    .mark_circle(size=60, opacity=0.05)
    .encode(
        x="x",
        y="y",
        tooltip=["content_sentence", "content_sentence_tokens", "cluster", "appId"],
        color="cluster",
    )
).interactive()

fig
# -

df_cluster = df.copy()

from ast import literal_eval

df_cluster["cluster_tokens"] = df_cluster.content_sentence_tokens.apply(
    lambda x: literal_eval(x)
)

clust_docs = []
for cluster in df.cluster.unique():
    clust_tokens = df_cluster[df_cluster.cluster == cluster].cluster_tokens.to_list()
    clust_tokens = [c for cs in clust_tokens for c in cs]
    clust_docs.append(clust_tokens)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(  # CountVectorizer(
    analyzer="word",
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
)
X = vectorizer.fit_transform(list(clust_docs))

id_to_token = dict(
    zip(list(vectorizer.vocabulary_.values()), list(vectorizer.vocabulary_.keys()))
)

n = 10
clust_words = []
for i in range(X.shape[0]):
    x = X[i, :].todense()
    x = list(np.flip(np.argsort(np.array(x)))[0])[0:n]
    clust_words.append([id_to_token[j] for j in x])

for i, c in enumerate(clust_words):
    print(i, c)

# ## Clustering: k-means and reval

import reval
from reval.best_nclust_cv import FindBestClustCV
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from reval.visualization import plot_metrics

data = make_blobs(1000, 2, centers=2, random_state=42)
plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap="rainbow_r")
plt.show()

X_tr, X_ts, y_tr, y_ts = train_test_split(
    data[0], data[1], test_size=0.30, random_state=42, stratify=data[1]
)

classifier = KNeighborsClassifier()
clustering = AgglomerativeClustering()
findbestclust = FindBestClustCV(
    nfold=2, nclust_range=list(range(2, 11)), s=classifier, c=clustering, nrand=100
)
metrics, nbest = findbestclust.best_nclust(X_tr, iter_cv=10, strat_vect=y_tr)
out = findbestclust.evaluate(X_tr, X_ts, nbest)

plot_metrics(metrics, title="Reval metrics")
