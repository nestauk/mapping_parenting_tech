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
#       jupytext_version: 1.13.8
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

5 / 36

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
df["score"] = df.score.astype(str)
df = df.rename(columns={"title": "appTitle"})

df.head(1)

# ## Getting cluster labels
#
# - Find all tokens within a cluster
# - Calculate the cluster centroid
# - Find the token embeddings
# - Find tokens that are closest to the centroid
# - Get the most frequent, closest tokens

# +
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import cdist


def check_most_similar(vect, vects):
    sims = cdist(vect.reshape(1, -1), vects, "cosine")
    return list(np.argsort(sims[0]))


# -

# Load sentence embeddings
token_embeddings = np.load(INPUT_DATA / f"embeddings/{TOKENS_EMBEDDINGS}")
token_table = pd.read_csv(INPUT_DATA / f"embeddings/reviews_tokens.csv")

# +
# token_table.sample(20)
# -

print(token_embeddings.shape)
print(len(token_table))

df_cluster = df.copy()
len(df_cluster)

# Get cluster tokens
df_cluster["cluster_tokens"] = df_cluster.content_sentence_tokens.apply(
    lambda x: literal_eval(x)
)


# +
# # Cluster documents
# clust_docs = []
# for cluster in df.cluster.unique():
#     clust_tokens = df_cluster[df_cluster.cluster == cluster].cluster_tokens.to_list()
#     clust_tokens = [c for cs in clust_tokens for c in cs]
#     clust_docs.append(clust_tokens)

# +
# # Cluster centroids
# cluster = '5'
# clust_df = df_cluster[df_cluster.cluster == cluster]
# clust_indices = clust_df.index.to_list()
# clust_centroid = sentence_embeddings[clust_indices,:].mean(axis=0)

# +
# clust_df.sample(5).content
# -


def topic_keywords(clust_docs, topic_words=True, n=10, Vectorizer=TfidfVectorizer):

    vectorizer = Vectorizer(  # CountVectorizer(
        analyzer="word",
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
    )
    X = vectorizer.fit_transform(list(clust_docs))

    id_to_token = dict(
        zip(list(vectorizer.vocabulary_.values()), list(vectorizer.vocabulary_.keys()))
    )

    clust_words = []
    for i in range(X.shape[0]):
        x = X[i, :].todense()
        if topic_words is None:
            x = list(np.flip(np.argsort(np.array(x)))[0])[0:n]
            clust_words.append([id_to_token[j] for j in x])
        else:

            # Cluster centroids
            cluster = str(i)
            clust_df = df_cluster[df_cluster.cluster == cluster]
            clust_indices = clust_df.index.to_list()
            clust_centroid = sentence_embeddings[clust_indices, :].mean(axis=0)
            topic_words = (
                token_table.iloc[check_most_similar(clust_centroid, token_embeddings)]
                .head(100)
                .token.to_list()
            )

            topic_words_ = [
                t for t in topic_words if t in list(vectorizer.vocabulary_.keys())
            ]
            topic_word_counts = [
                X[i, vectorizer.vocabulary_[token]] for token in topic_words_
            ]
            best_i = np.flip(np.argsort(topic_word_counts))
            top_n = best_i[0:n]
            words = [topic_words_[t] for t in top_n]
            clust_words.append(words)
    # logging.info(f"Generated keywords for {len(cluster_ids)} topics")
    return clust_words


topic_key_terms = topic_keywords(clust_docs, topic_words=True)

topic_frequent_terms = topic_keywords(
    clust_docs, topic_words=True, Vectorizer=CountVectorizer
)

topic_basic_terms = topic_keywords(clust_docs, topic_words=None)

# +
# topic_key_terms[2]
# -

for i, c in enumerate(topic_key_terms):
    print(i, c)

topic_key_terms_dict = {str(i): str(terms) for i, terms in enumerate(topic_key_terms)}
topic_key_terms_dict["-1"] = "noisy points"

df["cluster_keywords"] = df.cluster.apply(lambda x: topic_key_terms_dict[x])

len(df[df.cluster == "-1"]) / len(df)

# +
# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
fig = (
    alt.Chart(
        df[df.cluster != "-1"],
        # df,
        width=800,
        height=800,
    )
    .mark_circle(size=60, opacity=0.33)
    .encode(
        x=alt.X("x", axis=alt.Axis(labels=False), title="dim 1"),
        y=alt.Y("y", axis=alt.Axis(labels=False), title="dim 2"),
        tooltip=["content_sentence", "score", "appTitle", "cluster"],
        color="cluster",
    )
    .configure_axis(grid=False)
).interactive()

fig

# +
from mapping_parenting_tech.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)

driver = google_chrome_driver_setup()
# -

save_altair(fig, "Parenting_app_reviews", driver)

# +
# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
fig = (
    alt.Chart(
        df[df.cluster != "-1"],
        # df,
        width=800,
        height=800,
    )
    .mark_circle(size=60, opacity=0.9)
    .encode(
        x=alt.X("x", axis=alt.Axis(labels=False), title="dim 1"),
        y=alt.Y("y", axis=alt.Axis(labels=False), title="dim 2"),
        tooltip=["content_sentence", "score", "appTitle", "cluster"],
        color="score",
    )
    .configure_axis(grid=False)
)

fig
# -

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
