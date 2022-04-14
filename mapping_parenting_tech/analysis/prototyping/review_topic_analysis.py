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
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# %%
from mapping_parenting_tech.utils import lda_modelling_utils as lmu
from mapping_parenting_tech.utils import play_store_utils as psu
from mapping_parenting_tech import PROJECT_DIR, logging
from tqdm import tqdm

import pandas as pd
import numpy as np

import random

TPM_DIR = PROJECT_DIR / "outputs/data/tpm"
MODEL_NAME = "play_store_reviews"
DATA_DIR = PROJECT_DIR / "outputs/data/"
REVIEWS_DATA = DATA_DIR / "app_reviews"
INPUT_DIR = PROJECT_DIR / "inputs/data/play_store"

# %%
# Load model and associated data - step can take a few minutes
model_data = lmu.load_lda_model_data(model_name=MODEL_NAME, folder=TPM_DIR)
mdl = model_data["model"]

# %%
model_data.keys()

# %%
lmu.print_model_info(mdl)

# %%
# Run once or as required, but suggest keep commented by default
# lmu.make_pyLDAvis(mdl, TPM_DIR / f"{MODEL_NAME}_pyLDAvis.html")

# %%
doc_topic_probabilities = lmu.get_document_topic_distributions(mdl)

topic_probability_table = lmu.create_document_topic_probability_table(
    model_data["document_ids"], model_data["model"], doc_topic_probabilities
)

# %%
topic_probability_table

# %%
# Load the app ids for the relevant apps
relevant_apps = pd.read_csv(INPUT_DIR / "relevant_app_ids.csv")

# Extract the clusters
clusters = relevant_apps.cluster.unique().tolist()

# Load the reviews for these apps
relevant_reviews = psu.load_some_app_reviews(relevant_apps.appId.to_list())

# %%
# add the app details, including cluster info, to the reviews dataframe

relevant_reviews = relevant_reviews.merge(relevant_apps, on="appId")

# %%
# merge the reviews onto the topic_probability_table such that we have a consolidated table with the review id, topic probabilities and review content all in one place

review_topic_details = topic_probability_table.merge(
    relevant_reviews, left_on="id", right_on="reviewId"
).drop(columns=["reviewId"])

# %%
# quick reminder of the clusters

clusters

# %%
# calculate the topic probabilities for each cluster
# each review is associated with one app, which is in one cluster, so we can combine those numbers
# thus for each cluster, we calculate the probability a topic is associated with that cluster based only on the reviews about apps in that cluster

cluster_topics = dict()

for cluster in clusters:
    _cluster_topics = dict()

    # get all the reviews for a cluster
    focus_data = review_topic_details[review_topic_details.cluster == cluster]

    # for each topic, sum probabilities of all the reviews w/in this cluster
    topic_p = []
    for topic_id in range(mdl.k):
        topic_p.append(focus_data[f"topic_{topic_id}"].sum())

    # add the final probability for each cluster/topic pair to the dict by dividing the probability for each topic by the total probability for the cluster
    # thus, the sums for all the topic probabilities across a cluster will sum to 1
    for i, p in enumerate(topic_p):
        _cluster_topics.update({f"topic_{i}": p / sum(topic_p)})

    cluster_topics[cluster] = _cluster_topics

# %%
cluster_topics_df = pd.DataFrame(cluster_topics).T
cluster_topics_df

# %%
cluster_topics_df.to_csv(DATA_DIR / f"{MODEL_NAME}_cluster_topic_probabilities.csv")

# %%
# print the top (highest probability) and save the top 10 reviews for each cluster into a dict

cluster_top_ten_summary = dict()

for i in range(mdl.k):
    topic = f"topic_{i}"
    print("\n>>>>>>>>>>>>>>>>>")
    print(topic)

    best_cluster = cluster_topics_df[topic].idxmax(), cluster_topics_df[topic].max()
    print(f"+ {best_cluster}")

    top_ten_review_ids = (
        topic_probability_table.sort_values(topic, ascending=False)
        .head(10)
        .id.to_list()
    )
    top_ten_review_probs = (
        topic_probability_table.sort_values(topic, ascending=False)
        .head(10)[topic]
        .to_list()
    )
    top_ten_reviews = relevant_reviews[
        relevant_reviews["reviewId"].isin(top_ten_review_ids)
    ].content.to_list()

    print("\n-----")
    print(top_ten_reviews[0])

    cluster_top_ten_summary[topic] = {
        "best_cluster": best_cluster,
        "reviewId": top_ten_review_ids,
        "p": top_ten_review_probs,
        "review": top_ten_reviews,
    }

# %% [markdown]
# ## Review samples
#
# Fetch the ids of the documents whose probabilities are in the 99th percentile of probabilities for each topic. This will be around 3,000 reviews for each topic.
#
# `top_topic_docs` is a dict with the following format:
# `[topic id]: {
# "probabilities": [list of document probabilities],
# "ids": [list of document ids]
# }`
#
# The position of the document probability in `probabilities` aligns with the position of the review id in `ids`

# %%
relevant_reviews.columns

# %%
topic_probability_stats = dict()
top_topic_docs = dict()

for i in tqdm(range(mdl.k)):
    topic = f"topic_{i}"

    foo = topic_probability_table[topic].to_list()

    _stats = {
        "median": np.median(foo),
        "max": np.max(foo),
        "min": np.min(foo),
        "pc": np.percentile(foo, 99),
    }

    topic_probability_stats[topic] = _stats

    _review_data = topic_probability_table[
        topic_probability_table[topic] > _stats["pc"]
    ][["id", topic]]

    _review_data = (
        _review_data.merge(
            relevant_reviews[["reviewId", "appId", "score", "content"]],
            how="left",
            left_on="id",
            right_on="reviewId",
        )
        .drop(columns=["reviewId"])
        .rename(columns={topic: "probability"})
    )

    top_topic_docs[topic] = _review_data.to_dict(orient="list")

# %%
# calculate the average review score for each topic, based on the top reviews, as saved in `top_topic_docs`

topic_scores = []

for i in tqdm(range(mdl.k)):
    topic = f"topic_{i}"
    _topic_scores = relevant_reviews[
        relevant_reviews["reviewId"].isin(top_topic_docs[topic]["id"])
    ].score

    topic_scores.append({"topic": topic, "topic_score": np.mean(_topic_scores)})

topic_scores = pd.DataFrame(topic_scores)
topic_scores.sort_values("topic_score", ascending=False)

# %%
# print 5 random reviews within a topic from `top_topic_docs` - prints the app that the review is about and the review itself

for topic, reviews in top_topic_docs.items():
    print(f"*************************\n{topic.upper()}")
    review_sample_ids = random.sample(reviews["id"], k=5)
    review_sample_contents = relevant_reviews[
        relevant_reviews["reviewId"].isin(review_sample_ids)
    ][["appId", "content"]]

    for r in range(len(review_sample_contents)):
        print(
            f"+ {review_sample_contents.iloc[r, 0]}\n{review_sample_contents.iloc[r, 1]}\n"
        )


# %%
def topic_sample(
    review_source: pd.DataFrame,
    sample_frame: dict,
    topic_number: int,
    sample_size: int = 5,
) -> list:
    """
    Returns a sample of reviews for a given topic

    Args:
        review_source - Panda's dataframe: a dataframe with all of the relevant reviews
        sample_frame - dict: a dict containing the reviews that you'd like to sample
        topic_number - int: the topic number that you're sampling
        sample_size - int: the number of sample reviews that you'd like returned
    """

    sample_ids = random.sample(
        sample_frame[f"topic_{topic_number}"]["id"], k=sample_size
    )
    sample_reviews = review_source[review_source["reviewId"].isin(sample_ids)][
        ["appId", "content"]
    ]

    return sample_reviews


# %%
sample_reviews = topic_sample(relevant_reviews, top_topic_docs, 50, 10)

for r in range(len(sample_reviews)):
    print(f"+ {sample_reviews.iloc[r, 0]}\n{sample_reviews.iloc[r, 1]}\n")

# %%
# convert the top_topic_docs dict into a dataframe to make it more amenable to searching

top_topic_docs_df = pd.DataFrame.from_dict(top_topic_docs, orient="index")
top_topic_docs_df = (
    top_topic_docs_df.explode(top_topic_docs_df.columns.to_list())
    .reset_index()
    .rename(columns={"index": "topic"})
)

# %%
# search for a needle in the haystack...
# looking for a search term (`needle`) within reviews of the top topics
# this will return a sample of 5 reviews that contain the search term (I think it will throw an error if fewer than 5 reviews contain that term...)

needle = "cry"

haystack = top_topic_docs_df["content"].str.contains(needle, False)

targets = haystack[haystack == True].index.to_list()

for r in top_topic_docs_df.iloc[targets][["appId", "content"]].sample(5).iterrows():
    print(f"+ {r[1][0]}")
    print(r[1][1], "\n")


# %%
# find reviews within a given topic (`topic_num`) of a particular score (`target_score`)

topic_num = "50"
target_score = 1

scored_topic_reviews = top_topic_docs_df.loc[
    (top_topic_docs_df["score"] == target_score)
    & (top_topic_docs_df["topic"] == f"topic_{topic_num}")
]

# %%
# visualise a sample of the topics
sample_count = 10

scored_topic_reviews.sample(sample_count)[["appId", "content"]].to_dict(
    orient="records"
)

# %%
# show the best / worst apps that people are talking about
scored_topic_reviews.groupby("appId").count().sort_values(
    by="id", ascending=False
).head(10)
