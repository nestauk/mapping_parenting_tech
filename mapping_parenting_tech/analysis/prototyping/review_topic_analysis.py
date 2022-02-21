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
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# %%
from mapping_parenting_tech.utils import lda_modelling_utils as lmu
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
def load_some_app_reviews(app_ids: list) -> pd.DataFrame:
    """
    Load reviews for a given set of Play Store apps

    Args:
        app_ids: list - a list of app ids whose reviews will be loaded

    Returns:
        Pandas DataFrame

    """

    data_types = {
        "appId": str,
        "content": str,
        "reviewId": str,
        "score": int,
        "thumbsUpCount": int,
    }

    fields = list(data_types.keys())

    reviews_df_list = []
    logging.info("Reading app reviews")
    for app_id in tqdm(app_ids, position=0):
        try:
            review_df = pd.read_csv(
                REVIEWS_DATA / f"{app_id}.csv",
                usecols=fields,
                dtype=data_types,
            )
        except FileNotFoundError:
            logging.info(f"No reviews for {app_id}")
            review_df = []
        reviews_df_list.append(review_df)

    logging.info("Concatenating reviews")
    reviews_df = pd.concat(reviews_df_list)

    del reviews_df_list

    logging.info("Reviews loaded")

    return reviews_df


# %%
# Load model and associated data - step can take a few minutes
model_data = lmu.load_lda_model_data(model_name=MODEL_NAME, folder=TPM_DIR)
mdl = model_data["model"]

# %%
model_data.keys()

# %%
lmu.print_model_info(mdl)

# %%
lmu.make_pyLDAvis(mdl, TPM_DIR / f"{MODEL_NAME}_pyLDAvis.html")

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
relevant_reviews = load_some_app_reviews(relevant_apps.appId.to_list())

# %%
relevant_reviews = relevant_reviews.merge(relevant_apps, on="appId")

# %%
review_topic_details = topic_probability_table.merge(
    relevant_reviews, left_on="id", right_on="reviewId"
).drop(columns=["reviewId"])

# %%
clusters

# %%
cluster_topics = dict()

for cluster in clusters:
    _cluster_topics = dict()
    focus_data = review_topic_details[review_topic_details.cluster == cluster]

    topic_p = []
    for topic_id in range(mdl.k):
        topic_p.append(focus_data[f"topic_{topic_id}"].sum())

    for i, p in enumerate(topic_p):
        _cluster_topics.update({f"topic_{i}": p / sum(topic_p)})

    cluster_topics[cluster] = _cluster_topics

# %%
cluster_topics_df = pd.DataFrame(cluster_topics).T
cluster_topics_df

# %%
cluster_topics_df.to_csv(DATA_DIR / f"{MODEL_NAME}_cluster_topic_probabilities.csv")

# %%
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

    _probs = [p for p in foo if p > _stats["pc"]]
    _ids = topic_probability_table[topic_probability_table[topic].isin(_probs)][
        "id"
    ].to_list()
    _appIds = relevant_reviews[relevant_reviews["reviewId"].isin(_ids)][
        "appId"
    ].to_list()

    top_topic_docs[topic] = {
        "probability": _probs,
        "id": _ids,
        "appId": _appIds,
    }

# %%
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
    Returns a sample
    """

    sample_ids = random.sample(
        sample_frame[f"topic_{topic_number}"]["id"], k=sample_size
    )
    sample_reviews = review_source[review_source["reviewId"].isin(sample_ids)][
        ["appId", "content"]
    ]

    return sample_reviews


# %%
sample_reviews = topic_sample(relevant_reviews, top_topic_docs, 31, 10)

for r in range(len(sample_reviews)):
    print(f"+ {sample_reviews.iloc[r, 0]}\n{sample_reviews.iloc[r, 1]}\n")

# %%
