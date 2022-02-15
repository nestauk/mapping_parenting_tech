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
#     name: python3812jvsc74a57bd09d0629e00499ccf218c6720a848e8111287e8cbf09d1f93118d5865a19869c30
# ---

# %%
from mapping_parenting_tech.utils import (
    play_store_utils as psu,
    text_preprocessing_utils as tpu,
)
from mapping_parenting_tech import PROJECT_DIR, logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import tomotopy as tp
import altair as alt
import pandas as pd
import numpy as np
import csv

INPUT_DATA = PROJECT_DIR / "inputs/data/play_store"
OUTPUT_DATA = PROJECT_DIR / "outputs/data"
REVIEWS_DATA = PROJECT_DIR / "outputs/data/app_reviews"

# %%
# Read in the ids of the relevant apps (as manually id'd by Nesta staff)
app_info = pd.read_csv(INPUT_DATA / "relevant_app_ids.csv")
app_clusters = app_info["cluster"].unique().tolist()


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
        "app_id": str,
        "content": str,
        "score": int,
        "thumbsUpCount": int,
        "reviewCreatedVersion": str,
        "replyContent": str,
        "reviewId": str,
    }

    reviews_df_list = []
    logging.info("Reading app reviews")
    for app_id in tqdm(app_ids, position=0):
        try:
            review_df = pd.read_csv(
                REVIEWS_DATA / f"{app_id}.csv",
                dtype=data_types,
                parse_dates=["at", "repliedAt"],
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
# load the reviews for the relevant apps
app_reviews = load_some_app_reviews(app_info["app_id"].to_list())

# rename the appId column...
app_reviews.rename(columns={"appId": "app_id"}, inplace=True)

# add the cluster that each add is in to the review
app_reviews = app_reviews.merge(app_info, left_on="app_id", right_on="app_id")

# what have we got...?
app_reviews.shape

# %%
# get a subset of reviews - here it's those written in the last year about apps in the
# cluster 'Numeracy development'
target_reviews = app_reviews.loc[
    (app_reviews["at"] >= "2021-02-01")
    # & (app_reviews["cluster"] == "Numeracy development")
]
target_reviews.shape

# %%
# if we want to visualise the distribution of the reviews, group the apps by their id and cluster
# and count the number of reviews

app_review_counts = target_reviews.groupby(["app_id", "cluster"]).agg(
    review_count=("reviewId", "count"),
)

# reset the index so we just have one
app_review_counts.reset_index(inplace=True)

# %%
# plot the data

stripplot = (
    alt.Chart(app_review_counts, width=75)
    .mark_circle(size=20)
    .encode(
        x=alt.X(
            "jitter:Q",
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y(
            "review_count:Q",
            # scale=alt.Scale(range=(0,10000)),
        ),
        color=alt.Color("cluster:N", legend=None),
        column=alt.Column(
            "cluster:N",
            header=alt.Header(
                labelAngle=-90,
                titleOrient="top",
                labelOrient="bottom",
                labelAlign="right",
                labelPadding=3,
            ),
        ),
        tooltip=["app_id"],
    )
    .transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    )
    .configure_facet(spacing=0)
    .configure_view(stroke=None)
)

stripplot


# %%
# Now let's look at the distributions of reviews for each cluster

app_review_counts.groupby("cluster").agg(
    total_reviews=("review_count", "sum"),
    mean=("review_count", "mean"),
    median=("review_count", np.median),
).sort_values("total_reviews")

# %% [markdown]
# ## Topic modelling

# %%
# get a subset of reviews

tp_reviews = target_reviews[target_reviews.cluster == "Drawing and colouring"]
tp_reviews.shape

# %%
# get a preprocess the review text ready for topic modelling
# this will take some time for 10,000s of reviews...

tp_reviews.loc[:, "preprocessed_review"] = tpu.get_preprocessed_documents(
    [str(review) for review in tp_reviews["content"].to_list()]
)

# %%
# get reviews that have 5 tokens or more

tp_reviews = tp_reviews[tp_reviews.preprocessed_review.str.count(" ") + 1 > 4]
tp_reviews.shape

# %%
# save processed reviews

tp_reviews.to_csv(OUTPUT_DATA / "pre_processed_drawing_app_reviews.csv")

# %%
# load reviews into a dict: {cluster: [review_1, review_2 ... review_n]

raw_reviews_by_cluster = dict()
for cluster in tqdm(app_clusters):
    raw_reviews_by_cluster[cluster] = tp_reviews[
        tp_reviews.cluster == cluster
    ].content.to_list()

# %%
# set up a Corpus which we'll load our documents (reviews) into

corpus = tp.utils.Corpus()

# %%
# tokenize the reviews and add to corpus

for review in tp_reviews.preprocessed_review.to_list():
    doc = tpu.simple_tokenizer(review)
    corpus.add_doc(doc)

# %%
# initialise the model and confirm the number of documents in it

model = tp.LDAModel(k=n_topics, corpus=corpus)

print(len(model.docs))


# %%
# Set up a function to help explore the variables we need to train the model - at what point does log likelihood and/or coherence converge?


def test_model(
    corpus,
    n_topics: int = 20,
    max_iters: int = 1000,
    iter_step: int = 50,
    seed: int = None,
) -> int:

    model = tp.LDAModel(k=n_topics, corpus=corpus)
    model_scores = []

    for i in range(0, max_iters, iter_step):
        model.train(iter=iter_step)
        ll_per_word = model.ll_per_word
        c_v = tp.coherence.Coherence(model, coherence="c_v")
        model_scores.append(
            {
                "N_TOPICS": n_topics,
                "N_ITER": i,
                "LOG_LIKELIHOOD": ll_per_word,
                "COHERENCE": c_v.get_score(),
                "SEED": seed,
            }
        )

    model_scores = pd.DataFrame(model_scores)

    return model_scores[model_scores["COHERENCE"] == model_scores["COHERENCE"].max()]


# %%
iters = []

for j in tqdm(range(5, 25), position=0):
    for i in range(100, 1500, 250):
        iters.append(test_model(corpus, n_topics=j, max_iters=1050, seed=250))

model_scores = pd.concat(iters)

# %%
# model_scores = model_scores.sort_values("COHERENCE", ascending=False)
# model_scores.head()
model_scores.plot.scatter("N_ITER", "LOG_LIKELIHOOD")

# %%
print(
    f"Number of unique words: {len(model.used_vocabs):,}",
    f"\nTotal number of tokens: {model.num_words:,}",
)


# %%
def print_topic_words(model, top_n=5):
    for k in range(model.k):
        top_words = model.get_topic_words(topic_id=k, top_n=top_n)
        top_words = [f"{tup[0]} ({tup[1]:.04f}%)" for tup in top_words]
        print(f"Topic #{k}:", f"\n+ {', '.join(top_words)}")


# %%
print_topic_words(model)

# %%
word_dist = model.get_topic_word_dist(topic_id=2)
pd.Series(word_dist, index=model.used_vocabs).sort_values(ascending=False).head(15)

# %%
topic_sizes = model.get_count_by_topics()

print("Number of words per topic:")
for k in range(0, 5):
    print(f"+ Topic #{k}: {topic_sizes[k]:,} words")

# %%
print("Topic proportion across the corpus:")
for k in range(0, 5):
    print(f"+ Topic #{k}: {topic_sizes[k] / model.num_words:0.2f}%")

# %%
print_topic_words(best_p)

# %%
n_topic_range = range(5, 35)
n_iters = 1050
coherence_scores = []
for k in tqdm(n_topic_range):
    _model = tp.LDAModel(k=k, corpus=corpus, seed=250)
    _model.train(iter=n_iters)
    coherence = tp.coherence.Coherence(_model, coherence="c_v")
    coherence_scores.append({"N_TOPICS": k, "COHERENCE_SCORE": coherence.get_score()})

# %%
coherence_scores = pd.DataFrame(coherence_scores)
coherence_scores = coherence_scores.sort_values("COHERENCE_SCORE", ascending=False)

# best_n_topic = coherence_scores.nlargest(1, "COHERENCE_SCORE")["N_TOPICS"].item()
# best_c = tp.LDAModel(k=best_n_topic, corpus=corpus, seed=357)
# best_c.train(iter=n_iters)

# %%
coherence_scores.head(15)

# %%
coherence_scores.plot.scatter("N_TOPICS", "COHERENCE_SCORE")

# %%
print_topic_words(best_c)


# %%
def plot_topic_proportions(tm, model_name="", top_n=5):
    topic_proportions = tm.get_count_by_topics() / tm.num_words
    top_words = []
    for topic in range(tm.k):
        words = tm.get_topic_words(topic, top_n=top_n)
        words = f"Topic #{topic}: " + ", ".join([w[0] for w in words])
        top_words.append(words)

    to_plot = pd.Series(topic_proportions, index=top_words)
    to_plot = to_plot.sort_values()
    to_plot.plot.barh(
        figsize=(15, 15), title=f"Topic Proportions for {model_name}", xlabel="Topic"
    )


plot_topic_proportions(best_c, model_name="Best Coherence Model")

# %%
app_reviews_df["review_length"] = app_reviews_df["content"].str.count(" ") + 1

# %%
r_len_df = app_reviews_df.groupby("app_id").agg(
    av_review_len=("review_length", "mean"),
    max_review=("review_length", "max"),
    min_review=("review_length", "min"),
    median_rev=("review_length", "median"),
)

# %%
for cm in r_len_df.columns:
    print(cm, "\t", sum(r_len_df[cm]) / len(r_len_df[cm]))

# %%
r_len_df.describe()

# %%
app_reviews_df["content"].sample(25)

# %%
# load all of the preprocessed reviews

data_types = {
    "app_id": str,
    "content": str,
    "score": int,
    "thumbsUpCount": int,
    "reviewCreatedVersion": str,
    "replyContent": str,
    "reviewId": str,
    "preprocessed_review": str,
}

fields = list(data_types.keys())
fields.extend(["at", "repliedAt"])

tp_reviews = pd.read_csv(
    OUTPUT_DATA / "tpm/foo.csv",
    usecols=fields,
    dtype=data_types,
    parse_dates=["at", "repliedAt"],
)

tp_reviews.shape

# %%
# ...and get those with 5 tokens or more
tp_reviews = tp_reviews[tp_reviews.preprocessed_review.str.count(" ") + 1 > 4]

tp_reviews.shape

# %%
# set up a Corpus which we'll load our documents (reviews) into

corpus = tp.utils.Corpus()

# %%
# tokenize the reviews and add to corpus

for review in tp_reviews.preprocessed_review.to_list():
    doc = tpu.simple_tokenizer(review)
    corpus.add_doc(doc)


# %%
# Set up a function to help explore the variables we need to train the model - at what point does log likelihood and/or coherence converge?


def test_model(
    corpus,
    n_topics: int = 20,
    max_iters: int = 1000,
) -> int:

    model = tp.LDAModel(k=n_topics, corpus=corpus, seed=250)

    print("+ + Started training at ", datetime.now().time(), flush=True)

    model.train(max_iters)

    model_score = {
        "N_TOPICS": n_topics,
        "N_ITER": i,
        "LOG_LIKELIHOOD": model.ll_per_word,
        "COHERENCE": tp.coherence.Coherence(model, coherence="c_v").get_score(),
    }

    print("+ + Completed training at ", datetime.now().time(), flush=True)

    return model_score


# %%
from_iters = 1250
from_k = 31

for i in range(from_iters, 2100, 250):
    print(f"Testing {i} iterations", flush=True)
    for j in range(5, 55):
        if (i > from_iters) or ((i == from_iters) and (j > from_k)):
            print(f"+ Testing {j} topics", flush=True)
            model_score = test_model(
                corpus,
                n_topics=j,
                max_iters=i,
            )

            with open(OUTPUT_DATA / "tpm/model_scores.csv", mode="a", newline="") as f:
                csv_field_names = list(model_score.keys())
                writer = csv.DictWriter(f, fieldnames=csv_field_names)
                writer.writerow(model_score)

# %%
test_topics = [52]

from_iters = 900
from_k = 0

for i in range(from_iters, 1250, 50):
    print(f"Testing {i} iterations", flush=True)
    for j in test_topics:
        if (i > from_iters) or ((i == from_iters) and (j > from_k)):
            print(f"+ Testing {j} topics", flush=True)
            model_score = test_model(
                corpus,
                n_topics=j,
                max_iters=i,
            )

            with open(OUTPUT_DATA / "tpm/model_scores.csv", mode="a", newline="") as f:
                csv_field_names = list(model_score.keys())
                writer = csv.DictWriter(f, fieldnames=csv_field_names)
                writer.writerow(model_score)

# %%
model_scores = pd.read_csv(OUTPUT_DATA / "tpm/model_scores.csv")

best_score = model_scores[model_scores["COHERENCE"] == model_scores["COHERENCE"].max()]

n_topics = int(best_score["N_TOPICS"])
n_iter = int(best_score["N_ITER"])

# %%
model_scores.plot.scatter("N_ITER", "COHERENCE")

# %%
model_scores.plot.scatter("N_TOPICS", "COHERENCE")

# %%
print(f"Optimum number of topics: {n_topics}")
print(f"Optimum number of training iterations: {n_iter}")

# %%
final_model = tp.LDAModel(k=n_topics, corpus=corpus, seed=250)

for i in tqdm(range(0, n_iter, 10)):
    final_model.train(10)

# %%
print(tp.coherence.Coherence(final_model, coherence="c_v").get_score())

# %%
from mapping_parenting_tech.utils import lda_modelling_utils as lmu

# %%
MODEL_NAME = "play_store_reviews"

lmu.save_lda_model_data(
    MODEL_NAME,
    OUTPUT_DATA / "tpm",
    final_model,
    list(tp_reviews["reviewId"]),
)

# %%
