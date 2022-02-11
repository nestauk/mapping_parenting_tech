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

import altair as alt
import pandas as pd
import numpy as np

INPUT_DATA = PROJECT_DIR / "inputs/data/play_store"
OUTPUT_DATA = PROJECT_DIR / "outputs/data"
REVIEWS_DATA = PROJECT_DIR / "outputs/data/app_reviews"

# %%
app_df = pd.read_csv(INPUT_DATA / "relevant_app_ids.csv")
app_ids = app_df["app_id"].to_list()
app_clusters = app_df["cluster"].unique().tolist()


# %%
def load_some_app_reviews(app_ids: list) -> pd.DataFrame:
    """
    Load reviews for a given set of Play Store apps

    Args:
        app_ids: list - a list of app ids whose reviews will be loaded

    Returns:
        Pandas DataFrame

    """

    reviews_df_list = []
    logging.info("Reading app reviews")
    for app_id in tqdm(app_ids, position=0):
        try:
            review_df = pd.read_csv(REVIEWS_DATA / f"{app_id}.csv")
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
app_reviews_df = load_some_app_reviews(app_ids)
app_reviews_df.rename(columns={"appId": "app_id"}, inplace=True)
app_reviews_df["at"] = pd.to_datetime(app_reviews_df["at"], format="%Y-%m-%d")

# %%
target_reviews_df = app_reviews_df.loc[
    (app_reviews_df["at"] >= "2021-02-01")
    # & (app_reviews_df["score"] == 5)
]
app_review_counts = target_reviews_df.groupby("app_id").agg(
    review_count=("reviewId", "count"),
)
app_df = app_df.merge(app_review_counts, left_on="app_id", right_index=True)

# %%
stripplot = (
    alt.Chart(app_df, width=75)
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
app_df.groupby("cluster").agg(
    total_reviews=("review_count", "sum"),
    mean=("review_count", "mean"),
    median=("review_count", np.median),
).sort_values("total_reviews")

# %%
import tomotopy as tp

# %%
raw_reviews = target_reviews_df["content"].to_list()
clean_reviews = tpu.get_preprocessed_documents([str(review) for review in raw_reviews])

# %%
target_reviews_df["cleaned_reviews"] = clean_reviews
target_reviews_df.to_csv(OUTPUT_DATA / "clean_app_reviews.csv")

# %%
# load reviews into a dict: {cluster: [review_1, review_2 ... review_n]
raw_reviews_by_cluster = dict()
for cluster in tqdm(app_clusters):
    raw_reviews_by_cluster[cluster] = target_reviews_df[
        target_reviews_df["app_id"].isin(app_df[app_df["cluster"] == cluster].app_id)
    ].content.to_list()

# %%
len(raw_reviews_by_cluster["Literacy - English / ABCs"])

# %%
# pre-process reviews
clean_reviews_by_cluster = dict()
# for reviews in tqdm(raw_reviews_by_cluster.items()):
#    print(f"Processing {len(reviews)} in {cluster}")
#    clean_reviews_by_cluster[cluster] = tpu.get_preprocessed_documents([str(review) for review in reviews])

clean_reviews_by_cluster["Literacy - English / ABCs"] = tpu.get_preprocessed_documents(
    [str(review) for review in raw_reviews_by_cluster["Literacy - English / ABCs"]]
)

# %%
corpus = tp.utils.Corpus()

# %%
for review in clean_reviews:
    if len(review) > 0:
        doc = tpu.simple_tokenizer(review)
        if len(doc) > 0:
            corpus.add_doc(doc)

# %%
n_topics = 10
n_iters = 1000
model = tp.LDAModel(k=n_topics, corpus=corpus, seed=325)

print(len(model.docs))

# %%
for i in tqdm(range(0, n_iters, 10)):
    model.train(iter=10)

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
n_topic_range = range(1, 25)

perplexity_scores = []
for k in tqdm(n_topic_range):
    _model = tp.LDAModel(k=k, corpus=corpus, seed=357)
    _model.train(iter=n_iters)
    perplexity_scores.append({"N_TOPICS": k, "PERPLEXITY_SCORE": _model.perplexity})

# %%
perplexity_scores = pd.DataFrame(perplexity_scores)
perplexity_scores = perplexity_scores.sort_values("PERPLEXITY_SCORE")

best_n_topic = perplexity_scores.nsmallest(1, "PERPLEXITY_SCORE")["N_TOPICS"].item()
best_p = tp.LDAModel(k=best_n_topic, corpus=corpus, seed=357)
best_p.train(iter=n_iters)

# %%
perplexity_scores.head(15)

# %%
print_topic_words(best_p)

# %%
coherence_scores = []
for k in tqdm(n_topic_range):
    _model = tp.LDAModel(k=k, corpus=corpus, seed=357)
    _model.train(iter=n_iters)
    coherence = tp.coherence.Coherence(_model, coherence="c_v")
    coherence_scores.append({"N_TOPICS": k, "COHERENCE_SCORE": coherence.get_score()})

# %%
coherence_scores = pd.DataFrame(coherence_scores)
coherence_scores = coherence_scores.sort_values("COHERENCE_SCORE", ascending=False)

best_n_topic = coherence_scores.nlargest(1, "COHERENCE_SCORE")["N_TOPICS"].item()
best_c = tp.LDAModel(k=best_n_topic, corpus=corpus, seed=357)
best_c.train(iter=n_iters)

# %%
coherence_scores.head(15)

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
