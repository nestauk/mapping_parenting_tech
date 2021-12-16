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
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# # Prototype multi grain topic modelling

# +
from mapping_parenting_tech import PROJECT_DIR, logging
import tomotopy
import pyLDAvis
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import json
from typing import Iterator
import textacy
from textacy import preprocessing
from textacy.extract import keyterms as kt
from functools import partial
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from toolz import pipe
from gensim import models
import itertools

nlp = spacy.load("en_core_web_sm")

INPUTS_PATH = PROJECT_DIR / "inputs/data/kaggle/reviews.csv"
PARENT_INPUTS_PATH = PROJECT_DIR / "outputs/data"
# -

# # Load and examine data

# ### Test data

reviews_df = pd.read_csv(INPUTS_PATH)

reviews_df.info()

profile = ProfileReport(reviews_df, title=f"Profiling report of {INPUTS_PATH.name}")
with open(
    INPUTS_PATH.parent / f"{INPUTS_PATH.stem}_profiling_report.html", "w"
) as outfile:
    outfile.write(profile.to_html())

# ### Fetched review data

# +
with open(PARENT_INPUTS_PATH / "play_store_ids.json", "r") as infile:
    play_store_ids = json.load(infile)

with open(PARENT_INPUTS_PATH / "play_store_reviews.json", "r") as infile:
    play_store_reviews = json.load(infile)
# -

len(play_store_ids["Parenting apps"])

# Convert dictionary to dataframe
df = pd.DataFrame()
for app_id, app_data in play_store_reviews.items():
    df_app = pd.DataFrame(app_data)
    df_app["appId"] = app_id
    df = df.append(df_app, ignore_index=True)


len(df)

profile = ProfileReport(df, title=f"Profiling report of Play Store apps")
with open(
    PARENT_INPUTS_PATH / f"play_store_reviews_profiling_report.html", "w"
) as outfile:
    outfile.write(profile.to_html())

# # Prepare text data
#
# - Emojis?
# - Normalise unicode (check what it does exactly)
# - Deal with punctuation (but keep sentence structure)
# - Lemmatise
# - Normalise white space
#
# - Tokenise based on phrases within sentences
#
# - Topic modelling
#

reviews_list = df.content.to_list()

reviews_list[999]

# +
### Preprocessing

textacy_preproc_pipeline = preprocessing.make_pipeline(
    partial(preprocessing.replace.emojis, repl=""),
    preprocessing.normalize.unicode,
    preprocessing.normalize.bullet_points,
    preprocessing.remove.punctuation,
    preprocessing.normalize.whitespace,
)


def split_sentences(doc: Doc) -> Span:
    return [sent for sent in doc.sents]


def remove_stopwords(text: Span) -> Span:
    return [token for token in text if token.is_stop is False]


def lemmatise(text: Span) -> Iterator[str]:
    return [token.lemma_ for token in text]


def lemmatise_and_join(text: Span) -> str:
    return " ".join(lemmatise(text))


def remove_empty_sentences(sentences: Iterator[str]) -> Iterator[str]:
    return [sent for sent in sentences if sent != ""]


def clean_sentence(text: Span) -> str:
    """Pipeline to process a single sentence"""
    return pipe(
        text,
        # remove_stopwords,
        lemmatise_and_join,
        textacy_preproc_pipeline,
    )


def review_preprocessor(doc: Doc) -> (Span, Iterator[str]):
    """Text preprocessing pipeline for a review"""
    sents = split_sentences(doc)
    return sents, remove_empty_sentences(
        [clean_sentence(sent).lower() for sent in sents]
    )


### Tokenising


def make_ngram(
    tokenised_corpus: list, n_gram: int = 2, threshold: float = 0.35, min_count: int = 5
) -> list:
    """Extract bigrams from tokenised corpus
    Args:
        tokenised_corpus: List of tokenised corpus
        n_gram: maximum length of n-grams. Defaults to 2 (bigrams)
        threshold:
        min_count: minimum number of token occurrences
    Returns:
        ngrammed_corpus
    """
    tokenised = tokenised_corpus.copy()
    t = 1
    # Loops while the ngram length less / equal than our target
    while t < n_gram:
        phrases = models.Phrases(
            tokenised, min_count=min_count, threshold=threshold, scoring="npmi"
        )
        ngram_phraser = models.phrases.Phraser(phrases)
        tokenised = ngram_phraser[tokenised]
        t += 1
    return list(tokenised), ngram_phraser


# def ngrammer(text, ngram_phraser, nlp=None):
#     return ngram_phraser[process_text(nlp(remove_newline(text)))
# -

txt = df[-df.content.isnull()].content.to_list()

selected_reviews_df = df[-df.content.isnull()]

# +
### Export reviews, split by sentence
# -

df.content.isnull().sum()

reviews_sentences = [split_sentences(doc) for doc in nlp.pipe(txt)]

reviews_sentences_str = [[t.text for t in review] for review in reviews_sentences]

# +
# selected_reviews_df.iloc[21]
# -

clustering_inputs = PROJECT_DIR / "outputs/data/clustering"

selected_reviews_df.reviewId.duplicated().sum()

import pickle

selected_reviews_df.to_csv(
    clustering_inputs / "reviews_for_clustering.csv", index=False
)
pickle.dump(
    reviews_sentences_str,
    open(clustering_inputs / "reviews_for_clustering_sentences.pickle", "wb"),
)

# +
### Tokenise reviews
# -

preprocessed_reviews = [review_preprocessor(doc) for doc in nlp.pipe(txt)]
tokenised_reviews = [[sent.split() for sent in sents] for sents in preprocessed_reviews]

tokenised_reviews_flat = [sent for sents in tokenised_reviews for sent in sents]

tok_ngram, ngram_phraser = make_ngram(tokenised_reviews_flat, n_gram=3)

# +
# def ngrammer(text, ngram_phraser, nlp=None):
#     if nlp is None:
#         nlp = setup_spacy_model(DEF_LANGUAGE_MODEL)
#     return ngram_phraser[process_text(nlp(remove_newline(text)))]
# -

ngram_phraser[tokenised_reviews[0][2]]

# +
## Topic modelling
# -


# +
# for review in tokenised_reviews:
#     ' '.join([ngram_phraser[sent] for sent in review]
# -

ngram_reviews = [
    [ngram_phraser[sent] for sent in review] for review in tokenised_reviews
]

review_corpus = [list(itertools.chain(*ngram_review)) for ngram_review in ngram_reviews]

txt[50]

len(review_corpus)

# +
# review_corpus[50]
# -

review_corpus[0]


# ## Simple topic model

# +
def train_model(mdl, iterations=1000, step=20):
    """Let's train the model"""
    for i in range(0, iterations, step):
        logging.info("Iteration: {:04}, LL per word: {:.4}".format(i, mdl.ll_per_word))
        mdl.train(step)
    logging.info(
        "Iteration: {:04}, LL per word: {:.4}".format(iterations, mdl.ll_per_word)
    )
    mdl.summary()
    return mdl


def get_topic_term_dists(mdl):
    return np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])


def get_doc_topic_dists(mdl):
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    return doc_topic_dists


def make_pyLDAvis(mdl, fpath=PROJECT_DIR / "outputs/data/ldavis_tomotopy.html"):
    topic_term_dists = get_topic_term_dists(mdl)
    doc_topic_dists = get_doc_topic_dists(mdl)
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
    vocab = list(mdl.used_vocabs)
    term_frequency = mdl.used_vocab_freq

    prepared_data = pyLDAvis.prepare(
        topic_term_dists,
        doc_topic_dists,
        doc_lengths,
        vocab,
        term_frequency,
        start_index=0,  # tomotopy starts topic ids with 0, pyLDAvis with 1
        sort_topics=False,  # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
    )
    pyLDAvis.save_html(prepared_data, str(fpath))


def print_model_info(mdl):
    print(
        "Num docs:{}, Num Vocabs:{}, Total Words:{}".format(
            len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
        )
    )
    print("Removed Top words: ", *mdl.removed_top_words)


# -

# Create a corpus instance
corpus = tomotopy.utils.Corpus()
for doc in review_corpus:
    corpus.add_doc(doc)

topic_model = tomotopy.LDAModel(min_df=5, rm_top=40, k=40, corpus=corpus, seed=1111)
topic_model.train(0)
print_model_info(topic_model)

train_model(topic_model, iterations=1000, step=100)

make_pyLDAvis(
    topic_model,
    fpath=PROJECT_DIR / "outputs/models/topic_modelling/topic_modelling.html",
)
