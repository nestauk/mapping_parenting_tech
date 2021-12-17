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
# # Prototype multi grain topic modelling

# %%
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
import pickle
from gensim.models.phrases import FrozenPhrases

nlp = spacy.load("en_core_web_sm")

PARENT_INPUTS_PATH = PROJECT_DIR / "outputs/data"
CLUTERING_OUTPUTS_PATH = PROJECT_DIR / "outputs/data/clustering"

# %% [markdown]
# # Load and examine data

# %%
from ast import literal_eval

# %%
INPUT_DATA = PROJECT_DIR / "outputs/data/clustering"
REVIEWS_TABLE = "reviews_for_clustering_processed.csv"

# %%
# Load in the table
reviews_df = pd.read_csv(INPUT_DATA / REVIEWS_TABLE)

# %%
reviews_df.content_sentence_tokens.apply(lambda x: literal_eval(x))

# %%
review_corpus = [list(itertools.chain(*ngram_review)) for ngram_review in ngram_reviews]


# %% [markdown]
# ## Simple topic model

# %%
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


# %%
# Create a corpus instance
corpus = tomotopy.utils.Corpus()
for doc in review_corpus:
    corpus.add_doc(doc)

# %%
topic_model = tomotopy.LDAModel(min_df=5, rm_top=40, k=40, corpus=corpus, seed=1111)
topic_model.train(0)
print_model_info(topic_model)

# %%
train_model(topic_model, iterations=1000, step=100)

# %%
make_pyLDAvis(
    topic_model,
    fpath=PROJECT_DIR / "outputs/models/topic_modelling/topic_modelling.html",
)

# %%
