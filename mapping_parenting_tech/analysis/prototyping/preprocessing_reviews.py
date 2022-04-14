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
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# # Prototype multi grain topic modelling

# +
from mapping_parenting_tech import PROJECT_DIR
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
from gensim.models.phrases import FrozenPhrases

nlp = spacy.load("en_core_web_sm")

PARENT_INPUTS_PATH = PROJECT_DIR / "outputs/data"
CLUTERING_OUTPUTS_PATH = PROJECT_DIR / "outputs/data/clustering"
# -

# # Load and examine data

# +
with open(PARENT_INPUTS_PATH / "play_store_ids.json", "r") as infile:
    play_store_ids = json.load(infile)

with open(PARENT_INPUTS_PATH / "play_store_reviews.json", "r") as infile:
    play_store_reviews = json.load(infile)

with open(PARENT_INPUTS_PATH / "play_store_details.json", "r") as infile:
    play_store_details = json.load(infile)
# -

len(play_store_ids["Parenting apps"])

# Convert dictionary to dataframe
reviews_df = pd.DataFrame()
for app_id, app_data in play_store_reviews.items():
    df_app = pd.DataFrame(app_data)
    df_app["appId"] = app_id
    reviews_df = reviews_df.append(df_app, ignore_index=True)


len(reviews_df)

profile = ProfileReport(reviews_df, title=f"Profiling report of Play Store apps")
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
        remove_stopwords,
        lemmatise_and_join,
        textacy_preproc_pipeline,
    )


def review_preprocessor(doc: Doc) -> (Span, Iterator[str]):
    """Text preprocessing pipeline for a review"""
    sents = split_sentences(doc)
    return sents, [clean_sentence(sent).lower() for sent in sents]


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


def ngrammer(text: str, ngram_phraser: FrozenPhrases, nlp: spacy.lang):
    """Coverts free text to preprocessed ngrammed sentences"""
    # Preprocess sentences
    _, preprocessed_sents = review_preprocessor(nlp(text))
    # Split sentences into words
    sents_tokenised_words = [sent.split() for sent in preprocessed_sents]
    # Join up words into ngrams/phrases
    return [ngram_phraser[sent] for sent in sents_tokenised_words]


# -

# ### Export reviews, split by sentence

# Select only reviews that have text and remove duplicates
selected_reviews_df = reviews_df[-reviews_df.content.isnull()].drop_duplicates(
    "reviewId"
)

selected_reviews_df.reviewId.duplicated().sum()

# Select review texts
review_texts = selected_reviews_df.content.to_list()

# Process reviews
processed_reviews = [review_preprocessor(doc) for doc in nlp.pipe(review_texts)]

# Convert split sentences from Span to strings
reviews_sentences = [r[0] for r in processed_reviews]
reviews_sentences_str = [[t.text for t in review] for review in reviews_sentences]


## Tokenise the sentences
# Preprocessed sentences
reviews_sentences_preprocessed = [r[1] for r in processed_reviews]
# Tokenise at the level of words
tokenised_reviews = [
    [sent.split() for sent in sents] for sents in reviews_sentences_preprocessed
]
tokenised_reviews_flat = [sent for sents in tokenised_reviews for sent in sents]
# Find ngrams/phrases
tok_ngram, ngram_phraser = make_ngram(tokenised_reviews_flat, n_gram=3)
# Tokenise sentences at the level of ngrams/phrases
ngram_reviews = [
    [ngram_phraser[sent] for sent in review] for review in tokenised_reviews
]

reviews_processed = selected_reviews_df.copy()
reviews_processed["title"] = reviews_processed["appId"].apply(
    lambda x: play_store_details[x]["title"]
)
reviews_processed["content_sentence"] = reviews_sentences_str
reviews_processed["content_sentence_tokens"] = ngram_reviews
reviews_processed = reviews_processed.drop(["userName", "userImage"], axis=1).explode(
    ["content_sentence", "content_sentence_tokens"]
)


reviews_processed.head(2)


len(reviews_processed)

reviews_processed.to_csv(
    CLUTERING_OUTPUTS_PATH / "reviews_for_clustering_processed.csv", index=False
)

# +
# selected_reviews_df.to_csv(CLUTERING_OUTPUTS_PATH / 'reviews_for_clustering.csv', index=False)
# pickle.dump(reviews_sentences_str, open(CLUTERING_OUTPUTS_PATH / 'reviews_for_clustering_sentences.pickle', 'wb'))
