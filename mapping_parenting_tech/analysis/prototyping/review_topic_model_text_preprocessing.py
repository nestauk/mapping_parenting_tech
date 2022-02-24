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
from mapping_parenting_tech.utils import play_store_utils as psu
from mapping_parenting_tech import logging, PROJECT_DIR

import pandas as pd
from tqdm import tqdm

OUTPUT_DIR = PROJECT_DIR / "outputs/data"
INPUT_DIR = PROJECT_DIR / "inputs/data/play_store"
REVIEWS_DATA = OUTPUT_DIR / "app_reviews"


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
            logging.info(f"No reviews found for {app_id}")
            review_df = []
        reviews_df_list.append(review_df)

    logging.info("Concatenating reviews")
    reviews_df = pd.concat(reviews_df_list)
    del reviews_df_list
    logging.info("Reviews loaded")

    return reviews_df


# %%
app_ids = pd.read_csv(INPUT_DIR / "relevant_app_ids.csv")

# %%
app_subset = app_ids[app_ids["cluster"] == "Numeracy development"].app_id.to_list()

# %%
app_reviews = load_some_app_reviews(app_subset)

# %%
foo = app_reviews
foo.shape

# %%
to_process = foo["content"]
processed_reviews = get_preprocessed_documents([str(doc) for doc in to_process])

# %%
foo["processed_review"] = processed_reviews

# %%
foo[["content", "processed_review"]].sample(15)

# %% [markdown]
# Following code is from text_proprocessing_utils...

# %%
"""
mapping_parenting_tech.utils.text_preprocessing_utils.py

Module for preprocessing (eg, cleaning) text data

"""
from functools import partial
from typing import Iterator
from toolz import pipe
import textacy
from textacy import preprocessing
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
import re

DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_PARTS_OF_SPEECH = {"NOUN"}
DEFAULT_REMOVE_NERS = {
    "ORG",
    "DATE",
    "QUANTITY",
    "PERSON",
    "CARDINAL",
    "ORDINAL",
    "GPE",
    "LOC",
}


textacy_preproc_pipeline = preprocessing.make_pipeline(
    preprocessing.remove.html_tags,
    partial(preprocessing.replace.emojis, repl=""),
    preprocessing.normalize.unicode,
    preprocessing.normalize.bullet_points,
    preprocessing.normalize.whitespace,
)


def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z ]+", "", text)


def remove_newline(text: str) -> str:
    """Removes new line symbol '\n'"""
    return re.sub("\n", " ", text)


def lowercase(text: str) -> str:
    """Transforms text to lower case"""
    return text.lower()


def clean_text(text):
    """
    Pipeline to transform text to lowercase, and remove new lines,
    html tags, emojis, bullet points and whitespaces
    """
    return pipe(
        text,
        lowercase,
        remove_newline,
        remove_non_alphanumeric,
        textacy_preproc_pipeline,
    )


def split_sentences(doc: Doc) -> Span:
    """Splits sentences using spacy"""
    return [sent for sent in doc.sents]


def remove_stopwords(text: Span) -> Span:
    """Removes stopwords"""
    return [token for token in text if token.is_stop is False]


def remove_features(text: Span) -> Span:
    """Removes punctuation, urls and spaces"""
    return [
        token
        for token in text
        if (token.is_punct is False)
        & (token.like_url is False)
        & (token.is_space is False)
    ]


def remove_ners(text: Span, ners=DEFAULT_REMOVE_NERS) -> Span:
    """Removes specified named entities"""
    if ners is None:
        return [token for token in text]
    else:
        return [token for token in text if (token.ent_type_ not in ners)]


def keep_pos(text: Span, pos=DEFAULT_PARTS_OF_SPEECH) -> Span:
    """Keeps specified parts-of-speech"""
    if pos is None:
        return [token for token in text]
    else:
        return [token for token in text if (token.pos_ in pos)]


def lemmatise(text: Span) -> Iterator[str]:
    """Lemmatises text using spacy"""
    return [token.lemma_ for token in text]


def lemmatise_and_join(text: Span) -> str:
    """Lemmatises text and creates a string object"""
    return " ".join(lemmatise(text))


def normalise_spaces(text: str) -> str:
    """Normalise spaces of arbitrary length to single spaces"""
    return " ".join([s.strip() for s in text.split(" ")])


def remove_empty_sentences(sentences: Iterator[str]) -> Iterator[str]:
    """Removes empty sentences"""
    return [sent for sent in sentences if sent != ""]


def clean_sentence(text: Span) -> str:
    """Pipeline to process a single sentence"""
    return pipe(
        text,
        remove_stopwords,
        remove_features,
        keep_pos,
        remove_ners,
        lemmatise_and_join,
        normalise_spaces,
    )


def text_preprocessor(doc: Doc) -> (Span, Iterator[str]):
    """
    Text preprocessing for a single text document: Splits the
    document into sentences and processes each sentence separately.

    Args:
        doc: spacy document

    Returns:
        list of split, non-formatted sentences in spacy format (inputs)
        list of processed sentences (outputs)

    """
    # Split sentences
    sents = split_sentences(doc)
    # Return both the non-formatted and formatted split sentences
    return sents, [clean_sentence(sent) for sent in sents]


def preprocess_documents(texts: Iterator[str], nlp=None):
    """
    Loads a spacy language model and preprocesses a collection of text documents
    """
    # Load a spacy model
    if nlp is None:
        nlp = spacy.load(DEFAULT_SPACY_MODEL)
    # Basic text preprocessing
    texts = [clean_text(text) for text in texts]
    # Further preprocessing with spacy (lemmatisation, lingustic features)
    return [text_preprocessor(doc) for doc in nlp.pipe(texts)]


def get_preprocessed_documents(texts: Iterator[str], nlp=None):
    """
    Preprocesses text into cleaned sentences and joins them up
    """
    logging.info("Beginning pre-processing")
    preprocessed_texts = preprocess_documents(texts, nlp)
    return [" ".join(doc[1]) for doc in preprocessed_texts]


def simple_tokenizer(text: str) -> Iterator[str]:
    return [token.strip() for token in text.split(" ") if len(token) > 0]


# %%
