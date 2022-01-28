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
    preprocessed_texts = preprocess_documents(texts, nlp)
    return [" ".join(doc[1]) for doc in preprocessed_texts]


def simple_tokenizer(text: str) -> Iterator[str]:
    return [token.strip() for token in text.split(" ") if len(token) > 0]
