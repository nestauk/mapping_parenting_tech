from mapping_parenting_tech import logging
from typing import Iterator
from os import PathLike


def save_text_items(list_of_terms: Iterator[str], filepath: PathLike):
    """Writes a text file with comma-separated text terms"""
    with open(filepath, "w") as outfile:
        outfile.write(", ".join(list_of_terms))
    logging.info(f"Saved {len(list_of_terms)} terms in {filepath}")


def read_text_items(filepath: PathLike) -> Iterator[str]:
    """Reads in a text file with comma-separated text terms"""
    with open(filepath, "r") as infile:
        txt = infile.read()
    list_of_terms = txt.split(", ")
    logging.info(f"Loaded {len(list_of_terms)} text items from {filepath}")
    return list_of_terms
