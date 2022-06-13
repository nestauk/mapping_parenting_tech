from mapping_parenting_tech import logging
from typing import Iterator
from os import PathLike
import json


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


def save_json(data, filepath: PathLike):
    """Saves a dictionary as a json file"""
    with open(filepath, "w") as outfile:
        json.dump(data, outfile, indent=3)


def load_json(filepath: PathLike):
    """Loads a json file as a dictionary"""
    with open(filepath, "r") as infile:
        return json.load(infile)
