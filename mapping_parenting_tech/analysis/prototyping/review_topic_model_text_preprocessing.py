# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
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
from mapping_parenting_tech import logging, PROJECT_DIR

import pandas as pd
from tqdm import tqdm

OUTPUT_DIR = PROJECT_DIR / "outputs/data"
INPUT_DIR = PROJECT_DIR / "inputs/data/play_store"
REVIEWS_DATA = OUTPUT_DIR / "app_reviews"

# %%
app_ids = pd.read_csv(INPUT_DIR / "relevant_app_ids.csv")

# %%
app_subset = app_ids[app_ids["cluster"] == "Numeracy development"].app_id.to_list()

# %%
app_reviews = psu.load_some_app_reviews(app_subset)

# %%
foo = app_reviews
foo.shape

# %%
to_process = foo["content"]
processed_reviews = tpu.get_preprocessed_documents([str(doc) for doc in to_process])

# %%
foo["processed_review"] = processed_reviews

# %%
foo[["content", "processed_review"]].sample(15)
