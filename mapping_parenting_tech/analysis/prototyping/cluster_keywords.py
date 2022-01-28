# -*- coding: utf-8 -*-
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
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example how to generate cluster keywords

# %%
import pandas as pd
from mapping_parenting_tech import PROJECT_DIR
from mapping_parenting_tech.utils import text_preprocessing_utils as tpu
import mapping_parenting_tech.analysis.cluster_analysis_utils as cau
import spacy

# Get this file from S3
input_file = PROJECT_DIR / "inputs/data/misc/HLE app reviewing - apps_to_review.csv"

# %% [markdown]
# ## Prerequisites

# %%
# Create a spacy model instance
nlp = spacy.load("en_core_web_sm")

# Get the input data
apps_df = pd.read_csv(input_file)

# %%
print(len(apps_df))
apps_df.head(1)

# %% [markdown]
# ## Generate keywords

# %%
# Clean the description text (might take a minute for 1000s of apps)
clean_descriptions = tpu.get_preprocessed_documents(apps_df.description.to_list())

# %%
# Generate cluster keywords
cluster_keywords = cau.cluster_keywords(
    clean_descriptions, apps_df.cluster_purpose.to_list()
)

# %%
cluster_keywords

# %%
