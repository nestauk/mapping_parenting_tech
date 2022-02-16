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
from mapping_parenting_tech.utils import lda_modelling_utils as lmu
from mapping_parenting_tech import PROJECT_DIR, logging

import pandas as pd
import numpy as np

TPM_DIR = PROJECT_DIR / "outputs/data/tpm"
MODEL_NAME = "play_store_reviews"

# %%
mdl = lmu.load_lda_model_data(model_name=MODEL_NAME, folder=TPM_DIR)

# %%
mdl.keys()

# %%
lmu.print_model_info(mdl["model"])

# %%
mdl["model"].summary()

# %%
