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
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3812jvsc74a57bd09d0629e00499ccf218c6720a848e8111287e8cbf09d1f93118d5865a19869c30
# ---

# %% [markdown]
# # Developer analysis
#
# Short routine/s to explore app developers

# %%
import pandas as pd
import json

from mapping_parenting_tech import PROJECT_DIR, logging
from pathlib import Path

DATA_PATH = PROJECT_DIR / "outputs/data"

# %% [markdown]
# Load app details

# %%
app_data = pd.read_json(DATA_PATH / "all_app_details.json", orient="index")
app_data.reset_index(inplace=True)
app_data.rename(columns={"index": "appId"}, inplace=True)

# %%
dev_groups = app_data.groupby(["developer"])

# %%
dev_counts = dev_groups["developer"].count()
dev_counts[dev_counts >= 11].sort_values(ascending=False)

# %%
dev_counts[dev_counts["appId"] >= 20].appId.sum()

# %%
dev_installs = dev_groups["minInstalls"].sum()
dev_ratings = dev_groups["score"].mean()

# %%
df_data = {
    "appCount": dev_counts,
    "dev_installs": dev_installs,
    "dev_ratings": dev_ratings,
}
dev_df = pd.concat(df_data, axis=1)
dev_df.sort_values("dev_installs", ascending=False).head(20)

# %%
app_data.columns

# %%
