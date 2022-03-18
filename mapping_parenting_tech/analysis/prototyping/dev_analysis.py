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
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Developer analysis
#
# Short routine/s to explore app developers

# %%
import pandas as pd
import json
import numpy as np

from mapping_parenting_tech import PROJECT_DIR, logging
from pathlib import Path

INPUT_PATH = PROJECT_DIR / "inputs/data/play_store"
DATA_PATH = PROJECT_DIR / "outputs/data"

# %% [markdown]
# Load app details

# %%
app_data = pd.read_json(DATA_PATH / "all_app_details.json", orient="index")
app_data.reset_index(inplace=True)
app_data.rename(columns={"index": "app_id"}, inplace=True)

# %%
relevant_apps = pd.read_csv(INPUT_PATH / "relevant_app_ids.csv")
relevant_apps.rename(columns={"appId": "app_id"}, inplace=True)

# %%
relevant_app_details = app_data[app_data.app_id.isin(relevant_apps.app_id)]
relevant_app_details = relevant_app_details.merge(
    relevant_apps, left_on="app_id", right_on="app_id"
)

# %%
relevant_apps.cluster.unique().tolist()

# %%
apps_for_parents = [
    "Tracking babies' rhythms",
    "Helping babies to sleep",
    "Parental support",
    "Pregnancy tracking",
    "Baby photos",
    "Fertility tracking",
]
relevant_apps[relevant_apps["cluster"].isin(apps_for_parents)].shape

# %%
relevant_apps.groupby("cluster").count()

# %%
target_data = relevant_app_details  # [relevant_app_details.cluster == "Literacy - English / ABCs"]

# %%
dev_groups = target_data.groupby(["developer"])

# %%
dev_counts = dev_groups["developer"].count()
dev_counts[dev_counts >= 3].sort_values(ascending=False)

# %%
np.mean(
    target_data.loc[(target_data.developer == "Sago Mini") & (target_data.score > 0)][
        "score"
    ]
)


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

dev_df["av_install_per_app"] = dev_df["dev_installs"] / dev_df["appCount"]
dev_df["app_by_score"] = (
    dev_df["av_install_per_app"] * dev_df["dev_ratings"]
) / 10**6

# %%
dev_df[dev_df["dev_ratings"] > 0].sort_values(by=["app_by_score"], ascending=True).head(
    10
)

# %%
summary_df = dev_df.sort_values(by=["appCount", "dev_ratings"], ascending=False).head(5)

print("Total number of apps:", summary_df["appCount"].sum())
print("Total number of installs:", summary_df["dev_installs"].sum())

summary_df

# %%
top_devs = summary_df.index.tolist()
foo = relevant_app_details.groupby(["developer", "cluster"], as_index=False).agg(
    cCount=("cluster", "count")
)
foo[foo["developer"].isin(top_devs)]
