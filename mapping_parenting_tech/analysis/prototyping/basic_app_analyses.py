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
#     display_name: Python 3.8.0 ('mapping_parenting_tech')
#     language: python
#     name: python3
# ---

# %%
from mapping_parenting_tech.utils import play_store_utils as psu
from mapping_parenting_tech import logging, PROJECT_DIR

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

OUTPUT_DIR = PROJECT_DIR / "outputs/data"
INPUT_DIR = PROJECT_DIR / "inputs/data/play_store"

# %%
app_id_clusters = pd.read_csv(INPUT_DIR / "relevant_app_ids.csv")
app_details = (
    pd.DataFrame(psu.load_all_app_details())
    .T.reset_index()
    .rename(columns={"index": "appId"})
)
app_details = app_details.merge(app_id_clusters, on="appId")
app_id = app_id_clusters.appId.to_list()
app_clusters = app_id_clusters.cluster.unique().tolist()
app_details["score"].replace(0, np.NaN, inplace=True)

# %%
df = app_details.groupby("cluster", as_index=False).agg(
    sumMinInstalls=("minInstalls", np.sum),
    medianInstalls=("minInstalls", np.median),
    appCount=("appId", np.count_nonzero),
    meanRating=("score", np.mean),
    ratingSD=("score", np.std),
    medianRating=("score", np.median),
)
df["installsPerApp"] = df["sumMinInstalls"] / df["appCount"]

# round installs per app to nearest 1,000, sort and then format nicely with commas
df.installsPerApp = np.round(df.installsPerApp.to_list(), -3)

df.sort_values("installsPerApp", ascending=False, inplace=True)

df["installsPerApp"] = df["installsPerApp"].apply("{:,}".format)

# %%
df

# %%
df.sort_values("meanRating").plot.bar(x="cluster", y="meanRating")

# %%
app_details["compound_score"] = app_details["minInstalls"] * app_details["score"]

# %%
top_number = 25
_top_apps = []

for cluster in app_clusters:
    _top_apps.append(
        app_details[app_details.cluster == cluster]
        .sort_values(["minInstalls", "score"], ascending=False)
        .head(top_number)
    )

top_apps = pd.concat(_top_apps).reset_index()

# %%
target_cluster = "Tracking babies' rhythms"
top_apps[top_apps["cluster"] == target_cluster][
    ["cluster", "appId", "title", "developer", "installs", "score"]
]

# %%
top_apps[
    ["cluster", "appId", "title", "developer", "installs", "description", "score"]
].to_csv(OUTPUT_DIR / "top_apps.csv")
