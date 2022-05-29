# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Download app icons

# %%
import urllib.request
import utils
import pandas as pd
import re
from tqdm.notebook import tqdm
import time

# %%
outputs = utils.OUTPUT_DIR / "icons"
outputs.mkdir(parents=True, exist_ok=True)

# %%
app_details = utils.get_app_details()

# %%
top_apps = pd.concat(
    [
        utils.get_top_cluster_apps(app_details, cluster, sort_by="minInstalls", top_n=5)
        for cluster in utils.clusters_children
    ],
    ignore_index=True,
)


# %%
def fetch_icons(df, output_dir=outputs, restart_i=-1):
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if i > restart_i:
            filename = re.sub("\.", "_", row.appId)
            urllib.request.urlretrieve(row.icon, output_dir / f"{filename}.png")
            time.sleep(0.25)


# %%
fetch_icons(app_details, restart_i=1064)

# %%
