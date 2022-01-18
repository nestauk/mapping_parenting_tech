# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: 'Python 3.8.0 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import json
from pathlib import Path
from mapping_parenting_tech import PROJECT_DIR, logging
from tqdm import tqdm

DATA_DIR = PROJECT_DIR / "outputs/data"
REVIEWS_DIR = DATA_DIR / "app_reviews"

# %%
# get list of all CSV and JSON files in target folder
raw_list = [
    file
    for file in DATA_DIR.iterdir()
    if (file.is_file() and (file.suffix == ".csv" or file.suffix == ".json"))
]

# %% [markdown]
# ## Handle app ids
# Take app ids from any file ending in '_ids.csv' and save them to a single, consolidated file, de-duplicating in the process

# %%
csv_pd = pd.DataFrame
csv_list = list()

for file in tqdm(raw_list):
    if file.suffix == ".csv" and file.stem[-4:] == "_ids":
        csv_pd = pd.read_csv(file, index_col=None, header=0)
        csv_list.extend(csv_pd[csv_pd.columns[0]].to_list())

csv_list = list(set(csv_list))
csv_pd = pd.DataFrame(csv_list, columns=["appId"])
csv_pd.to_csv(DATA_DIR / "app_id_list.csv", index=False)


# %% [markdown]
# ## Handle app details
# Take app descriptions from any file ending in '_details.json' and save into a single, consolidated file, de-duplicating in the process. Note: saves two formats, CSV and JSON

# %%
def load_details_set(filename: str) -> pd.DataFrame:
    # Load in the descriptions

    with open(filename, "rt") as all_data_handle:
        all_data = json.load(all_data_handle)

    all_data = {k: v for d in all_data for k, v in d.items()}
    return pd.DataFrame(all_data).T


# %%
def load_details(file_list: list) -> pd.DataFrame:

    list_pd = [load_details_set(f) for f in file_list]
    return_df = pd.concat(list_pd, axis=0, ignore_index=True)
    return_df.drop_duplicates(subset=["appId"], inplace=True)
    return return_df


# %%
target_json = [
    file for file in raw_list if file.suffix == ".json" and file.stem[-8:] == "_details"
]

# %%
all_details = load_details(target_json)
all_details = all_details.set_index(["appId"])

# %%
all_details.to_csv(DATA_DIR / "all_app_details.csv")

# %%
all_details.to_json(DATA_DIR / "all_app_details.json", indent=2, orient="index")

# %% [markdown]
# ## Handle app reviews
# Get app reviews from all files ending in '_reviews.csv' and save into a separate CSV file for each app

# %%
df_list = list()
for f in raw_list:
    if (f.stem[-8:] == "_reviews") and (f.suffix == ".csv"):
        logging.info(f"Processing {f}")
        loader_df = pd.read_csv(f, low_memory=False)
        df_list.append(loader_df)

logging.info("Loaded files; concatenating dataframes")
app_reviews_df = pd.concat(df_list, axis=0, ignore_index=True)
logging.info("Loaded files; concatenating dataframes")
app_reviews_df.drop_duplicates(subset=["reviewId"], inplace=True)

# %%
reviewed_apps = set(app_reviews_df["appId"].to_list())
for app_id in tqdm(reviewed_apps):
    review_set = app_reviews_df[app_reviews_df["appId"] == app_id]
    review_set = review_set.set_index(["appId"])
    review_set.to_csv(REVIEWS_DIR / f"{app_id}.csv")


# %% [markdown]
# # Helper functions to retrieve data

# %%
def load_all_app_reviews() -> pd.DataFrame():
    review_files = REVIEWS_DIR.iterdir()
    reviews_df_list = list()

    for file in tqdm(review_files):
        reviews_df = pd.read_csv(file, header=0, index_col=None)
        reviews_df_list.append(reviews_df)

    all_reviews = pd.concat(reviews_df_list, axis=0, ignore_index=True)
    return all_reviews


# %%
def load_all_app_details() -> dict():

    details_df = pd.read_json(DATA_DIR / "all_app_details.json", orient="index")
    return details_df.to_dict(orient="index")


# %%
def load_all_app_ids() -> list():

    app_ids_df = pd.read_csv(DATA_DIR / "all_app_ids.csv", index_col=None, header=0)
    return app_ids_df[app_ids_df.columns[0]].to_list()


# %%
