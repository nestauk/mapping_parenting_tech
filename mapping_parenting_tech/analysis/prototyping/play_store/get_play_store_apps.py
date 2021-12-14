# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Code to extract app ids from [list of] web pages saved from Google Play store

# %% [markdown]
# Do imports and set file locations

# %%
from pathlib import Path
from google_play_scraper import Sort, app, reviews_all
import re
import json
import sys

INPUT_PATH = Path("~/github/mapping_parenting_tech/inputs/data/")
OUTPUT_PATH = Path("~/github/mapping_parenting_tech/outputs/data/")
STORE_PATH = "play_store/"
PLAY_DATA_FILE = "playstore_innerHTML.html"
OUTPUT_PLAY_IDS_FILE = "play_store_ids.json"
OUTPUT_PLAY_REVIEWS_FILE = "play_store_reviews.json"

input_data = INPUT_PATH.expanduser() / STORE_PATH / PLAY_DATA_FILE

if input_data.exists():
    print("Ok to proceed")
else:
    print("Didn't find file")


# %% [markdown]
# ## For Play Store
# Read HTML and extract list of app ids

# %%
html_doc = open(input_data, "r")
html = html_doc.read()
html_doc.close

link_targets = re.findall(r"(?<=href=\"\/store\/apps\/details\?id=)(.*?)(?=\")", html)
app_ids = list(dict.fromkeys(link_targets))
output_ids = {"Parenting apps": app_ids}
print("Apps ids parsed ok")

# %%
output_target = OUTPUT_PATH.expanduser() / OUTPUT_PLAY_IDS_FILE
with open(output_target, "w") as output_file:
    json.dump(output_ids, output_file)
output_file.close
print("App ids saved to", output_target)

# %% [markdown]
# Download reviews for the apps in the list

# %%
all_app_reviews = dict()
running_total = 0
counter = 1

for app_id in output_ids["Parenting apps"]:
    try:
        app_details = app(
            app_id, lang="en", country="gb"  # defaults to 'en'  # defaults to 'us'
        )
        sys.stdout.write("\r")
        sys.stdout.write(
            f'Downloading {app_details.get("reviews")} reviews for app {counter} of {len(output_ids["Parenting apps"])}    '
        )
        sys.stdout.flush()
        if app_details["reviews"] < 1000:
            app_reviews = reviews_all(
                app_id,
                sleep_milliseconds=0,  # defaults to 0
                lang="en",
                country="gb",
                sort=Sort.MOST_RELEVANT  # defaults to Sort.MOST_RELEVANT
                # filter_score_with=5 # defaults to None(means all score)
            )
            all_app_reviews.update({app_id: app_reviews})
            running_total += len(app_reviews)

    except:
        print(f"App id {app_id} not found")

    counter += 1

print(f"Retrieved {running_total} reviews")


# %%
output_target = OUTPUT_PATH.expanduser() / OUTPUT_PLAY_REVIEWS_FILE
with open(output_target, "w") as output_file:
    json.dump(all_app_reviews, output_file, indent=2, default=str)
output_file.close

print(f"{running_total} reviews saved to {output_target}")

# %%
print(running_total)
print(len(all_app_reviews))
print(app_id)
# sys.stdout.write(f"Downloading {app_details.get("reviews")} reviews for app {counter} of {len(output_ids["Parenting apps"])}")
msg = f'Downloading {app_details.get("reviews")} reviews for app {counter} of {len(output_ids["Parenting apps"])}'
sys.stdout.write(msg)
print(msg)
