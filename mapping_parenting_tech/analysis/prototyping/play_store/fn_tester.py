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

# %% [markdown]
# # Testing notebook
# Testing refactoring of functions to manage ids, details and reviews of Play Store apps

# %%
from mapping_parenting_tech.utils import play_store_utils as psu
from mapping_parenting_tech import PROJECT_DIR, logging

# %% [markdown]
# ## Working with app ids
# ### Getting app ids from source/s
# App ids are retrieved in two ways:
# 1. parsing an HTML file saved from a Play Store web page - this can be done with a single web page or via a folder that contains multiple web pages
# 2. snowballing from a given app id and using that to identify related apps

# %%
# test downloading apps from a single HTML web page
appIds = psu.get_playstore_app_ids_from_html("play_store/parenting_top_free.html")
appIds

# %%
# test parsing a folder of HTML files
appIds = psu.get_playstore_app_ids_from_folder("play_store/education_apps")
len(appIds)

# %%
# test retrieving related apps to a depth of 3
appIds = psu.app_snowball("uk.org.bestbeginnings.babybuddymobile", 3)
len(appIds)

# %% [markdown]
# ### Saving app ids and loading saved app ids
# App ids are saved to a single CSV file `outputs/data/all_app_ids.csv` with one app id per line

# %%
# test saving new apps - dry run allows us to do this without updating the file itself
# function returns complete list of new apps (i.e., not just those that have been added)
new_appIds = psu.update_all_app_id_list(appIds, dry_run=False)
new_appIds

# %%
# load all app ids
all_appIds = psu.load_all_app_ids()
len(all_appIds)

# %%
# check if an app is in the list of all app ids - here we're using the results of the snowball above
compare_list = psu.is_app_in_list(appIds)

# enumerate through compare_list to identify apps that aren't present
print(f"{compare_list.count(False)} apps are not already saved")
for i, present in enumerate(compare_list):
    if not present:
        print(appIds[i])

# %% [markdown]
# ## Working with app details
# App details can be downloaded into memory in a standalone fashion or saved to and loaded from disk via the JSON file: `outputs/data/all_app_details.json`. This file is updated/synchronised according to the saved app ids.

# %%
# get the app details for the list of apps in `app_list`

app_list = [
    "com.amboss.medical.knowledge",
    "com.excelatlife.depression",
    "au.com.sightwords.parrotfish.lite",
    "com.kitefaster.whitenoise",
    "com.apololo.brightest.flashlight",
]
app_list_details = psu.get_playstore_app_details(app_list)
app_list_details["com.amboss.medical.knowledge"]["title"]

# %%
# load all app details into a dict
all_app_details = psu.load_all_app_details()
len(all_app_details)

# %%
# update the details for apps whose ids are already saved - this returns the details for all of the apps in question
all_app_details = psu.update_all_app_details()
len(all_app_details)

# %% [markdown]
# ## Working with app reviews
# App reviews are saved into `outputs/data/app_reviews` but are loaded into a consolidated Pandas DataFrame. Due to the potential number of reviews for each app, downloading can take some time

# %%
# get some app reviews for a given app - if there are more than 200 reviews, this will return the most recent
# 200 reviews. Note that nothing is saved to disc in this procedure
my_app_reviews, _ = psu.get_playstore_app_reviews("com.amboss.medical.knowledge")
len(my_app_reviews)

# %%
# download reviews for apps in our example app list, forcing all reviews to be downloaded
app_list = [
    "com.amboss.medical.knowledge",
    "com.excelatlife.depression",
    "au.com.sightwords.parrotfish.lite",
    "com.kitefaster.whitenoise",
    "com.apololo.brightest.flashlight",
]
psu.save_playstore_app_list_reviews(app_list, force_download=True, run_quietly=False)

# %%
# get a list of app ids for those apps whose reviews haven't been downloaded
apps_to_get = psu.list_missing_app_reviews()
apps_to_get

# %%
# load every review for every app that we have - this can take a couple of minutes
all_app_reviews = psu.load_all_app_reviews()
all_app_reviews.shape

# %%
# load the review for the apps in our list - we'll just load the content and reviewId
app_list = [
    "com.amboss.medical.knowledge",
    "com.excelatlife.depression",
    "au.com.sightwords.parrotfish.lite",
    "com.kitefaster.whitenoise",
    "com.apololo.brightest.flashlight",
]

my_app_reviews = psu.load_some_app_reviews(app_list)
my_app_reviews.shape
