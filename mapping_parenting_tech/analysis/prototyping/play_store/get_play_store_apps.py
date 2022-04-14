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
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Extract app ids from web pages saved from Google Play store
# The notebook below establishes and tests functions to achieve the following:
# 1. retrieve the ids of apps from the Play Store
# 2. save and load app ids so they can be saved, retrieved and used later
# 3. download details for an app/s using its/their app id/s
# 4. save and load app details so they can be saved, retrieved and used later
#
# In step 4, app details are saved on-the-fly, as they are retrieved. As this step can take some time for 100s of apps or for apps with 1,000s of reviews (or both), the process can timeout or fail unexpectedly. Saving app details as they are retrieved allows the step to be resumed part-way through as a log file is saved with details of progress for each app.
#
# Note that app ids, details and reviews are managed independently and may not be synchronised. A list of 100 app ids does not mean that those 100 apps have had their details and/or reviews saved too. It's also possible that some apps' details and/or reviews might have been saved, yet their ids are not saved in the app id list. Two functions update *from* the app ids (`update_all_app_details` and `update_all_app_reviews`), which use the app id list as their basis for what to fetch. However, no functions exist in any other direction - i.e., if there are more app details than app ids, there is no function to add the missing app ids to the id list.
#
# These functions are dependent on the [google_play_scraper](https://pypi.org/project/google-play-scraper/) library.

# %% [markdown]
# ## Do imports and set file locations

# %%
from mapping_parenting_tech import logging
from mapping_parenting_tech.utils import play_store_utils as psu

# %% [markdown]
# ## Putting it together
# ### Retrieving and using app ids via a set of downloaded web pages

# %%
app_ids = psu.get_playstore_app_ids_from_folder("play_store/kids_under_five")
psu.update_all_app_id_list(app_ids)

# %%
# update app details in light of the new app ids we've retrieved
psu.update_all_app_details()

# %%
# get the reviews just for the apps we've identified
psu.save_playstore_app_list_reviews(app_ids)

# %% [markdown]
# Take a list of apps (here, `INTERESTING_APPS`), identify those that are new and then snowball from them

# %%
existing_app_list = psu.load_all_app_ids()
apps_to_explore = [x for x in psu.INTERESTING_APPS if x not in existing_app_list]
app_details_set = list()
for x in apps_to_explore:
    logging.info(f"Getting apps related to {x}")
    related_apps = psu.app_snowball(x)
    app_details_set.extend(related_apps)

# %%
# save the ids of apps identified via the snowballing in the previous step
psu.update_all_app_id_list(app_details_set)

# %% [markdown]
# Download the reviews for apps whose reviews have not yet been downloaded

# %%
apps_to_do = psu.list_missing_app_reviews()
logging.info(f"Reviews for {len(apps_to_do)} apps' reviews to be downloaded.")

# %%
psu.save_playstore_app_list_reviews(apps_to_do)
