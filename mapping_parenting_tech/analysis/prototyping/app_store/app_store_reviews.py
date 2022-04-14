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
# # Hidden API info
# `http://itunes.apple.com/WebObjects/MZStore.woa/wa/` contains web pages that serve up information about apps. Two are (incompletely) described here, `customerReviews` and `userReviewsRow`.
#
# See also https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/Searching.html#//apple_ref/doc/uid/TP40017632-CH5-SW1
#
# ## Summary information about an app's reviews via `customerReviews`
# `customerReviews` returns information (JSON) about reviews but not any reviews. Returned items include:
# - `totalNumberOfReviews` number of reviews
# - `ratingCount` number of ratings
# - `ratingAverage` average rating
# - `ratingCountList` list of ratings counts, seemingly ordered by counts for 1..5 stars; e.g. `[148,203,705,1668,5178]`
# ## Specific reviews about an app via `userReviewsRow`
# `userReviewsRow` returns a list of reviews as JSON. Relevant items returned include:
# - `userReviewId`
# - `body`
# - `date`
# - `name`
# - `rating`
# - `title`
# ## Parameters
# - `id`: numeric id of the app.Applies to `customerReviews` and `userReviewsRow`
# - `displayable-kind`: seems to filter the kind of item that will be retrieved. Value of 11 seems to be iOS software. However, unclear whether/how this affects `customerReviews` or `userReviewsRow`
# - `sort` or `sortId`: (unclear which - could be both, could be one or the other); takes a numeric value and sorts reviews accordingly (see below). Applies to `userReviewsRow`
# -- value: sort order
# -- 1: Most Helpful
# -- 2: Most Favourable
# -- 3: Most Critical
# -- 4: Most Recent
# - `startIndex`: the index of the first review (based on the sort order) to retrieve. Applies to `userReviewsRow`
# - `endIndex`: the index of the last review (based on the sort order) to retrieve. Appliese to `userReviewsRow`
# - NB: It's unclear whether `startIndex` and `endIndex` have any sort of limit; e.g., whether there's a maximum difference between them. They have retrieved over 1,000 reviews in a single request.

# %% [markdown]
# Imports and initialise variables

# %%
import requests
import urllib.parse
import json
import pandas
import sys

app_id = 445196535

# setup the headers for the HTTP request
headers = {"X-Apple-Store-Front": "143444,29", "Accept-Language": "en"}

# %% [markdown]
# ### Get information about an app
# `http://itunes.apple.com/app/id{app_id}` returns a large JSON object, too complicated to fully describe here. Key items include:
# - `storePlatformData`
#     - `product-dv`
#         - `results`
#             - {appId}
#                 - `ageBand`
#                     - `minAge` (int)
#                     - `maxAge` (int)
#                 - `deviceFamilies`
#                     - [array of devices]
#                     - e.g., 'iphone', 'ipad', 'ipod'
#                 - `genreNames`
#                     - [array of genres]
#                     - e.g., 'Education', 'Family', 'Games'
#                 - `description`
#                     - `standard` (str) (long description of the app)
#                 - `kind` (str) (e.g., 'iosSoftware')
#                 - `name` (str) (e.g., 'Baby games for one year olds.')
#                 - `subtitle` (str) (e.g., 'Learning for kids age 1 2 3 4')
# - `pageData`
#     - `customersAlsoBoughtApps`
#         - [array of ids of other apps] (int)
#     - `moreByThisDeveloper`
#         - [array of ids of other apps] (int)
#     - `customerReviewsUrl` (str) (URL of customer reviews)
# - `properties` (just two items, neither seem relevant - 'revNum' and timestamp)

# %%
url = "http://itunes.apple.com/app/id" + str(app_id)
res = requests.get(url, headers=headers)

app_full_details = json.loads(res.text)
app_key_details = {
    **app_full_details["storePlatformData"]["product-dv"]["results"][f"{app_id}"],
    **app_full_details["pageData"],
}
app_description = app_key_details["description"]["standard"]
app_reviews_url = app_key_details["customerReviewsUrl"]
# note that this URL points to a summary page about the app's reviews; see next cell

# %% [markdown]
# Get information about an app's reviews (namely, how many), but not specific reviews

# %%
base = "http://itunes.apple.com/WebObjects/MZStore.woa/wa/customerReviews?"
# NB seems equivalent to https://itunes.apple.com/gb/customer-reviews/id{app_id}?dataOnly=true&displayable-kind=11

params = {"id": app_id, "displayable-kind": 11, "sort": 1}
url = base + urllib.parse.urlencode(params)
res = requests.get(url, headers=headers)
if res.status_code == 200:
    print("Success gathering review info")
    reviewInfo = json.loads(res.text)
else:
    sys.exit("Fail")

totalReviews = reviewInfo["totalNumberOfReviews"]

# %% [markdown]
# Get app's reviews

# %%
base = "http://itunes.apple.com/WebObjects/MZStore.woa/wa/userReviewsRow?"  # this gets reviews
params = {
    "id": app_id,
    "displayable-kind": 11,
    "startIndex": 0,
    "endIndex": totalReviews,
    "sort": 1,
}
url = base + urllib.parse.urlencode(params)

res = requests.get(url, headers=headers)

if res.status_code == 200:
    print("Success gathering reviews")
    reviews = json.loads(res.text)
else:
    sys.exit("Fail")

pd = pandas.DataFrame(reviews["userReviewList"])

print(len(reviews["userReviewList"]))
# for review in reviews['userReviewList']:
#    print(review['body'])

pd.sample(10)

# %% [markdown]
# Code below is obsolete / testing code

# %% [markdown]
# '''
# Deprecated - this was me trying to use an API to scrape the App store
#
# from mapping_parenting_tech.analysis.prototyping.app_store.api.itunes_app_scraper.scraper import AppStoreScraper
#
# scraper = AppStoreScraper()
# results = scraper.get_app_ids_for_query("fortnite", country="gb", lang="en")
# similar = scraper.get_similar_app_ids_for_app(results[0])
#
# app_details = scraper.get_multiple_app_details(similar)
# #print(list(app_details))
# '''

# %% [markdown]
# '''
# as above
#
# from api.itunes_app_scraper.scraper import AppStoreScraper
#
# apps = AppStoreScraper()
# foo = apps.get_app_ids_for_query("minecraft", num=10, country="gb", lang="en")
#
# print(foo)
# '''

# %%
country = "gb"
print("%s,24" % country)

# %%
"""
Another approach?
base = "https://itunes.apple.com/us/rss/customerreviews/"
params = {
    "id": 479516143,
    "page": 0,
    "sortby": 2
}
url = base + urllib.parse.urlencode(params)
url = url + "/json"
"""
