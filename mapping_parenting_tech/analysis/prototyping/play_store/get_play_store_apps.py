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
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Extract app ids from web pages saved from Google Play store

# %% [markdown]
# Do imports and set file locations

# %%
import re
import json
import pickle
import csv
from google_play_scraper import Sort, app, reviews
from tqdm import tqdm
from pathlib import Path
from mapping_parenting_tech import PROJECT_DIR, logging


# %% [markdown]
# ## Read HTML and extract list of app ids

# %%
def get_playstore_app_ids(filename: str) -> list:
    """
    Looks inside 'inputs/data' int the project directory for a given HTML file (filename) and scans that file
    for links containing ids to Play Store apps.

    Argument:
        filename: str - the file name of the HTML file you wish to get app ids from. Must be in <[project directory]/inputs/data/>

    Returns:
        List of app ids, each as a str
    """
    # open the input file
    with open(PROJECT_DIR / "inputs/data" / filename, "rt") as infile:
        html = infile.read()

    # setup the regular expression
    # works by looking for app ids in the links to apps on a category page (e.g., https://play.google.com/store/apps/collection/cluster?clp=0g4hCh8KGXRvcHNlbGxpbmdfZnJlZV9QQVJFTlRJTkcQBxgD:S:ANO1ljI8w1M&gsr=CiTSDiEKHwoZdG9wc2VsbGluZ19mcmVlX1BBUkVOVElORxAHGAM%3D:S:ANO1ljK7gT4)
    # this is looking for a link ('href') to '/store/apps/details?id=' and it grabs the text after 'id'
    re_pattern = r"(?<=href=\"\/store\/apps\/details\?id=)(.*?)(?=\")"

    # retrieve all RE matches - this will return duplicates
    link_targets = re.findall(re_pattern, html)

    # convert list of links into a dict to remove duplicates, and back into a list
    app_ids = list(dict.fromkeys(link_targets))

    return app_ids


# %% [markdown]
# ## Save and load app ids to/from a pickle file

# %%
def save_app_ids(app_list: list, filename: str) -> bool:
    """
    Saves a pickled list to a given text file in <PROJECT_DIR/outputs/data/>
    """

    output_target = PROJECT_DIR / "outputs/data/" / filename

    with open(output_target, "wb") as output_file:
        pickle.dump(app_list, output_file)

    # if we got here, it's all ok so return True
    return True


# %%
# save_app_ids(get_playstore_app_ids("play_store/parenting_top_grossing.html"), "parenting_top_grossing_ids.pickle")

# %%
def load_app_ids(filename: str) -> list:
    """
    Loads a pickled list of app ids from a given file and returns the list object

    Arguments:
        filename: str - file name of pickle file to load within PARENT_DIR
    """

    app_id_list = []
    with open(PROJECT_DIR / filename, "rb") as p_list:
        app_id_list = pickle.load(p_list)

    return app_id_list


# %% [markdown]
# Retrieve app details

# %%
def get_playstore_app_details(app_id_list: list):
    """
    Returns dict of app details
    """

    all_app_details = dict()
    remove_apps = list()

    for app_id in tqdm(
        app_id_list,
        desc="Retrieving app details",
    ):
        try:
            app_details = app(
                app_id, lang="en", country="gb"  # defaults to 'en'  # defaults to 'us'
            )
            all_app_details.update({app_id: app_details})

        except Exception as e:  # needs modifying to capture specific errors
            logging.info(f"Error on app id {app_id}: {e} {repr(e)}")
            remove_apps.append(app_id)

    for app_id in remove_apps:
        all_app_details.remove(app_id)

    return all_app_details


# %%
# app_list = load_app_ids("outputs/data/top_free_parenting.pickle")

# %%
# app_details = get_playstore_app_details(app_list)

# %% [markdown]
# ## Save and load app details to/from JSON file

# %%
def save_app_details(details_dict: dict, filename: str) -> bool:
    """
    Saves JSON representation of app details. Will append to file <filename> if it already exists. Returns True if exits successfully
    """

    output_target = PROJECT_DIR / "outputs/data/" / filename
    with open(output_target, "at") as output_file:  # append in text mode
        json.dump(details_dict, output_file, indent=2, default=str)

    return True


# %%
def load_app_details(filename: str) -> json:
    """
    Loads JSON file containing app details from <filename>. Returns JSON object.
    """

    result = {}
    with open(PROJECT_DIR / filename, "rt") as input_file:
        result = json.load(input_file)

    return result


# %% [markdown]
# ## Download reviews for the apps in a given list of app ids
# Ultimately, functions will enable iterative downloading into a single file, which is appended to as more reviews are added
#
# `fetch_play_app_reviews` gets reviews for a single app
#

# %%
def get_playstore_app_reviews(
    target_app_id, how_many: int = 200, continuation_token: str = None
) -> [[{}], object]:
    """
    Returns up to 200 reviews for a given app in the Google Play Store. If more reviews are available, 'continuation_token' is also returned, which can then
    be passed as a parameter to indicate where the function should resume.

    Arguments:
        target_app_id: id of the app for which you want to download reviews
        how_many: number of reviews you wish to download in one grab; defaults to 200, which is the maximum option
        continuation_token: indicates that the function should continue fetching reviews from this point

    Returns
        1. a list of reviews, each as a dict (see below)
        2. continuation_token: object to pass back to this function to pick up where it left off

    The reviews themselves are dicts in the following format:
        "userName": str,
        "userImage": str,
        "content": str,
        "score": int,
        "thumbsUpCount": int,
        "reviewCreatedVersion": str,
        "at": datetime.datetime,
        "replyContent": str,
        "repliedAt": datetime.datetime,
        "reviewId": str
    """

    if target_app_id == "":
        return {"No app id given"}

    if how_many > 200:
        how_many = 200

    if continuation_token is None:
        fetch_reviews, continuation_token = reviews(
            app_id=target_app_id,
            lang="en",
            country="gb",
            sort=Sort.NEWEST,
            count=how_many,
        )
    else:
        fetch_reviews, continuation_token = reviews(
            app_id=target_app_id, continuation_token=continuation_token
        )

    return (fetch_reviews, continuation_token)


# %%
def get_playstore_app_list_reviews(
    app_id_list: list,
    filename: str,
    force_download: bool = False,
    run_quietly: bool = False,
):
    """ """

    target_file = PROJECT_DIR / "outputs/data" / filename
    file_root = Path(target_file).stem
    log_file = PROJECT_DIR / "outputs/data" / (file_root + ".log")

    # Field names being used in the CSV file
    field_names = [
        "content",
        "score",
        "thumbsUpCount",
        "reviewCreatedVersion",
        "at",
        "replyContent",
        "repliedAt",
        "reviewId",
        "appId",
    ]

    # Does the target file (filename) already exist? If so, get a list of the ids from the reviews already in there
    if target_file.exists():
        with open(target_file, "rt", newline="") as csv_file:
            fetched_reviews = csv.DictReader(csv_file, fieldnames=field_names)
            existing_review_ids = [r["reviewId"] for r in fetched_reviews]

        # Is there a log file - this contains details of reviews downloaded for each app given, provided it's being saved to
        # the same filename as used originally
        if log_file.exists():
            log_file_handle = open(log_file, "rb")
            log_info = pickle.load(log_file_handle)
        else:
            log_file_handle = open(log_file, "wb")
            log_info = {}
        log_file_handle.close()

    # If not, create empty new file for reviews and log
    else:
        with open(target_file, "wt", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
            csv_writer.writeheader()

        fetched_reviews = []
        existing_review_ids = []
        log_file_handle = open(log_file, "wb")
        log_info = {}
        log_file_handle.close()

    # tqdm(app_id_list, desc="Retrieving app reviews"):
    for app_id in app_id_list:
        # reset variables ahead of downloading an app's reviews
        first_pass = True
        keep_going = True
        more_to_get = True
        continuation_token = None
        review_fetch = []
        app_review_count = 0

        # Is the app in the logfile? If so, is it completed?
        if app_id in log_info:
            if log_info[app_id]["completed"]:
                keep_going = False
            else:
                continuation_token = log_info[app_id]["continuation_token"]
                first_pass = False
                more_to_get = True
                log_dump = {app_id: log_info[app_id]}

        # if we're forcing a download, then at least download the first set of reviews
        if force_download:
            first_pass = True
            keep_going = True

        while (more_to_get or first_pass) and keep_going:
            review_fetch, continuation_token = get_playstore_app_reviews(
                target_app_id=app_id, continuation_token=continuation_token
            )

            # if we just grabbed a full 200 reviews, there are (probably) more to get
            more_to_get = True if len(review_fetch) == 200 else False

            # process what we've just downloaded:
            # 1. add the add id to each review
            # 2. remove the userName and userImage
            # 3. add to the target file if we don't already have it
            with open(target_file, "at", newline="") as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
                for review in review_fetch:
                    review.update({"appId": app_id})
                    del review["userName"]
                    del review["userImage"]
                    if review["reviewId"] not in existing_review_ids:
                        csv_writer.writerow(review)
                        existing_review_ids.append(review["reviewId"])

            # save a snapshot of where we are - the app_id and continuation_token - so we can resume later
            with open(log_file, "rb+") as log_file_handle:
                if first_pass:
                    log_dump = {
                        app_id: {
                            "completed": False,
                            "continuation_token": continuation_token,
                            "latest_review_id": review_fetch[0]["reviewId"],
                            "latest_review_time": review_fetch[0]["at"],
                            "downloaded": len(review_fetch),
                        }
                    }
                else:
                    log_dump[app_id].update(
                        {
                            "continuation_token": continuation_token,
                            "downloaded": (
                                log_dump[app_id]["downloaded"] + len(review_fetch)
                            ),
                        }
                    )
                log_info.update(log_dump)
                pickle.dump(log_info, log_file_handle)

            app_review_count += len(review_fetch)

            if not run_quietly:
                print(
                    f"Retrieved {app_review_count} new reviews for {app_id}; fetched {log_dump[app_id]['downloaded']} in total)          ",
                    end="\r",
                )

            first_pass = False
            if force_download:
                keep_going = True

        # if we've reached here and keep_going is true it's because we've downloaded all the results, so update logfile accordingly
        if keep_going:
            with open(log_file, "rb+") as log_file_handle:
                log_dump[app_id].update({"completed": True, "continuation_token": None})
                log_info.update(log_dump)
                pickle.dump(log_info, log_file_handle)
        if not run_quietly:
            print(f"\n{app_id}: done")

    return fetched_reviews


# %%
my_app_list = [
    "com.lingumi.lingumiplay",
    "com.microsoft.familysafety",
    "no.mobitroll.kahoot.android",
    "com.classdojo.android",
]

my_app_reviews = get_playstore_app_list_reviews(my_app_list, "myapps.csv")

# %%
field_names = [
    "content",
    "score",
    "thumbsUpCount",
    "reviewCreatedVersion",
    "at",
    "replyContent",
    "repliedAt",
    "reviewId",
    "appId",
]

moo = {}
with open(PROJECT_DIR / "outputs/data/myapps.csv", "rt", newline="") as csvfile:
    foo = csv.DictReader(csvfile, fieldnames=field_names)
    blob = list(foo)

print(json.dumps(blob[2222], default=str, indent=2))

# %%
all_app_reviews = dict()
running_total = 0

for app_id in tqdm(output_ids["Parenting apps"], desc="Retrieving app reviews"):
    try:
        app_reviews = reviews_all(
            app_id,
            sleep_milliseconds=0,  # defaults to 0
            lang="en",
            country="gb",
            sort=Sort.NEWEST  # defaults to Sort.MOST_RELEVANT
            # filter_score_with=5 # defaults to None(means all score)
        )
        all_app_reviews.update({app_id: app_reviews})
        running_total += len(app_reviews)

    except Exception as e:
        logging.info(f"Error on app id {app_id}: {e} {repr(e)}")

logging.info(f"Retrieved {running_total} reviews")


# %%
output_target = OUTPUT_PATH / OUTPUT_PLAY_REVIEWS_FILE
with open(output_target, "w") as output_file:
    json.dump(all_app_reviews, output_file, indent=2, default=str)
output_file.close()

print(f"{running_total} reviews saved to {output_target}")

# %%

# %%

# %%
