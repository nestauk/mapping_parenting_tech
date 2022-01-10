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
# The notebook below establishes and tests functions to achieve the following:
# 1. retrieve the ids of apps from the Play Store
# 2. save and load app ids so they can be saved, retrieved and used later
# 3. download details for an app/s using its/their app id/s
# 4. save and load app details so they can be saved, retrieved and used later
# In step 4, app details are saved on-the-fly, as they are retrieved. As this step can take some time for 100s of apps or for apps with 1,000s of reviews (or both), the process can timeout or fail unexpectedly. Saving app details as they are retrieved allows the step to be resumed part-way through as a log file is saved with details of progress for each app.
# These functions are dependent on the [google_play_scraper](https://pypi.org/project/google-play-scraper/) library.

# %% [markdown]
# ## Do imports and set file locations

# %%
import re
import json
import pickle
import csv
from google_play_scraper import Sort, app, reviews
from tqdm import tqdm
from pathlib import Path
from mapping_parenting_tech import PROJECT_DIR, logging
from typing import Iterator

APP_IDS_PATH = PROJECT_DIR / "outputs/data"

# Field names being used in the CSV file
FIELD_NAMES = (
    "appId",
    "content",
    "score",
    "thumbsUpCount",
    "reviewCreatedVersion",
    "at",
    "replyContent",
    "repliedAt",
    "reviewId",
)


# %% [markdown]
# ## Functions to retrieve app ids
# `get_playstore_app_ids` takes a local (downloaded) HTML page from the Play Store, specified by `filename`, and extracts app ids from it. As the Play Store loads items dynamically on scrolling, it's necessary to scroll to the bottom of the page, use a DOM Inspector (provided within Firefox, Safari, Chrome) and copy and paste the HTML surrounding the list of app links. Typically, this is a `div` element of class `ZmHEEd`.
#
# `parse_folder_for_app_ids` takes a folder and looks for HTML files (as described above) and calls `get_playstore_app_ids` on each one. It returns a single consolidated list of app ids. In this way, the various Play Store pages for an area (e.g., Parenting: top paid, top free, trending, etc.) can be consolidated into a single list of apps.
#
# `app_snowball` takes an app id and uses it to find related apps by using the similar apps listed on each app's page (details). It then looks at the similar apps listed on those apps' pages and so on. The `depth` argument specifies how many iterations down the process should go; the default is 5, which takes ~6 minutes (each extra step will take expotentially longer).
#
# `save_app_ids` saves a given list of app ids (i.e., from one of the steps above) and saves them to a CSV file. It takes the apps' category (e.g., *Education* or *under fives*) as a heading for its single column of data.
#
# `load_app_ids` retrieves the ids saved by `save_app_ids`. **NOTE** it returns two variables, the first is the category heading, the second is the list of app ids.

# %%
def get_playstore_app_ids(filename: str) -> list:
    """
    Looks inside 'inputs/data' int the project directory for a given HTML file (filename) and scans that file
    for links containing ids to Play Store apps.

    Args:
        filepath: str - the file name of the HTML file you wish to get app ids from. Must be in
        <[project directory]/inputs/data/> and should include further path details if file is in a sub-folder
        e.g., `play_store/app_page.html`

    Returns:
        List of app ids, each as a str
    """

    # open the input file
    with open(PROJECT_DIR / filename, "rt") as infile:
        html = infile.read()

    # setup the regular expression that looks for app ids in the links to apps on a category page
    # (e.g., https://play.google.com/store/apps/collection/cluster?clp=0g4hCh8KGXRvcHNlbGxpbmdfZnJlZV9QQVJFTlRJTkcQBxgD:S:ANO1ljI8w1M&gsr=CiTSDiEKHwoZdG9wc2VsbGluZ19mcmVlX1BBUkVOVElORxAHGAM%3D:S:ANO1ljK7gT4)
    # this is looking for a link ('href') to '/store/apps/details?id=' and it grabs the text after 'id'
    re_pattern = r"(?<=href=\"\/store\/apps\/details\?id=)(.*?)(?=\")"

    # retrieve all RE matches - this will return duplicates
    link_targets = re.findall(re_pattern, html)

    # convert list of links into a dict to remove duplicates, and back into a list
    app_ids = list(set(link_targets))

    return app_ids


# %%
def parse_folder_for_app_ids(folder: str) -> list:
    """
    Scans a folder for HTML files and extracts the app ids from those files, returning a unified, de-duplicated list of app ids.

    Args:
        folder: str - the folder to be scanned. This should be located in 'inputs/data' in a Nesta cookiecutter project

    Returns:
        A list object of Play Store app ids, each as a str

    """

    target_folder = PROJECT_DIR / "inputs/data" / folder
    app_id_list = list()

    # iterate .html files in the given folder, pass them to get_playstore_app_ids and assign results by extending app_id_list
    app_id_list.extend(
        [
            get_playstore_app_ids(file_name)
            for file_name in target_folder.iterdir()
            if (file_name.is_file() and file_name.suffix == ".html")
        ]
    )

    # flatten app_id_list (each file parsed will have returned a separate list)
    app_id_list = [item for sublist in app_id_list for item in sublist]

    # use `set` to remove duplicates in the return
    return list(set(app_id_list))


# %%
def app_snowball(seed_app_id: str, depth: int = 5, __current_depth: int = 1) -> list:
    """
    Retrieves ids of Play Store apps related to `seed_app_id` by calling itself recursively.

    Args:
        seed_app_id: str - the app id of the app of interest
        depth: int, default = 5 - the depth of recursion. This will increase the number of apps interrogated (and
        therefore the time taken for the initial call to complete) exponentially
        current_depth: used for recursion, should be left blank by user

    Returns:
        a list of app ids
    """

    app_details = app(seed_app_id, country="gb")
    similar_apps = app_details["similarApps"]

    snowball = set()
    snowball.update(similar_apps)

    if __current_depth < depth:
        for this_app in similar_apps:
            snowball.update(app_snowball(this_app, depth, (__current_depth + 1)))

    return list(snowball)


# %%
def save_app_ids(
    app_list: list, app_category: str, filename: str, output_path: Path = APP_IDS_PATH
) -> bool:
    """
    Saves a list of app ids to a given CSV file. If the file already exists, it is overwritten.

    Args:
        app_list: list - a list of app ids to be saved
        app_category: str - the title of the app category being saved, used as the header at the top of the list
        filename: str - the filename to be used, including extension
        output_path: Path - destination folder; default is `outputs/data`

    Returns:
        True if executed successfully

    """

    output_target = output_path / filename

    with open(output_target, "wt", newline="\n") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow([app_category])
        # NB can't use writerows() as it will treat each string as a list and split it into comma-separated letters
        # hence why `app_id` (and `app_category`, above) is encapsulated in square brackets
        for app_id in app_list:
            csv_writer.writerow([app_id])

    # if we got here, it's all ok so return True
    return True


# %%
def load_app_ids(filename: str, file_path: Path = APP_IDS_PATH) -> tuple:
    """
    Loads a pickled list of app ids from a given file and returns the list object

    Args:
        filename: str - file name of pickle file to load within PARENT_DIR
        file_path: Path - location of the file given in `filename`; default is `outputs/data`

    Returns:
        str: the name of the app category
        list: list of app ids

    """

    app_id_list = []
    with open(file_path / filename, "rt") as id_list:
        csv_reader = csv.reader(id_list)
        for app_id in csv_reader:
            app_id_list.extend(app_id)

    app_category = app_id_list.pop(0)
    return (app_category, app_id_list)


# %% [markdown]
# ## Retrieving app details
# These functions get details for a given list of apps, specified by their ids (as retrieved above) and save and load those details.
#
# `get_playstore_app_details` takes a list of app ids and returns the details for those apps as a list of `dict` objects. The schema for each `dict` can be seen on [https://pypi.org/project/google-play-scraper/](https://pypi.org/project/google-play-scraper/)
#
# `save_app_details` saves the list of `dicts` retrieved by `get_playstore_app_details`
#
# `load_app_details` loads the app details saved by `save_app_details`

# %%
def get_playstore_app_details(app_id_list: list):
    """
    Uses `google-play-scraper` (https://pypi.org/project/google-play-scraper/) to retrieve details about apps given in
    `app_id_list` and returns dict of app details

    Args:
        app_id_list: list - a list of app ids from which details will be retrieved

    Returns:
        List of dict objects with app details. Each dict includes:
        - title
        - appId
        - description
        - summary
        for specifics of all data returned, see https://pypi.org/project/google-play-scraper/

    """

    all_app_details = list()
    remove_apps = list()

    for app_id in tqdm(
        app_id_list,
        desc="Retrieving app details",
    ):
        try:
            app_details = app(app_id, country="gb")
            all_app_details.append({app_id: app_details})

        except Exception as e:  # needs modifying to capture specific errors
            logging.warning(f"Error on app id {app_id}: {e} {repr(e)}")
            remove_apps.append(app_id)

    for app_id in remove_apps:
        all_app_details.pop(app_id, None)

    return all_app_details


# %%
def save_app_details(
    details_dict: dict, filename: str, folder: str = APP_IDS_PATH
) -> bool:
    """
    Saves JSON representation of app details. Will append to file <filename> if it already exists.
    Returns True if exits successfully

    Args:
        details_dict: dict - a dictionary of app details
        filename: str - name of the file to save the details to
        folder: str, default is `APP_IDS_PATH` - name of the folder where `filename` is to be saved

    Returns:
        True if successful

    """

    output_target = folder / filename
    with open(output_target, "at") as output_file:  # append in text mode
        json.dump(details_dict, output_file, indent=2, default=str)

    return True


# %%
def load_app_details(filename: str, folder: str = APP_IDS_PATH) -> json:
    """
    Loads JSON file containing app details from <filename>. Returns JSON object.

    Args:
        filename: str - the name of the file where the app details are saved
        folder: str, default = `APP_IDS_PATH` - the folder where `filename` is located

    """

    result = {}
    with open(folder / filename, "rt") as input_file:
        result = json.load(input_file)

    return result


# %% [markdown]
# ## Download reviews for the apps in a given list of app ids
# These functions enable iterative downloading into a single file, which is appended as more reviews are added
#
# `get_playstore_app_reviews` gets up to 200 reviews for a single app and returns them as a list of dicts (JSON)
#
# `save_playstore_app_list_reviews` retrievs reviews for a given list of apps and saves them on-the-fly to a CSV file. If this function is interrupted, it can be called again witht the same parameters to resume where it left off. This function uses several helper functions:
# - _init_target_file: if the target CSV file doesn't exist, this function creates it
# - _init_log_file: initialises the log file that saves details about the progress of `save_playstore_app_list_reviews`
# - _process_review_grab: parses the reviews as they are retrieved by `save_playstore_app_list_reviews`, removing fields containing personal data, adding the app id and deduplicating.
# - _update_log_file: updates the log file with details about the progress of `save_playstore_app_list_reviews`
#
# `load_app_reviews` retrieves the app reviews from the CSV file saved by `save_playstore_app_list_reviews`

# %%
def get_playstore_app_reviews(
    target_app_id, how_many: int = 200, continuation_token: object = None
) -> (Iterator[dict], object):
    """
    Returns up to 200 reviews for a given app in the Google Play Store. If more reviews are available, 'continuation_token' is also returned, which can then
    be passed as a parameter to indicate where the function should resume.

    Args:
        target_app_id: id of the app for which you want to download reviews
        how_many: number of reviews you wish to download in one grab; defaults to 200, which is the maximum option
        continuation_token: indicates that the function should continue fetching reviews from this point

    Returns:
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
        logging.warning("No app id given")
        return False

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
def _init_target_file(target_file: str) -> list:
    """
    Helper function for `save_playstore_app_list_reviews` that initialises the file where reviews are to be saved.
    The target file is a CSV file that contains the reviews downloaded for apps

    Args:
        target_file: str - name of the file where reviews are to be saved. If the file already exists, the ids of
        existing reviews are returned

    """

    newfile = False if target_file.exists() else True

    existing_review_ids = list()
    with open(target_file, "at+", newline="") as csv_file:
        if newfile:
            csv_writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)
            csv_writer.writeheader()
        else:
            csv_file.seek(0)
            fetched_reviews = csv.DictReader(csv_file)
            existing_review_ids = [r["reviewId"] for r in fetched_reviews]

    return existing_review_ids


# %%
def _init_log_file(log_file: str, existing_get: bool) -> dict:
    """
    Helper function for `save_playstore_app_list_reviews` which initialises the log file and gets existing log information,
    if it exists.

    Args:
        log_file: str - name of the log file
        existing_get: bool - indicates whether to try and read data from an existing log file

    Returns:
        dict object:
            `app_id`: [
                "completed": bool,
                "continuation_token": continuation_token object,
                "latest_review_id": str,
                "latest_review_time": datetime,
                "downloaded": int
            ]
    """

    with open(log_file, "ab+") as log_file_handle:
        try:
            log_file_handle.seek(0)
            log_info = pickle.load(log_file_handle) if existing_get else dict()
        except:
            log_info = dict()

    return log_info


# %%
def _process_review_grab(
    app_id: str, review_fetch: list, existing_review_ids: list
) -> tuple:
    """
    Helper function for `save_playstore_app_list_reviews` that processes a set of reviews by:
    1. adding the app_id to each review
    2. removing userName and userImage
    3. removing existing reviews

    Args:
        app_id: str - the id of the that we're working with
        review_fetch: list - the list of reviews to be processed
        existing_review_ids: list - list of reviews that have already been downloaded to avoid duplication

    Returns:
        a list of processed reviews
        a list of review ids that have been processed

    """

    for index, review in enumerate(review_fetch):
        review["appId"] = app_id
        review.pop("userName", None)
        review.pop("userImage", None)
        if review["reviewId"] not in existing_review_ids:
            existing_review_ids.append(review["reviewId"])
        else:
            review_fetch.pop(index)
        review = {key: review[key] for key in FIELD_NAMES}

    return (review_fetch, existing_review_ids)


# %%
def _update_log_info(
    app_id: str,
    log_info: dict,
    continuation_token: object,
    review_fetch: list,
    first_pass: bool = False,
    completed: bool = False,
) -> dict:
    """
    Helper function for `save_playstore_app_list_reviews` that updates the `log_info` dict using data from the other args

    Args:
        app_id: the id of the app who's log info we're updating
        log_info: dict - a dict containing information about the status of the current app review get run. Schema is:
            app_id: {
                completed: bool,
                continuation_token: continuation token object from google-play-store API function
                latest_review_id: str - taken from the results of the API call
                latest_review_time: datetime - taken from the results of the API call
                downloaded: int - the number of reviews downloaded
            }
        continuation_token: object - returned by google-play-store API function
        review_fetch: list - list of reviews fetched by google-play-store API function
        first_pass: bool, default False - indicates whether this is the first time the function is being run, in which case it should
        save slightly different data
        completed: bool, default False - indicates whether all the reviews have been downloaded; like `first_pass`, this will affect
        the data that's saved

    Returns:
        a dict, derived from `log_info`, which has been updated according to the arguments provided

    """

    if first_pass:
        log_info.update(
            {
                app_id: {
                    "completed": False,
                    "continuation_token": continuation_token,
                    "latest_review_id": review_fetch[0]["reviewId"],
                    "latest_review_time": review_fetch[0]["at"],
                    "downloaded": len(review_fetch),
                }
            }
        )
    else:
        log_info[app_id]["continuation_token"] = continuation_token
        log_info[app_id]["downloaded"] = log_info[app_id]["downloaded"] + len(
            review_fetch
        )

    if completed:
        log_info[app_id]["completed"] = True
        log_info[app_id]["continuation_token"] = None

    return log_info


# %%
def save_playstore_app_list_reviews(
    app_id_list: list,
    filename: str,
    force_download: bool = False,
    run_quietly: bool = False,
) -> Iterator[dict]:

    """
    Saves and returns reviews for a given list of apps on the Play Store using their ids (e.g., as returned by
    get_playstore_app_ids)

    Args:
        app_id_list: list, list of app ids for the PLay Store
        filename: str, target file to save the returned reviews
        force_download: bool, true will force download of reviews, even if they've already been downloaded
        run_quietly: bool, true will mean that no status updates are provided

    The function returns a list of dict objects, which it also saves on-the-fly in the target file. `username` and
    `userimage` are removed from the reviews that are returned and saved for data privacy.

    `filename` is also used as the basis of a logfile `[filename].log`. If the function fails during run-time, it
    will resume where it left off, provided the same filename is given.

    """

    target_file = PROJECT_DIR / APP_IDS_PATH / filename
    log_file = PROJECT_DIR / APP_IDS_PATH / (Path(target_file).stem + ".log")

    # Initialise the target file and log file, retrieving data if they already exist
    existing_review_ids = _init_target_file(target_file)
    log_info = _init_log_file(log_file, bool(existing_review_ids))

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

        while (more_to_get or first_pass) and (keep_going or force_download):
            # get (up to 200) reviews for the app using the google-play-store API function, `get_playstore_app_reviews`
            review_fetch, continuation_token = get_playstore_app_reviews(
                target_app_id=app_id, continuation_token=continuation_token
            )

            if len(review_fetch) == 0:
                break

            # if we just grabbed a full 200 reviews, there are (probably) more to get, so set more_to_get to True
            more_to_get = True if len(review_fetch) == 200 else False

            # process and save what we've just downloaded:
            review_fetch, existing_review_ids = _process_review_grab(
                app_id, review_fetch, existing_review_ids
            )
            with open(target_file, "at", newline="") as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)
                csv_writer.writerows(review_fetch)

            # save a snapshot of where we are so we can resume later
            log_info = _update_log_info(
                app_id,
                log_info,
                continuation_token,
                review_fetch,
                first_pass=first_pass,
            )
            with open(log_file, "rb+") as log_file_handle:
                pickle.dump(log_info, log_file_handle)

            app_review_count += len(review_fetch)

            if not run_quietly:
                logging.info(
                    f"Retrieved {app_review_count} new reviews for {app_id}; fetched {log_info[app_id]['downloaded']} in total"
                )

            first_pass = False

        # if we've reached here and keep_going is true it's because we've downloaded all the results, so update logfile accordingly
        if keep_going and len(review_fetch) > 0:
            with open(log_file, "rb+") as log_file_handle:
                log_info = _update_log_info(
                    app_id, log_info, continuation_token, review_fetch, completed=True
                )
                pickle.dump(log_info, log_file_handle)

        if not run_quietly:
            logging.info(f"{app_id}: done")

    return True


# %%
def load_app_reviews(filename: str, folder: str = APP_IDS_PATH) -> list:
    """ """

    review_list = list()
    with open(folder / filename) as csv_reader:
        reviews = csv.DictReader(csv_reader)
        for review in reviews:
            review_list.append(review)

    return review_list


# %% [markdown]
# ## Putting it together
# ### Retrieving and using app ids via a set of downloaded web pages

# %%
app_ids = parse_folder_for_app_ids("play_store/kids_under_five")
save_app_ids(app_ids, "kids_under_five", "kids_under_five_ids.csv")

# %%
# _, app_ids = load_app_ids("kids_under_five_ids.csv")
app_details = get_playstore_app_details(app_ids)

# %%
save_app_details(app_details, "kids_under_five_details.json")

# %%
save_playstore_app_list_reviews(app_ids, "kids_under_five_reviews.csv")

# %% [markdown]
# ### Retrieving and using app ids via apps related to a seed
# Using `com.easypeasyapp.epappns` here

# %%
app_set = app_snowball("com.easypeasyapp.epappns", 5)
logging.info(f"Retrieved {len(app_set)} unique apps")
save_app_ids(app_set, "related_to_easypeasy", "related_to_easypeasy_ids.csv")
save_app_details(
    get_playstore_app_details(app_set), "related_to_easypeasy_details.json"
)
save_playstore_app_list_reviews(app_set, "related_to_easypeasy_reviews.csv")
