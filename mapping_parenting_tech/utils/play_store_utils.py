"""
Functions to enable the following:
 1. retrieve the ids of apps from the Play Store
 2. save and load app ids so they can be saved, retrieved and used later
 3. download details for an app/s using its/their app id/s
 4. save and load app details so they can be saved, retrieved and used later

In step 4, app details are saved on-the-fly, as they are retrieved. As this step can take some time for 100s of apps or for apps with 1,000s of reviews (or both), the process can timeout or fail unexpectedly. Saving app details as they are retrieved allows the step to be resumed part-way through as a log file is saved with details of progress for each app.

Note that app ids, details and reviews are managed independently and may not be synchronised. A list of 100 app ids does not mean that those 100 apps have had their details and/or reviews saved too. It's also possible that some apps' details and/or reviews might have been saved, yet their ids are not saved in the app id list. Two functions update *from* the app ids (`update_all_app_details` and `update_all_app_reviews`), which use the app id list as their basis for what to fetch. However, no functions exist in any other direction - i.e., if there are more app details than app ids, there is no function to add the missing app ids to the id list.

 These functions are dependent on the [google_play_scraper](https://pypi.org/project/google-play-scraper/) library.
"""

# Do imports and set file locations
import re
import json
import pickle
import csv
import pandas as pd
from google_play_scraper import Sort, app, reviews
from tqdm import tqdm
from pathlib import Path
from mapping_parenting_tech import PROJECT_DIR, logging
from typing import Iterator

APP_IDS_PATH = PROJECT_DIR / "outputs/data"
DATA_DIR = PROJECT_DIR / "outputs/data"
REVIEWS_DIR = DATA_DIR / "app_reviews"
LOG_FILE = DATA_DIR / "app_review_getting.log"
CONTINUATION_TOKENS_DATA = DATA_DIR / "app_review_continuation_tokens.pickle"

# Field names used in the reviews' CSV file
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

# Lists of apps. Interesting apps have been given by AFS; extra apps have been crowd sourced from Nesta staff
INTERESTING_APPS = [
    "com.easypeasyapp.epappns",
    "com.lingumi.lingumiplay",
    "com.learnandgo.kaligo",
    "com.kamicoach",
    "com.storytoys.myveryhungrycaterpillar.free.android.googleplay",
    "uk.co.bbc.cbeebiesgoexplore",
    "tv.alphablocks.numberblocksworld",
    "com.acamar.bingwatchplaylearn",
    "uk.org.bestbeginnings.babybuddymobile",
    "com.mumsnet.talk",
    "com.mushuk.mushapp",
    "com.teampeanut.peanut",
    "com.flipsidegroup.nightfeed.paid",
]

EXTRA_APPS = [
    "org.twisevictory.apps",
    "com.domustechnica.tww.sleep",
    "com.domustechnica.backtoyou",
    "twwaudioapp.qsd.com.twwaudio.v2",
    "com.tinybeans",
    "com.edokicademy.montessoriacademy",
    "uk.co.bbc.cbeebiesplaytimeisland",
    "uk.co.bbc.cbeebiesgoexplore",
    "uk.co.bbc.cbeebiesgetcreative",
    "air.uk.co.bbc.cbeebiesstorytime",
    "com.teachyourmonstertoread.tmapp",
    "abc_kids.alphabet.com",
    "com.duckduckmoosedesign.kindread",
    "com.teampeanut.peanut",
    "com.ovuline.pregnancy",
    "com.ovuline.parenting",
    "com.pgs.emmasdiary",
    "com.backthen.android",
    "uk.co.happity.happity",
    "uk.co.hoop.android",
    "com.nighp.babytracker_android",
    "com.amila.parenting",
    "uk.org.bestbeginnings.babybuddymobile",
    "com.rvappstudios.baby.games.piano.phone.kids",
    "com.educational.baby.games",
    "com.happytools.learning.kids.games",
    "org.msq.babygames",
    "com.thepositivebirthcompany.freyasurgetimer",
    "com.mathletics",
    "com.huckleberry_labs.app",
    "com.propagator.squeezy",
    "com.sitekit.eRedBook",
]


def get_playstore_app_ids_from_html(filename: str) -> list:
    """
    Looks inside 'inputs/data' in the project directory for a given HTML file (filename) and scans that file
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


def parse_folder_for_app_ids(folder: str) -> list:
    """
    Scans a folder for HTML files and extracts the app ids from those files, returning a unified, de-duplicated list
    of app ids.

    Args:
        folder: str - the folder to be scanned. This should be located in 'inputs/data' in a Nesta cookiecutter project

    Returns:
        A list object of Play Store app ids, each as a str

    """

    target_folder = PROJECT_DIR / "inputs/data" / folder
    app_id_list = list()

    # iterate .html files in the given folder, pass them to get_playstore_app_ids and assign results
    # by extending app_id_list
    app_id_list.extend(
        [
            get_playstore_app_ids(file_name)
            for file_name in target_folder.iterdir()
            if (file_name.is_file() and file_name.suffix == ".html")
        ]
    )

    # flatten app_id_list (each file parsed will have returned a separate list)
    app_id_list = [item for sublist in app_id_list for item in sublist]

    # use a set to remove duplicates in the return
    return list(set(app_id_list))


def app_snowball(seed_app_id: str, depth: int = 5, __current_depth: int = 1) -> list:
    """
    Retrieves ids of Play Store apps related to `seed_app_id` by calling itself recursively.

    Args:
        seed_app_id: str - the app id of the app of interest
        depth: int, default = 5 - the depth of recursion. This will increase the number of apps interrogated (and
        therefore the time taken for the initial call to complete) exponentially
        __current_depth: used for recursion, should be left blank by user

    Returns:
        a list of app ids
    """

    app_details = app(seed_app_id, country="gb")
    similar_apps = app_details["similarApps"]

    snowball = set([seed_app_id])
    try:
        snowball.update(similar_apps)
    except:
        logging.warning(f"{seed_app_id} had a problem. Maybe it has no related apps.")

    if __current_depth < depth:
        for this_app in similar_apps:
            snowball.update(app_snowball(this_app, depth, (__current_depth + 1)))

    return list(snowball)


def update_all_app_id_list(app_id_list: list, dry_run: bool = False) -> set:
    """
    Takes a list of apps and adds them to the existing app list, removing any duplicates.

    Args:
        app_id_list: list of app ids for the Play Store
        dry_run: bool, default is False, if True doesn't save the list but does update in memoey

    Returns:
        A set of all app ids, i.e., it contains the original app ids plus the new ones given, minus any duplicates

    """

    logging.info(f"Attempting to insert {len(app_id_list)} apps to app list")

    app_id_df = pd.read_csv(DATA_DIR / "all_app_ids.csv", index_col=None, header=0)
    app_ids = app_id_df[app_id_df.columns[0]].to_list()

    orig_length = len(app_ids)
    app_ids.extend(app_id_list)

    logging.info(f"Added {len(set(app_ids)) - orig_length} new app ids")

    if dry_run == False:
        app_id_df = pd.DataFrame(set(app_ids), columns=["appId"])
        app_id_df.to_csv(DATA_DIR / "all_app_ids.csv", index=False)
        logging.info(f"App id list saved successfully")

    return set(app_ids)


def load_all_app_ids() -> set:
    """
    Loads the saved ids of all apps and returns them in a set:

    Takes no arguments; returns a single set.
    """

    app_ids_df = pd.read_csv(DATA_DIR / "all_app_ids.csv", index_col=None, header=0)
    return set(app_ids_df[app_ids_df.columns[0]].to_list())


def is_app_in_list(app_list: list) -> list:
    """
    Identifies app ids that have been saved.

    Arguments
        app_list: list - a list of app ids to be checked

    Returns:
        list of boolean values, reflecting whether the app id in the equivalent position in `app_list` is present.

    For example:
        If we want to check three apps, we pass their ids to the function:
        check_apps = is_app_in_list(["app 1 id", "app 2 id", app 3 id"])

        If only the id for app 3 is in the list, `check_apps` will return:
        [False, False, True]
    """

    all_app_ids = load_all_app_ids()
    r_list = list()

    for x in app_list:
        to_add = True if x in all_app_ids else False
        r_list.append(to_add)

    return r_list


def get_playstore_app_details(app_id_list: list) -> dict:
    """
    Uses `google-play-scraper` (https://pypi.org/project/google-play-scraper/) to retrieve details about apps given in
    `app_id_list` and returns dict of app details

    Args:
        app_id_list: list - a list of app ids from which details will be retrieved

    Returns:
        A dict containing app details; the key for each dict item is the app id. Each dict includes:
        - title
        - description
        - summary
        for specifics of all data returned, see https://pypi.org/project/google-play-scraper/

    """

    all_app_details = dict()
    remove_apps = list()

    for app_id in tqdm(
        app_id_list,
        desc="Retrieving app details",
    ):
        try:
            app_details = app(app_id, country="gb")
            all_app_details.update({app_id: app_details})

        except Exception as e:  # needs modifying to capture specific errors
            logging.warning(f"Error on app id {app_id}: {e} {repr(e)}")
            remove_apps.append(app_id)

    return all_app_details


def update_all_app_details(force_all: bool = False) -> dict:
    """
    Updates the file containing app details, ensuring that all apps saved in all_app_ids.csv have corresponding details
    saved in all_app_details.json

    Args:
        force_all: bool, default is False, if set to True, will download details for ALL apps

    Returns:
        A dict containing details for **all** apps. The key for each dict item is the app id.

    """

    all_app_ids = load_all_app_ids()
    existing_app_details = load_all_app_details()

    target_list = (
        all_app_ids
        if force_all
        else [x for x in all_app_ids if x not in existing_app_details.keys()]
    )

    logging.info(f"There are {len(target_list)} apps whose details will be updated.")

    existing_app_details.update(get_playstore_app_details(target_list))

    with open(DATA_DIR / "all_app_details.json", "wt") as f:
        json.dump(existing_app_details, f, indent=2, default=str)

    logging.info("Updated app details file.")
    return existing_app_details


def load_all_app_details() -> dict:
    """
    Loads JSON file containing app details and returns a dict object.

    Args:
        None

    Returns:
        A dict of app details for **all** apps. The key for each item is the app id.

    """

    with open(DATA_DIR / "all_app_details.json") as f:
        details = json.load(f)

    return details


def get_playstore_app_reviews(
    target_app_id, how_many: int = 200, continuation_token: object = None
) -> (Iterator[dict], object):
    """
    Returns up to 200 reviews for a given app in the Google Play Store. If more reviews are available,
    'continuation_token' is also returned, which can then be passed as a parameter to indicate where the
    function should resume.

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

    # NB: code can be extended/improved. `not_fetched` intended to allow code to loop until in case of a timeout error,
    # but need to consider other errors that could be raised and how to respond to them
    not_fetched = True
    while not_fetched:
        try:
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
            not_fetched = False
        except KeyboardInterrupt as e:
            not_fetched = False
        except Exception as e:
            logging.warning(
                f"An error occured trying to fetch reviews for {target_app_id}: {repr(e)}"
            )
            not_fetched = False

    return (fetch_reviews, continuation_token)


def _init_log_info() -> dict:
    """
    Helper function for `save_playstore_app_list_reviews` which gets existing log information, if it exists,
    or initialises the file if being run for the first time. The human readable log file does not contain the
    'continuation token' required to resume downloading reviews for an app whose progress was interrupted. Thus,
    this function add the continuation token/s to the dict for any apps that need to be resumed.

    Args:
        None

    Returns:
        dict object:
            `app_id`: {
                "completed": bool,
                "latest_review_id": str,
                "latest_review_time": datetime,
                "downloaded": int,
                "continuation_token": continuation_token
            }(,{...})
    """

    log_info = dict()

    if LOG_FILE.exists():
        with open(LOG_FILE, "rt") as f:
            f.seek(0)
            log_info.update(json.load(f))

    if CONTINUATION_TOKENS_DATA.exists():
        with open(CONTINUATION_TOKENS_DATA, "rb") as f:
            try:
                c_tokens = pickle.load(f)
                for app_id in [
                    k for k, v in log_info.items() if v["completed"] == False
                ]:
                    log_info[app_id].update({"continuation_token": c_tokens[app_id]})
            except:
                logging.warning(f"Expected to find continuation token/s but didn't.")

    return log_info


def _init_target_file(target_file: str) -> list:
    """
    Helper function for `save_playstore_app_list_reviews` that initialises the file where reviews are to be saved.
    The target file is a CSV file that contains the reviews downloaded for a single app

    Args:
        target_file: str - name of the file where reviews are to be saved. If the file already exists, the ids of
        existing reviews are returned

    """

    newfile = False if target_file.exists() else True

    existing_review_ids = list()
    if newfile:
        with open(target_file, "wt", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)
            csv_writer.writeheader()
    else:
        pd_col = pd.read_csv(target_file, usecols=["reviewId"])
        existing_review_ids = pd_col["reviewId"].to_list()

    return existing_review_ids


def _update_app_log_info(
    review_fetch: list,
    total_downloads: int,
    first_pass: bool = False,
    continuation_token: object = None,
    completed: bool = False,
) -> dict:
    """
    Helper function for `save_playstore_app_list_reviews` that updates the `log_info` dict using data from the other args

    Args:
        review_fetch: list - list of reviews fetched by google-play-store API function
        total_downloads: the total number of reviews downloaded for this app
        first_pass: bool, default False - indicates whether this is the first time the function is being run, in which
        case it should save slightly different data
        continuation_token: object - returned by google-play-store API function
        completed: bool, default False - indicates whether all the reviews have been downloaded; like `first_pass`, this
        affects the data that's saved

    Returns:
        a dict, derived from `log_info`, which has been updated according to the arguments provided

    """

    # set default info
    app_log_info = {
        "completed": False,
        "downloaded": total_downloads,
        "continuation_token": continuation_token,
    }

    if first_pass and len(review_fetch) > 0:
        app_log_info.update(
            {
                "latest_review_id": review_fetch[0]["reviewId"],
                "latest_review_time": review_fetch[0]["at"],
            }
        )

    if completed:
        app_log_info.update({"completed": True})
        app_log_info.pop("continuation_token", None)

    return app_log_info


def _process_review_grab(
    app_id: str,
    review_fetch: list,
    existing_review_ids: list,
    force_download: bool = False,
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
        force_download: downloads all reviews, regardless of whether they exist in `existing_review_ids`

    Returns:
        a dict of processed reviews
        a list of review ids that have been saved

    """

    processed_reviews = list()
    for review in review_fetch:
        if (review["reviewId"] not in existing_review_ids) or force_download:
            review.update({"appId": app_id})
            existing_review_ids.append(review["reviewId"])
            new_review = {key: review[key] for key in FIELD_NAMES}
            processed_reviews.append(new_review)

    return (processed_reviews, existing_review_ids)


def _save_log_info(updated_log_info: dict):
    """
    Helper function for `save_playstore_app_list_reviews` that saves the log information for the function as
    it progresses. `continuation_token` is stripped from the log_info dict and saved separately, but only for
    apps whose reviews are not fully downloaded. `continuation_token`s are saved in a separate pickle file.

    """

    c_tokens = dict()

    # strip continuation tokens from the log info and, for incomplete downloads, save them separately
    for app_id in updated_log_info.keys():
        if "continuation_token" in updated_log_info[app_id].keys():
            c_token = updated_log_info[app_id].pop("continuation_token", None)
            if updated_log_info[app_id]["completed"] == False:
                c_tokens.update({app_id: c_token})

    try:
        with open(LOG_FILE, "wt") as f:
            json.dump(updated_log_info, f, indent=2, default=str)

        with open(CONTINUATION_TOKENS_DATA, "wb") as f:
            pickle.dump(c_tokens, f)

    except:
        logging.warning(
            "Interrupted during writing log file. Check log file for corruption."
        )

    # unclear whether this is necessary, but this adds back continuation tokens that were stripped above
    for k in c_tokens.keys():
        updated_log_info[k].update({"continuation_token": c_tokens[k]})


def save_playstore_app_list_reviews(
    app_id_list: list, force_download: bool = False, run_quietly: bool = False
) -> bool:

    """
    Saves and returns reviews for a given list of apps on the Play Store using their ids (e.g., as returned by
    get_playstore_app_ids)

    Args:
        app_id_list: list, list of app ids for the PLay Store
        force_download: bool, true will force download of reviews, even if they've already been downloaded
        run_quietly: bool, true will mean that no status updates are provided

    The function returns True on completing successfully.

    Downloaded reviews are saved on-the-fly in a separate file for each app; the file is named [app_id].csv.
    For data privacy, `username` and `userimage` are removed from the reviews that are returned and saved.

    If the function fails during run-time, it will resume where it left off.

    """

    # Initialise log info, retrieving existing information
    log_info = _init_log_info()

    for i, app_id in enumerate(app_id_list, start=1):
        target_file = REVIEWS_DIR / f"{app_id}.csv"
        existing_review_ids = _init_target_file(target_file)

        logging.info(
            f"Starting to download reviews for {app_id} ({i} of {len(app_id_list)})"
        )

        # reset variables ahead of downloading an app's reviews
        first_pass = True
        more_to_get = True
        continuation_token = None
        review_fetch = []
        app_review_count = len(existing_review_ids) if force_download == False else 0

        # Is the app in the logfile?
        if app_id in log_info.keys():
            # if force_download is in effect, we should start at the beginning, otherwise pick up where we left off
            if log_info[app_id]["completed"] == False:
                continuation_token = (
                    log_info[app_id]["continuation_token"]
                    if not force_download
                    else None
                )
            first_pass = False
        else:
            # if not, set up a log entry for it
            log_info.update({app_id: {"completed": False}})

        while first_pass or force_download or more_to_get:
            # get (up to 200) reviews for the app using the google-play-store API function, `get_playstore_app_reviews`
            review_fetch, continuation_token = get_playstore_app_reviews(
                target_app_id=app_id, continuation_token=continuation_token
            )

            # process and save what we've just downloaded:
            processed_reviews, existing_review_ids = _process_review_grab(
                app_id, review_fetch, existing_review_ids, force_download
            )

            # if we got zero reviews or all reviews fetched already exist, exit the loop
            if ((len(processed_reviews) == 0) and (force_download == False)) or (
                len(review_fetch) == 0
            ):
                break

            app_review_count += len(processed_reviews)

            # if we grabbed fewer than 200 reviews, there are (probably) no more to get, so set more_to_get to
            # False, otherwise True
            more_to_get = True if len(processed_reviews) == 200 else False

            with open(target_file, "at", newline="") as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)
                csv_writer.writerows(processed_reviews)

            # save a snapshot of where we are so we can resume later
            log_info[app_id].update(
                _update_app_log_info(
                    processed_reviews,
                    app_review_count,
                    first_pass=first_pass,
                    continuation_token=continuation_token,
                )
            )
            _save_log_info(log_info)

            if not run_quietly and (app_review_count % 5000 == 0):
                logging.info(
                    f"Retrieved {app_review_count} new reviews for {app_id}; fetched {log_info[app_id]['downloaded']} in total"
                )

            first_pass = False

        # update logfile
        log_info[app_id].update(
            _update_app_log_info(processed_reviews, app_review_count, completed=True)
        )
        _save_log_info(log_info)

        if not run_quietly:
            logging.info(
                f"Completed fetching reviews for {app_id}; {app_review_count} were downloaded in total"
            )

    return True


def load_all_app_reviews() -> pd.DataFrame():
    """
    Loads all app reviews into a Pandas dataframe.

    This may take some time for 1,000s of apps (e.g., ~2.5minutes for reviews of 3,500 apps)

    Returns:
        Pandas dataframe containing all reviews for all apps - note, this can be quite large!
    """

    all_reviews = pd.DataFrame()
    review_files = REVIEWS_DIR.iterdir()
    reviews_df_list = list()

    logging.info("Loading files")
    for file in tqdm(review_files):
        reviews_df = pd.read_csv(file, header=0, index_col=None)
        reviews_df_list.append(reviews_df)

    logging.info("Concatenating data")
    all_reviews = pd.concat(reviews_df_list, axis=0, ignore_index=True)

    return all_reviews


def load_some_app_reviews(app_ids: list) -> pd.DataFrame:
    """
    Load reviews for a given set of Play Store apps

    Args:
        app_ids: list - a list of app ids whose reviews will be loaded

    Returns:
        Pandas DataFrame

    """

    reviews_df_list = []
    logging.info("Reading app reviews")
    for app_id in tqdm(app_ids, position=0):
        try:
            review_df = pd.read_csv(REVIEWS_DATA / f"{app_id}.csv")
        except FileNotFoundError:
            logging.info(f"No reviews found for {app_id}")
            review_df = []
        reviews_df_list.append(review_df)

    logging.info("Concatenating reviews")
    reviews_df = pd.concat(reviews_df_list)
    del reviews_df_list
    logging.info("Reviews loaded")

    return reviews_df


def list_missing_app_reviews() -> set():
    """
    Returns a set of apps whose ids are saved but not their reviews, plus apps whose review fetch is incomplete.

    This function takes no arguments.

    Returns a set of app ids, which can be passed straight to `save_playstore_app_list_reviews`
    """

    t_df = load_all_app_reviews()
    reviewed_apps = set(t_df.appId.to_list())
    del t_df

    all_apps = load_all_app_ids()

    apps_to_get = [x for x in all_apps if x not in reviewed_apps]

    log_details = _init_log_info()
    apps_to_get.extend(k for k, v in log_details.items() if v["completed"] == False)

    return set(apps_to_get)
