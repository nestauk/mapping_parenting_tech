"""
Helper variables and functions
"""
import pandas as pd
from mapping_parenting_tech import PROJECT_DIR, logging
from mapping_parenting_tech.utils import play_store_utils as psu
import numpy as np
import re

DATA_DIR = PROJECT_DIR / "outputs/data"
REVIEWS_DIR = DATA_DIR / "app_reviews"
INPUT_DIR = PROJECT_DIR / "inputs/data/play_store"
OUTPUT_DIR = PROJECT_DIR / "outputs/data"
TABLES_DIR = PROJECT_DIR / "outputs/figures/tables"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#### Dealing with cluster names and broader categories

cluster_names = [
    "Drawing and colouring",
    "Simple puzzles",
    "Literacy - English / ABCs",
    "Literacy - English / Reading and Stories",
    "Literacy - non-English",
    "Numeracy development",
    "Learning play",
    "General play",
    "Tracking babies' rhythms",
    "Helping babies to sleep",
    "Parental support",
    "Pregnancy tracking",
    "Fertility tracking",
    "Baby photos",
]

clusters_children = [
    "Drawing and colouring",
    "Simple puzzles",
    "Literacy - English / ABCs",
    "Literacy - English / Reading and Stories",
    "Literacy - non-English",
    "Numeracy development",
    "Learning play",
    "General play",
]

clusters_parents = [
    "Tracking babies' rhythms",
    "Helping babies to sleep",
    "Parental support",
    "Pregnancy tracking",
    "Fertility tracking",
    "Baby photos",
]

clusters_literacy = [
    "Literacy - English / ABCs",
    "Literacy - English / Reading and Stories",
    "Literacy - non-English",
]

clusters_simple_games = [
    "Drawing and colouring",
    "Simple puzzles",
]

clusters_play = [
    "Learning play",
    "General play",
]

# Mapping clusters to the type of user (Parents vs Children)
cluster_to_user_dict = dict(
    zip(
        clusters_parents + clusters_children,
        ["Parents"] * len(clusters_parents) + ["Children"] * len(clusters_children),
    )
)


def map_cluster_to_user(cluster: str) -> str:
    return cluster_to_user_dict[cluster]


#### Functions to load data
def get_relevant_apps() -> pd.DataFrame:
    """Get list of relevant ('focus') apps, as identified by automatic clustering and then manually reviewed"""
    return pd.read_csv(INPUT_DIR / "relevant_app_ids.csv", header=0)


def get_app_details() -> pd.DataFrame:
    """Loads a table with app information:

    # load app details
    # Note that...
    # 1. only apps of interest are retained
    # 2. the cluster for each app is added
    # 3. `score` and `minInstalls` are converted to float and int, respetively
    # 4. `score` is rounded to one decimal place
    """
    # Load apps that we've selected as relevant
    focus_apps = get_relevant_apps()
    focus_apps_list = focus_apps.appId.to_list()
    # Load the app details
    return (
        pd.DataFrame(psu.load_all_app_details())
        .T.drop("appId", axis=1)
        .reset_index()
        .rename(columns={"index": "appId"})
        .query("appId in @focus_apps_list")
        .merge(focus_apps, on="appId")
        .astype({"score": "float16", "minInstalls": "int64"})
        .assign(score_rounded=lambda x: np.around(x.score.to_list(), 1))
        .assign(user=lambda x: x.cluster.apply(map_cluster_to_user))
        .assign(releaseYear=lambda x: pd.to_datetime(x["released"]).dt.year)
    )


def get_app_reviews(test: bool = False) -> pd.DataFrame:
    # Load apps that we've selected as relevant
    focus_apps = get_relevant_apps()
    focus_apps_list = focus_apps.appId.to_list()
    focus_apps_list = focus_apps_list[0:10] if test else focus_apps_list
    return (
        psu.load_some_app_reviews(focus_apps_list)
        .assign(reviewYear=lambda x: pd.to_datetime(x["at"]).dt.year)
        .assign(reviewMonth=lambda x: pd.to_datetime(x["at"]).dt.month)
        .merge(focus_apps, on="appId")
        .merge(get_app_details()[["appId", "minInstalls"]], on="appId")
        .drop_duplicates("reviewId")
        .assign(user=lambda x: x.cluster.apply(map_cluster_to_user))
    )


def get_review_dates(app_reviews) -> pd.DataFrame:
    """Create a new dataframe, `reviewDates`, with the number of reviews for each app per year"""

    review_dates = (
        app_reviews.groupby(["appId", "reviewYear"])["appId"]
        .count()
        .unstack()
        .reset_index()
    )
    app_total_reviews = app_reviews.groupby(["appId"])["appId"].count()
    review_dates["total_reviews"] = review_dates["appId"].map(app_total_reviews)
    review_dates = review_dates.merge(get_relevant_apps(), on=["appId"])
    return review_dates


def save_data_table(table: pd.DataFrame, filename: str, folder=TABLES_DIR):
    table.to_csv(folder / f"{filename}.csv", index=False)


#### Helper functions for descriptive app analysis
def get_top_cluster_apps(
    app_df: pd.DataFrame, cluster: str, sort_by: str = "minInstalls", top_n: int = 10
) -> pd.DataFrame:
    """Show top apps in a specified cluster"""
    return (
        app_df.query("cluster == @cluster")
        .sort_values(sort_by, ascending=False)
        .head(top_n)
    )


def percentage_in_cluster(
    app_df: pd.DataFrame, cluster_names, return_percent: bool = True
) -> float:
    """Show percentage of all apps that are in specified clusters"""
    n = app_df.cluster.isin(cluster_names).sum()
    if return_percent:
        return np.round(n / len(app_df) * 100, 1)
    else:
        return n


def install_labels(number: float):
    if number < 1000:
        return f"{number: .0f}"
    if (number >= 1000) and (number < 1e6):
        return f"{number / 1000: .0f}K"
    if (number >= 1e6) and (number < 1e9):
        return f"{number / 1e+6: .0f}M"
    if number >= 1e9:
        return f"{number / 1e+9: .0f}B"


def install_labels_range(number: float):
    if number < 1000:
        return f"0-1K"
    if (number >= 1000) and (number < 10_000):
        return f"1K-10K"
    if (number >= 10_000) and (number < 100e3):
        return f"10K-100K"
    if (number >= 100_000) and (number < 1e6):
        return f"100K-1M"
    if (number >= 1e6) and (number < 10e6):
        return f"1M-10M"
    if number >= 10e6:
        return f"10M+"


def install_labels(number: float):
    if number < 10_000:
        return f"{number}"
    if (number >= 10_000) and (number < 1000e3):
        return f"{number/1e+3:.0f}K"
    if number >= 1000_000:
        return f"{number/1e+6:.0f}M"


app_install_ranges = ["0-1K", "1K-10K", "10K-100K", "100K-1M", "1M-10M", "10M+"]
app_install_categories = ["<100K", "100K-1M", "1M+"]

#### Helper functions for trends


def moving_average(
    timeseries_df: pd.DataFrame, window: int = 3, replace_columns: bool = False
) -> pd.DataFrame:
    """
    Calculates rolling mean of yearly timeseries (not centered)
    Args:
        timeseries_df: Should have a 'year' column and at least one other data column
        window: Window of the rolling mean
        rename_cols: If True, will create new set of columns for the moving average
            values with the name pattern `{column_name}_sma{window}` where sma
            stands for 'simple moving average'; otherwise this will replace the original columns
    Returns:
        Dataframe with moving average values
    """
    # Rolling mean
    df_ma = timeseries_df.rolling(window, min_periods=1).mean().drop("year", axis=1)
    # Create new renamed columns
    if not replace_columns:
        column_names = timeseries_df.drop("year", axis=1).columns
        new_column_names = ["{}_sma{}".format(s, window) for s in column_names]
        df_ma = df_ma.rename(columns=dict(zip(column_names, new_column_names)))
        return pd.concat([timeseries_df, df_ma], axis=1)
    else:
        return pd.concat([timeseries_df[["year"]], df_ma], axis=1)


def magnitude(time_series: pd.DataFrame, year_start: int, year_end: int) -> pd.Series:
    """
    Calculates signals' magnitude (i.e. mean across year_start and year_end)
    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
    Returns:
        Series with magnitude estimates for all data columns
    """
    magnitude = time_series.set_index("year").loc[year_start:year_end, :].mean()
    return magnitude


def percentage_change(initial_value, new_value):
    """Calculates percentage change from first_value to second_value"""
    return (new_value - initial_value) / initial_value * 100


def growth(
    time_series: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.Series:
    """Calculates a growth estimate
    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
    Returns:
        Series with smoothed growth estimates for all data columns
    """
    # Smooth timeseries
    df = time_series.set_index("year")
    # Percentage change
    return percentage_change(
        initial_value=df.loc[year_start, :], new_value=df.loc[year_end, :]
    )


def smoothed_growth(
    time_series: pd.DataFrame, year_start: int, year_end: int, window: int = 3
) -> pd.Series:
    """Calculates a growth estimate by using smoothed (rolling mean) time series
    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
        window: Moving average windows size (in years) for the smoothed growth estimate
    Returns:
        Series with smoothed growth estimates for all data columns
    """
    # Smooth timeseries
    ma_df = moving_average(time_series, window, replace_columns=True).set_index("year")
    # Percentage change
    return percentage_change(
        initial_value=ma_df.loc[year_start, :], new_value=ma_df.loc[year_end, :]
    )


def estimate_magnitude_growth(
    time_series: pd.DataFrame, year_start: int, year_end: int, window: int = 3
) -> pd.DataFrame:
    """
    Calculates signals' magnitude, estimates their growth and returns a combined dataframe
    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
        window: Moving average windows size (in years) for the smoothed growth estimate
    Returns:
        Dataframe with magnitude and growth trend estimates; magnitude is in
        absolute units (e.g. GBP 1000s if analysing research funding) whereas
        growth is expresed as a percentage
    """
    magnitude_df = magnitude(time_series, year_start, year_end)
    growth_df = smoothed_growth(time_series, year_start, year_end, window)
    combined_df = (
        pd.DataFrame([magnitude_df, growth_df], index=["magnitude", "growth"])
        .reset_index()
        .rename(columns={"index": "trend"})
    )
    return combined_df


def impute_empty_periods(
    df_time_period: pd.DataFrame,
    time_period_col: str,
    period: str,
    min_year: int,
    max_year: int,
) -> pd.DataFrame:
    """
    Imputes zero values for time periods without data
    Args:
        df_time_period: A dataframe with a column containing time period data
        time_period_col: Column containing time period data
        period: Time period that the data is grouped by, 'M', 'Q' or 'Y'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for
    Returns:
        A dataframe with imputed 0s for time periods with no data
    """
    max_year_data = np.nan_to_num(df_time_period[time_period_col].max().year)
    max_year = max(max_year_data, max_year)
    full_period_range = (
        pd.period_range(
            f"01/01/{min_year}",
            f"31/12/{max_year}",
            freq=period,
        )
        .to_timestamp()
        .to_frame(index=False, name=time_period_col)
        .reset_index(drop=True)
    )
    return full_period_range.merge(df_time_period, "left").fillna(0)


def minInstalls_coarse_partition(number: float):
    if number < 100e3:
        return "<100K"
    if (number >= 100e3) and (number < 1e6):
        return "100K-1M"
    if number > 1e6:
        return "1M+"


def result_dict_to_dataframe(
    result_dict: dict, sort_by: str = "counts", category_name: str = "cluster"
) -> pd.DataFrame:
    """Prepares the output dataframe"""
    return (
        pd.DataFrame(result_dict)
        .T.reset_index()
        .sort_values(sort_by)
        .rename(columns={"index": category_name})
    )


def get_category_time_series(
    time_series_df: pd.DataFrame,
    category_of_interest: str,
    time_column: str = "releaseYear",
    category_column: str = "cluster",
) -> dict:
    """Gets cluster or user-specific time series"""
    return (
        time_series_df.query(f"{category_column} == @category_of_interest")
        .drop(category_column, axis=1)
        .sort_values(time_column)
        .rename(columns={time_column: "year"})
        .assign(year=lambda x: pd.to_datetime(x.year.apply(lambda y: str(int(y)))))
        .pipe(
            impute_empty_periods,
            time_period_col="year",
            period="Y",
            min_year=2010,
            max_year=2021,
        )
        .assign(year=lambda x: x.year.dt.year)
    )


def get_estimates(
    time_series_df: pd.DataFrame,
    value_column: str = "counts",
    time_column: str = "releaseYear",
    category_column: str = "cluster",
    estimate_function=growth,
    year_start: int = 2019,
    year_end: int = 2020,
):
    """
    Get growth estimate for each category

    growth_estimate_function - either growth, smoothed_growth, or magnitude
    For growth, use 2019 and 2020 as year_start and year_end
    For smoothed_growth and magnitude, use 2017 and 2021
    """
    time_series_df_ = time_series_df[[time_column, category_column, value_column]]

    result_dict = {
        category: estimate_function(
            get_category_time_series(
                time_series_df_, category, time_column, category_column
            ),
            year_start=year_start,
            year_end=year_end,
        )
        for category in time_series_df[category_column].unique()
    }
    return result_dict_to_dataframe(result_dict, value_column, category_column)


def get_magnitude_vs_growth(
    time_series_df: pd.DataFrame,
    value_column: str = "counts",
    time_column: str = "releaseYear",
    category_column: str = "cluster",
):
    """Get magnitude vs growth esitmates"""
    df_growth = get_estimates(
        time_series_df,
        value_column=value_column,
        time_column=time_column,
        category_column=category_column,
        estimate_function=smoothed_growth,
        year_start=2017,
        year_end=2021,
    ).rename(columns={value_column: "Growth"})

    df_magnitude = get_estimates(
        time_series_df,
        value_column=value_column,
        time_column=time_column,
        category_column=category_column,
        estimate_function=magnitude,
        year_start=2017,
        year_end=2021,
    ).rename(columns={value_column: "Magnitude"})

    return df_growth.merge(df_magnitude, on=category_column)


def get_smoothed_timeseries(
    time_series_df: pd.DataFrame,
    value_column: str = "counts",
    time_column: str = "releaseYear",
    category_column: str = "cluster",
):
    """TODO: Finish this one!"""
    time_series_df_ = time_series_df[[time_column, category_column, value_column]]

    return pd.concat(
        [
            moving_average(
                get_category_time_series(
                    time_series_df_, category, time_column, category_column
                ),
                replace_columns=True,
            )
            for category in time_series_df[category_column].unique()
        ],
        ignore_index=True,
    )


#### Utils for app icons
def app_name_to_filename(name: str):
    return re.sub("\.", "_", name)


def filename_to_app_name(name: str):
    return re.sub("_", "\.", name)


def fetch_icons(df, output_dir):
    for i, row in tqdm(df.iterrows(), total=len(df)):
        filename = re.sub("\.", "_", row.appId)
        urllib.request.urlretrieve(row.icon, output_dir / f"{filename}.png")
        time.sleep(0.5)
