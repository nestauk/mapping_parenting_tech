# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''mapping_parenting_tech'': conda)'
#     language: python
#     name: python3812jvsc74a57bd09d0629e00499ccf218c6720a848e8111287e8cbf09d1f93118d5865a19869c30
# ---

# %%
from mapping_parenting_tech.utils import (
    play_store_utils as psu,
    text_preprocessing_utils as tpu,
)
from mapping_parenting_tech import PROJECT_DIR, logging
from pathlib import Path
from tqdm import tqdm

import altair as alt
import pandas as pd
import numpy as np

INPUT_DATA = PROJECT_DIR / "inputs/data/play_store"
REVIEWS_DATA = PROJECT_DIR / "outputs/data/app_reviews"

# %%
app_df = pd.read_csv(INPUT_DATA / "relevant_app_ids.csv")
app_ids = app_df["app_id"].to_list()
app_clusters = set(app_df["cluster"].to_list())


# %%
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
    for app_id in tqdm(app_ids):
        try:
            review_df = pd.read_csv(REVIEWS_DATA / f"{app_id}.csv")
        except FileNotFoundError:
            logging.info(f"No reviews for {app_id}")
            review_df = []
        reviews_df_list.append(review_df)

    logging.info("Concatenating reviews")
    reviews_df = pd.concat(reviews_df_list)
    del reviews_df_list
    logging.info("Reviews loaded")
    return reviews_df


# %%
app_reviews_df = load_some_app_reviews(app_ids)
app_reviews_df.rename(columns={"appId": "app_id"}, inplace=True)

# %%
app_review_counts = app_reviews_df.groupby("app_id")["content"].agg("count")
app_df.merge(app_review_counts, left_on="app_id", right_index=True)

# %%
stripplot = (
    alt.Chart(app_df, width=40)
    .mark_circle(size=20)
    .encode(
        x=alt.X(
            "jitter:Q",
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y(
            "review_count:Q",
            # scale=alt.Scale(range=(0,10000)),
        ),
        color=alt.Color("cluster:N", legend=None),
        column=alt.Column(
            "cluster:N",
            header=alt.Header(
                labelAngle=-90,
                titleOrient="top",
                labelOrient="bottom",
                labelAlign="right",
                labelPadding=3,
            ),
        ),
        tooltip=["app_id"],
    )
    .transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    )
    .configure_facet(spacing=0)
    .configure_view(stroke=None)
)

stripplot


# %%
app_df.groupby("cluster")["review_count"].agg(
    [
        ("total_reviews", "sum"),
        ("mean", "mean"),
        ("median", np.median),
        ("ninety_five_pc", lambda x: np.percentile(x, q=95)),
    ]
)

# %%
