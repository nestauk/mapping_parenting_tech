# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from tqdm.notebook import tqdm
import os, sys

from mapping_parenting_tech import PROJECT_DIR, logging

INPUT_DIR = PROJECT_DIR / "inputs/data/play_store"
OUTPUT_DIR = PROJECT_DIR / "outputs/data"

# %%
chunk_size = 10**4
file = INPUT_DIR / "Google-Playstore.csv"
file_size = os.path.getsize(file)

start = dt.now()

print(f"Started at {start}")

data_types = {
    "App Name": str,
    "App Id": str,
    "Category": str,
    "Rating": float,
    "Rating Count": str,
    "Installs": str,
    "Minimum Installs": str,
    "Maximum Installs": str,
    "Free": bool,
    "Price": float,
    "Currency": str,
    "Size": str,
    "Minimum Android": str,
    "Developer Id": str,
    "Developer Website": str,
    "Developer Email": str,
    "Released": str,
    "Last Updated": str,
    "Content Rating": str,
    "Privacy Policy": str,
    "Ad Supported": bool,
    "In App Purchases": bool,
    "Editors Choice": bool,
    "Scraped Time": str,
}
date_cols = ["Released", "Last Updated", "Scraped Time"]

data_chunks = []
for data_chunk in pd.read_csv(
    file, dtype=data_types, parse_dates=date_cols, chunksize=chunk_size
):
    data_chunks.append(data_chunk)

    sys.stdout.write(
        f"Loaded {len(data_chunks)} chunks of (maybe) {int(file_size / chunk_size)}\r"
    )
    sys.stdout.flush()

print("\n")
print("Concatenating dataframe")

df = pd.concat(data_chunks, ignore_index=True)

end = dt.now()
duration = end - start

print(f"Completed at {end}\nStep took {duration}s")

# %%
df.shape

# %%
plot_df = df[["App Id", "Released"]]
plot_df["year_released"] = plot_df["Released"].dt.year
plot_df["Month released"] = plot_df["Released"].dt.month
plot_df = plot_df.groupby("year_released", as_index=False).agg(
    app_count=("App Id", "count"),
    months_in_year=("Month released", lambda x: x.nunique()),
)
plot_df["apps_per_month"] = plot_df["app_count"] / plot_df["months_in_year"]
plot_df["growth"] = plot_df["apps_per_month"].pct_change()
plot_df.plot.bar(x="year_released", y=["growth"], figsize=(10, 8), ylim=(0, 2.2))
print("Average growth: ", plot_df["apps_per_month"].mean())

# %%
plot_df.to_csv(OUTPUT_DIR / "play_store_growth.csv")

# %%
df["Minimum Installs"].fillna(0, inplace=True)

# %%
df = df.astype({"Minimum Installs": "int64"})

# %%
df.columns

# %%
cat_sizes = (
    df.groupby("Category")
    .agg(cat_size=("Category", "count"))
    .sort_values("cat_size", ascending=False)
)
cat_sizes = cat_sizes.assign(
    size_pc=(cat_sizes.cat_size / cat_sizes.cat_size.sum()) * 100
)

# %%
cat_sizes

# %%
import altair as alt

# %%
fig = (
    alt.Chart(cat_sizes.reset_index(), width=700, height=550)
    .mark_bar()
    .encode(x="size_pc:Q", y=alt.Y("Category:N", sort="-x"), tooltip="size_pc")
)
fig

# %%
# cat_sizes.reset_index().sort_values("size_pc", ascending=False).to_csv("category_sizes.csv")

# %%
app_installs_df = df.groupby("Minimum Installs").agg(
    installCount=("Minimum Installs", "count"), av_score=("Rating", "mean")
)
app_installs_df = app_installs_df[app_installs_df.index != 0]

# %%
base = alt.Chart(app_installs_df.reset_index(), width=700, height=700).encode(
    x=alt.X("Minimum Installs", scale=alt.Scale(type="log"))
)

counts = base.mark_point(size=60, filled=True).encode(
    alt.Y("installCount", axis=alt.Axis(title="Number of installs"))
)

scores = base.mark_line(stroke="red").encode(
    alt.Y("av_score", axis=alt.Axis(title="Average score"))
)

alt.layer(counts, scores).resolve_scale(y="independent")

# %%
fig = (
    alt.Chart(app_installs_df.reset_index(), width=700, height=500)
    .mark_point()
    .encode(
        x=alt.X("Minimum Installs", scale=alt.Scale(type="log", base=10)),
        y="installCount",
    )
)
fig + fig.transform_loess("Minimum Installs", "installCount").mark_line()

# %%
# basic_app_details = df[
#    [
#        "appId",
#        "cluster",
#        "minInstalls",
#        "score",
#        "ratings",
#        "reviews",
#        "price",
#        "free",
#        "containsAds",
#        "offersIAP",
#    ]
# ]
basic_app_details = df[
    [
        "App Id",
        "Category",
        "Rating",
        "Minimum Installs",
        "Free",
        "Price",
        "Ad Supported",
        "In App Purchases",
    ]
]

# %%
plotter = (
    basic_app_details.groupby("Category")
    .agg(
        cluster_size=("Category", "count"),
        free=("Free", "sum"),
        IAPs=("In App Purchases", "sum"),
        ads=("Ad Supported", "sum"),
    )
    .reset_index()
)

turn_to_pc = ["free", "ads", "IAPs"]
for i in turn_to_pc:
    plotter[f"{i}_pc"] = plotter[i] / plotter.cluster_size

plotter

# %%
data_map = {
    "free_pc": "Number of free apps",
    "IAPs_pc": "Number of apps with in-app purchases",
    "ads_pc": "Number of apps with ads",
}

# %%
mean_free = plotter.free_pc.mean()
mean_IAPs = plotter.IAPs_pc.mean()
mean_ads = plotter.ads_pc.mean()
print(
    f" Mean number of free apps:\t{mean_free*100}%\n",
    f"Mean number of apps with IAPs:\t{mean_IAPs*100}%\n",
    f"Mean number of apps with Ads:\t{mean_ads*100}%",
)

# %%
df = plotter.sort_values("free_pc", ascending=False)
bar_width = round(1 / len(data_map), 2) - 0.1

fig, ax = plt.subplots(figsize=(18, 9))
plt.setp(
    ax.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize="medium"
)
plt.grid(visible=True, axis="y", which="major")
ax.set_ylabel("Percentage of apps")

x = np.arange(len(df.Category))
for i, (key, value) in enumerate(data_map.items()):
    ax.bar(x + (i * bar_width), df[key], label=data_map[key], width=bar_width)

ax.set_xticks(x + (len(data_map) * bar_width) / len(data_map))
ax.set_xticklabels(df.Category.unique())


fig.legend(loc="upper left")
