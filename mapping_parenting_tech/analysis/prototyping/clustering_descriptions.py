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
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Clustering of app descriptions
# Takes descriptions of Play Store apps and clusters then accordingly, plotting visualisation/s of the clusters produced.
#
# Process is to...
# 1. embed the descriptions into high dimensional vectors using `SentenceTransformer`
# 2. reduce dimensionality using `UMAP`
# 3. cluster the reduced-dimension vectors using `hdbscan`
# 4. further reduce dimensionality to 2D, such that results can be visualised
# 5. use `KNeighborsClassifier ` from `scikit-learn` to assign apps that weren't clustered by `hdbscan` to their closest cluster
# 5. plot results of clustering
#
# Following the initial clustering, clusters have been given more meaningful names and grouped together according to whether they
# are relevant, interesting, unlikely, or to be discarded. The results are saved to CSV for manual review (e.g. in Google Sheets)

# %%
from sentence_transformers import SentenceTransformer
from mapping_parenting_tech.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import umap
import hdbscan
import altair as alt
import numpy as np
import pandas as pd
import pickle
from mapping_parenting_tech import PROJECT_DIR, logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_mutual_info_score

alt.data_transformers.disable_max_rows()

INPUT_DATA = PROJECT_DIR / "outputs/data"

# %%
# Load an embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# %%
# Load in the descriptions
details = pd.read_json(INPUT_DATA / "all_app_details.json", orient="index")
details.reset_index(inplace=True)
details.rename(columns={"index": "appId"}, inplace=True)

# %%
# Generate sentence embeddings (might take a few minutes for 1000s of sentences)
# description_embeddings = np.array(model.encode(details.description.to_list()))

# %%
filename = "description_embeddings-22-01-21.pickle"
with open(INPUT_DATA / filename, "rb") as f:
    description_embeddings = pickle.load(f)

# %%
# Check the shape of the sentence embeddings array
print(description_embeddings.shape)

# %%
# Create a 2D embedding
reducer = umap.UMAP(n_components=2, random_state=1)
embedding = reducer.fit_transform(description_embeddings)

# %%
# Create another low-dim embedding for clustering
reducer_clustering = umap.UMAP(n_components=50, random_state=1)
embedding_clustering = reducer_clustering.fit_transform(description_embeddings)

# %%
# Clustering with hdbscan
np.random.seed(3667)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=47,
    min_samples=1,
    cluster_selection_method="leaf",
    prediction_data=True,
)
clusterer.fit(embedding_clustering)
logging.info(
    f"{len(set(clusterer.labels_))} clusters with {sum([bool(x) for x in clusterer.labels_ if int(x) == -1])} unassigned apps"
)

# %%
# Prepare dataframe for visualisation
df = details.copy()
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]
df["cluster"] = [int(x) for x in clusterer.labels_]
df["cluster_confidence"] = [str(x) for x in clusterer.probabilities_]

# %%
cluster_col = df.columns.get_loc("cluster")

radius = 3
neigh = KNeighborsClassifier(n_neighbors=radius)

current_unassigned = len(df[df.cluster == -1].cluster.to_list())

cluster_list = df.cluster.to_list()

unassigned_indices = []
unassigned_indices.extend([i for i, x in enumerate(cluster_list) if x == -1])
logging.info(f"{len(unassigned_indices)} apps not in clusters")

X_list = [x for i, x in enumerate(embedding_clustering) if i not in unassigned_indices]
y_list = [i for i in cluster_list if i != -1]

neigh.fit(X_list, y_list)

for x in unassigned_indices:
    df.iat[x, cluster_col] = neigh.predict([embedding_clustering[x]])

print(f"{len(df[df.cluster == -1].cluster.to_list())} unassigned apps remain.")

# %%
# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
fig = (
    alt.Chart(df.reset_index(), width=725, height=725)
    .mark_circle(size=60)
    .encode(x="x", y="y", tooltip=["cluster", "appId", "summary"], color="cluster:N")
).interactive()

fig

# %%
# two lists of apps that are of particular interest/relevance. Apps are captured in these lists to check
# which clusters they are in, if so wished

apps_of_interest = [
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

extra_apps = [
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

# %% [markdown]
# Based on the result of the above clustering, the apps seem to fall into four broad categories:
# 1. **Relevant**: apps that seem to have clear relevance to AFS and/or HLE
# 2. **Interesting**: apps that could be of interest but need closer inspection
# 3. **Unlikely**: similar to 2, but are more likely to not be of interest
# 4. **Discard**: apps that will almost certainly not be relevant and can be discarded from future analyses
#
# These categories will be carried forward into a CSV file that can be imported into Google Sheets/Excel etc. for manual review

# %%
cluster_options = {
    "relevant": {
        2: "period and fertility tracking",
        7: "maths games + science education",
        8: "baby sleep",
        9: "learning [foreign?] language",
        10: "drawing / colouring in",
        12: "parent support / keeping in touch / photo sharing",
        13: "baby 'habit' tracking, inc. breastfeeding",
        17: "stories",
        18: "learning to read and write",
        19: "ABCs",
        27: "baby/toddler educational games",
    },
    "interesting": {
        5: "music apps, inc for toddlers",
        11: "young games",
        28: "young games",
    },
    "unlikely": {
        6: "student learning",
        16: "being organised/time management",
        20: "medical/health support (for adults)",
        22: "meditation/mindfulness/rest",
        25: "positivity/self development",
        26: "TBC",
    },
    "discard": {
        0: "Russian",
        1: "religion",
        3: "driving test",
        4: "web safety",
        14: "outdoor / mapping",
        15: "engineering/physics",
        21: "medical education",
        23: "medical 'organising'",
        24: "medical",
    },
}

# %%
relevant_clusters = set()
for a_id in set(extra_apps + apps_of_interest):
    if a_id in df.appId.values:
        relevant_clusters.update(df[df.appId == a_id].cluster.values)
        if df[df.appId == a_id].cluster.values == -1:
            print(f"{a_id} not in cluster")

relevant_clusters

# %%
driver = google_chrome_driver_setup()

# %%
# save the figure if so wished (Currently throws an error on MRH M1)
# save_altair(fig, "cluster_descriptions_", driver)

# %% [markdown]
# Following code repeats the steps above, but uses apps' summaries (rather than their descriptions) to do the clustering. This
# provides a point of validation/confirmation

# %%
filename = "summary_embeddings-22-01-21.pickle"
with open(INPUT_DATA / filename, "rb") as f:
    summary_embeddings = pickle.load(f)

# %%
# Create a 2D embedding
s_reducer = umap.UMAP(n_components=2, random_state=1)
s_embedding = s_reducer.fit_transform(summary_embeddings)

# %%
# Check the shape of the reduced embedding array
s_embedding.shape

# %%
# Create another low-dim embedding for clustering
s_reducer_clustering = umap.UMAP(n_components=50, random_state=1)
s_embedding_clustering = s_reducer_clustering.fit_transform(summary_embeddings)

# %%
# Clustering with hdbscan
np.random.seed(3667)
s_clusterer = hdbscan.HDBSCAN(
    min_cluster_size=42,
    min_samples=1,
    cluster_selection_method="leaf",
    prediction_data=True,
)
s_clusterer.fit(s_embedding_clustering)
logging.info(
    f"{len(set(s_clusterer.labels_))} clusters with {sum([bool(x) for x in s_clusterer.labels_ if int(x) == -1])} unassigned apps"
)

# %%
# Prepare dataframe for visualisation
s_df = details.copy()
s_df["x"] = s_embedding[:, 0]
s_df["y"] = s_embedding[:, 1]
s_df["s_cluster"] = [int(x) for x in s_clusterer.labels_]
s_df["s_cluster_confidence"] = s_clusterer.probabilities_

# %%
s_cluster_col = s_df.columns.get_loc("s_cluster")

radius = 3
neigh = KNeighborsClassifier(n_neighbors=radius)

current_unassigned = len(s_df[s_df.s_cluster == -1].s_cluster.to_list())

s_cluster_list = s_df.s_cluster.to_list()

unassigned_indices = []
unassigned_indices.extend([i for i, x in enumerate(s_cluster_list) if x == -1])

X_list = [
    x for i, x in enumerate(s_embedding_clustering) if i not in unassigned_indices
]
y_list = [i for i in s_cluster_list if i != -1]

neigh.fit(X_list, y_list)

for x in unassigned_indices:
    s_df.iat[x, s_cluster_col] = neigh.predict([s_embedding_clustering[x]])

print(f"{len(s_df[s_df.s_cluster == -1].s_cluster.to_list())} unassigned apps remain.")

# Check the correspendence between both clustering approaches
print(adjusted_mutual_info_score(s_df.s_cluster.to_list(), df.cluster.to_list()))

# %%
# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
s_fig = (
    alt.Chart(s_df.reset_index(), width=750, height=750)
    .mark_circle(size=60)
    .encode(
        x="x", y="y", tooltip=["s_cluster", "appId", "summary"], color="s_cluster:N"
    )
).interactive()

s_fig

# %%
s_relevant_clusters = set()
for a_id in set(extra_apps + apps_of_interest):
    if a_id in s_df.appId.values:
        s_relevant_clusters.update(s_df[s_df.appId == a_id].s_cluster.values)
        if s_df[s_df.appId == a_id].s_cluster.values == -1:
            print(f"{a_id} not in cluster")

s_relevant_clusters

# %%
s_cluster_options = {
    "relevant": {
        2: "stories/lullabies/music",
        4: "learning through play / learning to code/ professional development",
        6: "health / medical (adult)",
        8: "animal-related (identify, sounds etc.)",
        9: "play / make-belief",
        10: "sleep (helping babies to)",
        11: "education / parenting support",
        17: "pregnancy",
        18: "baby monitoring / development tracking (photos)",
        19: "number games",
        24: "learning games for toddlers",
        26: "learning games (mostly ABCs)",
        27: "learning ABCs",
        28: "learning English (inc. as a foreign language)",
    },
    "interesting": {
        1: "drawing / colouring",
        12: "period / fertility tracking",
        25: "learning games",
    },
    "unlikely": {
        6: "student learning",
        16: "meditation / calm / sleep (for adults)",
        23: "mental health",
    },
    "discard": {
        0: "Setting light levels (eg screen brightness)",
        3: "driving test",
        5: "engineering",
        7: "connecting with people",
        13: "personal organisation",
        14: "child safety / web filtering",
        15: "outdoor / mapping",
        20: "religion",
        21: "minfulness / Buddhism",
        22: "motivation",
    },
}


# %%
def get_cluster_info(needle: int, haystack: dict):
    """
    Get information about a cluster from the 'relevance' dictionaries based on a cluster's id

    Example usage: relevance, purpose = get_cluster_info(4, s_cluster_options)
        Returns: relevance = "relevant"; purpose = "learning through play / learning to code/ professional development"

    Args:
        needle, int: the cluster to search for
        haystack, dict: the dict with the cluster info, within which you're looking

    Returns:
        Tuple(the key of the cluster that you're looking for, the purpose of the cluster)
    """

    for k, v in haystack.items():
        if needle in v.keys():
            return (k, v[needle])

    return (0, 0)


# %%
cluster_relevance_ = list()
s_cluster_relevance_ = list()

cluster_type_ = list()
s_cluster_type_ = list()

for i, app_id in enumerate(df.appId.tolist()):
    relevance, purpose = get_cluster_info(int(df.cluster.iloc[i]), cluster_options)
    cluster_relevance_.append(relevance)
    cluster_type_.append(purpose)

    relevance, purpose = get_cluster_info(
        int(s_df.s_cluster.iloc[i]), s_cluster_options
    )
    s_cluster_relevance_.append(relevance)
    s_cluster_type_.append(purpose)

df["cluster_relevance"] = cluster_relevance_
df["cluster_purpose"] = cluster_type_
df["s_cluster_relevance"] = s_cluster_relevance_
df["s_cluster_purpose"] = s_cluster_type_
df["s_cluster_confidence"] = s_df["s_cluster_confidence"]

# %%

fig = (
    alt.Chart(df.reset_index(), width=750, height=750)
    .mark_circle(size=60)
    .encode(
        x="x",
        y="y",
        tooltip=["cluster", "appId", "summary"],
        color="cluster_relevance:N",
    )
).interactive()

fig

# %%
df.sample(10)

# %%
# Save list of apps to review
# df.to_csv(INPUT_DATA / "apps_to_review.csv", columns=["appId", "title", "summary", "description", "score", "installs", "genre", "cluster_purpose", "cluster_confidence", "cluster_relevance", "s_cluster_purpose", "s_cluster_confidence", "s_cluster_relevance"], index=False)
