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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from mapping_parenting_tech.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import altair as alt
import pandas as pd
import numpy as np

# %%
driver = google_chrome_driver_setup()

# %%
# Generate a dataset
data = pd.DataFrame({"year": list(range(2000, 2022)), "value": np.random.rand(22)})

# %%
fig = (
    alt.Chart(data)
    .mark_line()
    .encode(x="year:O", y="value:Q", tooltip=["year", "value"])
).interactive()

# %%
save_altair(fig, "test_figure", driver)

# %%
