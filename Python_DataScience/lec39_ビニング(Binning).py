# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

years = [1990,1991,1992,2008,2012,2015,1987,1969,2013,2008,1999]

decade_bins = [1960,1970,1980,1990,2000,2010,2020]

decade_cat = pd.cut(years,decade_bins)

decade_cat

decade_cat.categories

pd.value_counts(decade_cat)

pd.cut(years,2)


