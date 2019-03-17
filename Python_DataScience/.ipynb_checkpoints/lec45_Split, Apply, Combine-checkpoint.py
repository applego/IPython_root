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

# <img src="./lec45_split_apply_combine.png" />

dframe_wine = pd.read_csv('lec44_winequality-red.csv',sep=';')
dframe_wine.head()


def ranker(df):
    df['alc_content_rank'] = np.arange(len(df)) + 1
    return df


dframe_wine.sort_values('alcohol', ascending=False, inplace=True)

dframe_wine = dframe_wine.groupby('quality').apply(ranker)

dframe_wine.head()

num_of_qual = dframe_wine['quality'].value_counts()

num_of_qual

dframe_wine[dframe_wine.alc_content_rank==1].sort_values('quality')


