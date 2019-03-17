# -*- coding: utf-8 -*-
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

# サンプルデータは、次のURLからダウンロードできます。
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/'
# !cat lec44_winequality-red.csv

dframe_wine = pd.read_csv('lec44_winequality-red.csv', sep=';')
#1行が１本のワインについて（quality:専門家の評価

dframe_wine.head()

dframe_wine['alcohol'].mean()


def max_to_min(arr):
    return arr.max() - arr.min()


wino = dframe_wine.groupby('quality')

wino.describe()

wino.agg(max_to_min)

wino.agg('mean')

dframe_wine['qual/alc ratio'] = dframe_wine['quality'] / dframe_wine['alcohol']

dframe_wine

dframe_wine.pivot_table(index=['quality'])

# %matplotlib inline

dframe_wine.plot(kind='scatter', x='quality', y='alcohol')


