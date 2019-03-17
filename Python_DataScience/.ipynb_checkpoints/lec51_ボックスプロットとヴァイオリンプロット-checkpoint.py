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

# +
import numpy as np
from numpy.random import randn
from scipy import stats

import seaborn as sns

# %matplotlib inline

# +
# 英語のページですが・・・
url = 'http://en.wikipedia.org/wiki/Box_plot#mediaviewer/File:Boxplot_vs_PDF.svg'

data1 = randn(100)
data2 = randn(100) + 2 # Off set the mean
# -

sns.distplot(data1)
sns.distplot(data2)

sns.boxplot(data=[data1,data2])

sns.boxplot(data=[data1,data2],whis=np.inf)


