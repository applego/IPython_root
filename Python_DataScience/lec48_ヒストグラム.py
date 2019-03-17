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
import pandas as pd

#scipyの統計パッケージ
from scipy import stats

#描画用
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#ブラウザ内に埋め込むための命令
# %matplotlib inline
# -

# ヒストグラムの説明は以下を参照してください。 https://ja.wikipedia.org/wiki/%E3%83%92%E3%82%B9%E3%83%88%E3%82%B0%E3%83%A9%E3%83%A0

dataset1 = randn(100)
plt.hist(dataset1)

dataset2 = randn(80)
plt.hist(dataset2,color='indianred', bins=30)

#plt.hist(dataset1, normed=True)
plt.hist(dataset1, density=True)
#面積が１になるように　標準化

plt.hist(dataset1, density=True, alpha=.5, bins=20)
plt.hist(dataset2,density=True, alpha=.5, bins=20, color="indianred")

data1 = randn(1000)
data2 = randn(1000)

sns.jointplot(data1,data2)
#同時分布、結合分布

sns.jointplot(data1,data2, kind='hex')


