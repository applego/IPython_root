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

from numpy.random import randn
import seaborn as sns
# %matplotlib inline

dataset = randn(100)

sns.distplot(dataset,rug=True) #distribution? 分布
# ヒストグラム＆カーネル密度関数

sns.distplot(dataset, rug=True, hist=False)

sns.distplot(dataset, bins=25,
            kde_kws={'color':'indianred', 'label':'KDE PLOT'},
            hist_kws={'color':'blue', 'label':'HISTGRAM'})

from pandas import Series

ser1 = Series(dataset, name='My_DATA')

ser1


