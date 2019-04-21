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

# ポアソン分布
# ある間隔の間に起こるエベントの回数に注目するものです。

# +
#lambdaは予約後なので使えない
lamb = 10

#ちょうど７人来る確率を計算したいので
k=7

from math import exp
from math import factorial

#確率質量関数を使って確率を計算します。
prob = (lamb**k)*exp(-lamb)/factorial(k)
print('昼のピーク時にお客さんが７人である確率は、{:0.2f}%です。'.format(100*prob))
# -

# scipyを使うと少し楽になる

# +
from scipy.stats import poisson

mu = 10

mean,var = poisson.stats(mu)

odds_seven = poisson.pmf(7,mu)

print('ピーク時に７人の確率は{:0.2f}%'.format(odds_seven*100))

print('平均={}'.format(mean))

# +
#確率質量関数をプロットしてみよう
import numpy as np

#ひとまず、３０人のお客さんが来る確率。理論的には∞までありえる
k=np.arange(30)

lamb = 10

pmf_pois = poisson.pmf(k,lamb)
pmf_pois

# +
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

plt.bar(k,pmf_pois)

# +
k,mu = 10,10

prob_up_to_ten = poisson.cdf(k,mu)

print('お客さんが１０人までの確率は、{:0.2f}%'.format(100*prob_up_to_ten))

# +
prob_more_than_ten = 1 - prob_up_to_ten

print('１０人より多くのお客さんが来る確率は、{:0.2f}%'.format(100*prob_more_than_ten))
# -


