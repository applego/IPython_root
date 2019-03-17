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

# カーネル密度関数に関する説明 https://ja.wikipedia.org/wiki/%E3%82%AB%E3%83%BC%E3%83%8D%E3%83%AB%E5%AF%86%E5%BA%A6%E6%8E%A8%E5%AE%9A
#
# 一言でいうとなめらかなヒストグラムを作る
# データがあるところにガウス推定（正規分布？）を作成→データが多いところは高くなる

# +
import numpy as np
from numpy.random import randn
import pandas as pd 

from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
# -

dataset = randn(25)
dataset

sns.rugplot(dataset)

plt.hist(dataset, alpha=0.3)
sns.rugplot(dataset)

# +
# Bandwidth selection 一個一個の幅を推定？
sns.rugplot(dataset)

x_min = dataset.min() - 2
x_max = dataset.max() +2
x_axis = np.linspace(x_min, x_max, 100)

bandwidth = ((4*dataset.std()**5)/(3*len(dataset)))**0.2

kernel_list = []
for data_point in dataset:
    kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    
    kernel = kernel / kernel.max()
    kernel = kernel * 0.4
    plt.plot(x_axis, kernel, color = 'gray', alpha=0.5)
plt.ylim(0,1)

# +
sum_of_kde = np.sum(kernel_list, axis=0)

fig = plt.plot(x_axis, sum_of_kde, color = 'indianred')
sns.rugplot(dataset)
plt.yticks([])
plt.suptitle('Sum of the Bases Functions')

# +
#seabornを使うと１行
# -

sns.kdeplot(dataset)

sns.rugplot(dataset, color='black')
for bw in np.arange(0.5,2,0.25): #バンド幅によってカーネル密度関数がどうなるか
    sns.kdeplot(dataset, bw=bw, label=bw)

# 正規分布を使ったがその他も使える

kernel_options = ['biw', 'cos', 'epa', 'gau', 'tri', 'triw']#詳しくは公式ドキュメントで
for kern in kernel_options:
    sns.kdeplot(dataset, kernel=kern, label=kern)

sns.kdeplot(dataset, vertical=True)

# Cumulative Distribution function
# 累積分布関数
# ヒストグラムを積み上げる感じ

plt.hist(dataset, cumulative=True)

#累積分布関数に関してもカーネル密度推定を用いてなめらかに
sns.kdeplot(dataset, cumulative=True)

mean=[0,0]
cov=[[1,0],[0,100]]
dataset2 = np.random.multivariate_normal(mean,cov, 1000)#多変量

dframe = pd.DataFrame(dataset2, columns=['X','Y'])
sns.kdeplot(dframe)

sns.kdeplot(dframe.X, dframe.Y)

sns.kdeplot(dframe.X, dframe.Y,shade=True)

sns.kdeplot(dframe, bw=1)

sns.kdeplot(dframe, bw='silverman')

#ジョイントディストリビューション 同時分布？
sns.jointplot('X','Y', dframe, kind='kde')


