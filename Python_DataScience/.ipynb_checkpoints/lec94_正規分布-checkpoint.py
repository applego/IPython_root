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

from IPython.display import Image
Image(url='http://upload.wikimedia.org/wikipedia/commons/thumb/2/25/The_Normal_Distribution.svg/725px-The_Normal_Distribution.svg.png')

# 分布の形を見て、正規分布の主な特徴を確認していきましょう。
#
# 1.) 左右に裾野を持ちます。
# 2.) 曲線は左右対称です。
# 3.) ピークは平均の値です。
# 4.) 標準偏差が曲線の形を特徴付けます。
#     - 背が高い分布は、小さな標準偏差のときです。
#     - 太った分布は、大きな標準偏差のときです。
# 5.) 曲線のしたの面積（AUC: area under the curve）は1です。
# 6.) 平均値、中央値、最頻値（mode）がすべて同じです。
# 平均が0、標準偏差が1の標準正規分布では、±1標準偏差に68%、±2標準偏差に95%が含まれ、±3標準偏差までには、全体の99.7%が含まれます。この1,2,3といった数字をz-scoreと呼ぶこともあります。

# +
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

# statsライブラリをImportします。
from scipy import stats

# 平均を0
mean = 0

# 標準偏差を1にしてみましょう。
std = 1

# 便宜的に領域を決めます。
X = np.arange(-4,4,0.01)

# 値を計算します。
Y = stats.norm.pdf(X,mean,std)

plt.plot(X,Y)

# +
import numpy as np

mu,sigma = 0,0.1

#正規分布に従う乱数を1000個生成します。
norm_set = np.random.normal(mu,sigma,1000)

# +
#seabornを使ってプロットしてみましょう。
import seaborn as sns

plt.hist(norm_set,bins=50)
# -


