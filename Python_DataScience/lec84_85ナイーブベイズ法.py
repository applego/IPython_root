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

# # ナイーブベイズ分類¶
#
# - Part 1: 数学的な準備
# - Part 2: ベイズの定理
# - Part 3: ナイーブベイズの紹介
# - Part 4: 数学的な背景
# - Part 5: 確率を使った分類
# - Part 6: Gaussian Naive Bayes
# - Part 7: Scikit Learnを使ったGaussian Naive Bayes
# - Part 1: 数学的な準備
#
# まずは足し算で使われるΣの、かけ算バージョン、Πについてです。
#
# ∏i=14i=1⋅2⋅3⋅4,
#  
# これは連続的なかけ算を意味するので、
# ∏i=14i=24.
#  
# Arg Max
#
# 与えられた関数を最大にする入力（定義域）を次のような記号で書くことがあります。
#
# argmaxxf(x):={x∣∀y:f(y)≤f(x)}
#  
# 例えば、f(x)が 1−|x| なら、この関数を最大にするxは0になります。
#
# argmaxx(1−|x|)={0}
#  
# - Part 2: ベイズの定理
# 統計に関して解説した付録のなかに、ベイズの定理を紹介したレクチャーがありますので、そちらを先にご覧ください。
#
# - Part 3: ナイーブベイズの紹介
# ナイーブベイズ（Naive Bayes）は、スパムメールの分類などに実際に利用されている機械学習アルゴリズムです。ただ、その背景を完全に理解するには、数学的な記述が欠かせません。ここでは、細かいところを省略しながら、その本質をご紹介します。
#
# - Part 4: 数学的な背景
# yが目的変数、説明変数が x1 から xn まであるとします。ベイズの定理を使うと、与えられた説明変数を元に、そのサンプルがどのクラスに属するかの確率を次のような式で計算できます。
#
# P(y∣x1,…,xn)=P(y)P(x1,…xn∣y)P(x1,…,xn)
#  
# **ナイーブベイズのナイーブは、各説明変数が互いに独立**であるという仮定から来ています。
# P(xi|y,x1,…,xi−1,xi+1,…,xn)=P(xi|y)
#  
# この仮定のもとでは、すべての i について、式を次のように変形できます。
#
# P(y∣x1,…,xn)=P(y)∏ni=1P(xi∣y)P(x1,…,xn)
#  
# それぞれの変数について、クラスごとの確率を求めればよいので、計算が楽になります。
#
# - Part 5: 確率を使った分類
# ナイーブベイズでは、それぞれのクラスに属する確率が計算されるので、最終的には、そのサンプルを、確率が最も大きいクラスに分類します。ここで、arg maxの記号が出てくる分けです。
#
# P(x1, ..., xn) は手元のデータセットに関しては一定の値なので、無視できます。
#
# P(y∣x1,…,xn)∝P(y)∏i=1nP(xi∣y)
#  
# 最終的には、もっとも大きな確率が割り当たるクラスに、サンプルを分類します。
#
# ŷ =argmaxyP(y)∏i=1nP(xi∣y),
#  
# - Part 6: Gaussian Naive Bayes
# 説明変数が連続値の場合、これを正規分布に従うものとしてモデル化すると、モデルの構築や計算が楽にです。サンプルデータのアヤメのデータも連続値ですので、後ほどの、Gaussian Naive Bayesを利用します。
#
# p(x=v|c)=12πσ2c‾‾‾‾‾√e−(v−μc)22σ2c
#  
# - Part 7: Scikit learnを使ったGaussian Naive Bayes

# +
import pandas as pd 
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# -

iris = datasets.load_iris()

X = iris.data
Y = iris.target

model = GaussianNB()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)

model.fit(X_train,Y_train)

predicted = model.predict(X_test)

predicted

metrics.accuracy_score(Y_test, predicted)

# ナイーブベイズ　
# 文章の分類や、スパムメールの検出


