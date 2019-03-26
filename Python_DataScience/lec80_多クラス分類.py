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

# 多クラス分類（Multi-Class Classification）
# ロジスティック回帰では、データを2つのクラスに分類する方法を学びました。しかし、実社会ではサンプルが3つ以上のクラスに分けられる問題も多くあります。
#
# ここからのレクチャーでは、こうした問題に対応出来る、多クラス分類の方法を学びます。
#
# 1.) Iris（アヤメ）データの紹介
# 2.) ロジスティック回帰を使った多クラス分類の紹介
# 3.) データの準備
# 4.) データの可視化
# 5.) scikit-learnを使った多クラス分類
# 6.) K近傍法（K Nearest Neighbors）の紹介
# 7.) scikit-learnを使ったK近傍法
# 8.) まとめ
# Step 1: Iris（アヤメ）のデータ
# 機械学習のサンプルデータとして非常によく使われるデータセットがあります。 それが、Iris（アヤメ）のデータ です。
#
# このデータセットは、イギリスの統計学者ロナルド・フィッシャーによって、1936年に紹介されました。
#
# 3種類のアヤメについて、それぞれ50サンプルのデータがあります。それぞれ、Iris setosa、Iris virginica、Iris versicolorという名前がついています。全部で150のデータになっています。4つの特徴量が計測されていて、これが説明変数になります。4つのデータは、花びら（petals）と萼片（sepals）の長さと幅です。
#
# 花びら（petals）と萼片（sepals）

# Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

# データの概要をまとめておきましょう。
#
# 3つのクラスがあります。
#
# Iris-setosa (n=50)
# Iris-versicolor (n=50)
# Iris-virginica (n=50)
#
# 説明変数は4つです。
#
# 萼片（sepal）の長さ（cm）
# 萼片（sepal）の幅（cm）
# 花びら（petal）の長さ（cm）
# 花びら（petal）の幅（cm）
# Step 2: 多クラス分類の紹介
# 最も基本的な多クラス分類の考え方は、「1対その他（one vs all, one vs rest）」というものです。 複数のクラスを、「注目するクラス」と「その他のすべて」に分けて、この2クラスについて、ロジスティック回帰の手法を使います。
#
# どのクラスに分類されるかは、回帰の結果もっとも大きな値が割り振られたクラスなります。
#
# 後半では、K近傍法という別の方法を紹介します。

# 英語になりますが、Andrew Ng先生の動画は、イメージを掴むのによいかもしれません。
from IPython.display import YouTubeVideo
YouTubeVideo("Zj403m-fjqg")

# +
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# %matplotlib inline
# -

from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data

Y = iris.target

print(iris.DESCR)

X

Y

iris_data = DataFrame(X, columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])

iris_target = DataFrame(Y, columns=['Species'])


def flower(num):
    if num == 0:
        return 'Setosa'
    if num == 1:
        return 'Versicolour'
    else:
        return 'Virginica'


iris_target['Species'] = iris_target['Species'].apply(flower)

iris_target

iris = pd.concat([iris_data,iris_target],axis=1)

iris

sns.pairplot(iris, hue='Species', size=2)

plt.figure(figsize=(12,4))
sns.countplot('Petal Length', data=iris,hue='Species')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logreg = LogisticRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=3)

logreg.fit(X_train,Y_train)

from sklearn import metrics

#Yの予測値
Y_pred = logreg.predict(X_test)

Y_pred

metrics.accuracy_score(Y_test,Y_pred)#テストデータの正解, 予測値

# ## k近傍法

# 93%と高い精度が得られました。random_stateを指定すれば、再現性がある結果を得ることができます。
#
# 次に、K近傍法に進んで行きましょう。
#
# Step 6: K近傍法
# K近傍法は英語で、k-nearest neighborなので、kNNと略されることもありますが、極めてシンプルな方法論です。
#
# 学習のプロセスは、単純に学習データを保持するだけです。新しいサンプルが、どちらのクラスに属するかを予測するときにだけ、すこし計算をします。
#
# 与えられたサンプルのk個の隣接する学習データのクラスを使って、このサンプルのクラスを予測します。 イメージをうまく説明した図がこちら。

Image('http://bdewilde.github.io/assets/images/2012-10-26-knn-concept.png',width=400, height=300)

# +
#K近傍法
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)
# -

Y_pred_k = knn.predict(X_test)

metrics.accuracy_score(Y_test, Y_pred_k) #さっきより少し上がった

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, Y_train)

Y_pred_k1 = knn1.predict(X_test)

metrics.accuracy_score(Y_test, Y_pred_k1)

k_range = range(1,90)
accuracy = []

for k in k_range:
    knn_ = KNeighborsClassifier(n_neighbors=k)
    knn_.fit(X_train, Y_train)
    Y_pred_ = knn_.predict(X_test)
    accuracy.append(metrics.accuracy_score(Y_test, Y_pred_))

plt.plot(k_range, accuracy)
plt.xlabel('K for kNN')
plt.ylabel('Testing Accuracy')

max(accuracy)

# Step 8: まとめ
# ロジスティック回帰とk近傍法を使った多クラス分類について学びました。
#
# 英語になりますが、参考資料をいくつかあげておきます。
#
# 1.) Wikipedia on Multiclass Classification
#
# 2.) MIT Lecture Slides on MultiClass Classification
#
# 3.) Sci Kit Learn Documentation
#
# 4.) DataRobot on Classification Techniques


