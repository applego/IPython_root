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

# # Part 1: SVMの原理
# まずは、SVMのおおまかな原理を掴んでいきましょう。

from IPython.display import Image
url = 'https://upload.wikimedia.org/wikipedia/commons/2/20/Svm_separating_hyperplanes.png'
Image(url, width=450)

# # Part 2: カーネル法
# いつも超平面で分離できるとは限りません。そんな時、役に立つのがカーネル法と呼ばれる、工夫です。

# 特徴量空間におけるカーネルトリック
url='http://i.imgur.com/WuxyO.png'
Image(url)

# カーネル法がよく分かる動画です。
from IPython.display import YouTubeVideo
YouTubeVideo('3liCbRZPrZA')

# # Part 3: その他の資料
# 英語になってしまいますが、その他の資料を挙げておきます。

# MITの講義
YouTubeVideo('_PwhiWxHK8o')

# # Part 4: scikit-learnを使ったSVMの実際

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data

Y = iris.target

print(iris.DESCR)

from sklearn.svm import SVC

model = SVC()

from sklearn.model_selection import train_test_split

X_train , X_test, Y_train, Y_test = train_test_split(X,Y,random_state =0)

model.fit(X_train,Y_train)

predicted = model.predict(X_test)

predicted

from sklearn import metrics

expected = Y_test

print(metrics.accuracy_score(expected,predicted))

# +
#kernel SVM　カーネルを調整する
