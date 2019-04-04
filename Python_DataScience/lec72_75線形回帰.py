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

# # 線形回帰（教師有り学習）
# ここから始まるレクチャーでは、線形回帰について学びます。 scikit-learnを使って、線形回帰のモデルを作り、新しいデータを使った予測を試みます。 サンプルデータは、アメリカの大都市ボストンの住宅価格です。初めは、1つだけの変数を使った単回帰をやってみます。
#
# 線形回帰の数学的な背景に興味がある場合は、以下のサイトが参考になります。
#
# wikipedia（日本語）
# wikipedia（英語）
# Andrew Ngの動画（英語）もあります youtube.
# 4回に分かれているレクチャーの概要です。
#
# Step 1: データの準備
# Step 2: ひとまず可視化
#
# Step 3: 最小二乗法の数学的な背景
#
# Step 4: Numpyを使った単回帰
#
# Step 5: 誤差について
#
# Step 6: scikit-learnを使った重回帰分析
#
# Step 7: 学習（Training）と検証Validation）
#
# Step 8: 価格の予測
#
# Step 9 : 残差プロット
#
# ## Step 1: データの準備
# scikit-learnに用意されているサンプルデータを使います。

# +
import numpy as np
import pandas as pd 
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline
# -

from sklearn.datasets import load_boston

boston = load_boston()

print(boston.DESCR) #Description

plt.hist(boston.target, bins=50)
plt.xlabel('Price($1,000)')
plt.ylabel('Number of houses')

plt.scatter(boston.data[:,5], boston.target) #RMと価格
plt.xlabel('Price($1,000)')
plt.ylabel('Number of rooms')

boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names

boston_df.head()

boston_df['Price'] = boston.target

boston_df.head()

sns.lmplot('RM', 'Price', data=boston_df)

# # 線形回帰その２
#
# ## Step 3: 最小二乗法の数学的な背景
# 回帰直線の係数を求めるのに使われる、「最小二乗法」について、すこし数学的になりますが、その背景を説明します。
#
# 回帰直線は、データ全体にうまく適合するように、描かれています。各点から、回帰直線への距離をDとしてみましょう。このDを最小にすれば良いわけです。このイメージを図にしてみます。

# wikipediaから拝借します。
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Linear_least_squares_example2.svg/220px-Linear_least_squares_example2.svg.png'
Image(url)

# 各点（赤）の座標は、(x, y)です。ここから、回帰直線（青線）への距離をDとすると、以下の値を最小にする直線が一番よさそうです。
#
# d=D21+D22+D23+D24+....+D2N
#  
# 直線の式は、
#
# y=ax+b
#  
# で表現されます。いま、 a と b を求めたいのですが、これはdを最小にする a と b を見つけ出すという問題と同じです。
#
# この問題はもちろん、手で計算することで解くことができますが、ここではこの計算をNumpyやscikit-leranにお任せします。 もし数学的な計算方法に興味がある方は、こちらが大変参考になります。
#
# ## Step 4: Numpyを使った単回帰
# Numpyは線形代数のライブラリの一部に、最小二乗法を解く関数を持っています。 まずはこれを使って、単回帰(説明変数が1つ）をやってみます。その後、scikit-learnを使って、重回帰（説明変数が複数）に進んで行きましょう。
#
# 入力として、2つのarray（XとY）を用意します。
#
# Yは目的変数なので1次元のarrayですが、Xは2次元のarrayで、行がサンプル、列が説明変数です。単回帰の場合は、列が1つになりますですので、そのshapeは、(506,1)です。これを作るには、いくつか方法がありますが、ここでは、vstackを使ってみます。

# 部屋数
X = boston_df.RM
print(X.shape)

X

# これを2次元のarrayにします。
X = np.vstack(boston_df.RM)
print(X.shape)

X

Y = boston_df.Price
print(Y.shape)

type(Y)

#変形
#リスト内包記
X = np.array([ [value, 1] for value in X])

X.dtype

type(X)

X = X.astype(np.float64)
X.dtype

Y = Y.astype(np.float64)
Y.dtype

#最小二乗法の計算を実行
np.linalg.lstsq(X,Y)

#最小二乗法の計算を実行
a,b = np.linalg.lstsq(X,Y)[0]

# +
plt.plot(boston_df.RM, boston_df.Price, 'o')

x = boston_df.RM
plt.plot(x, a*x+b, 'r')
# -

#
# ## Step 5: 誤差について
# Pythonを使って、最小二乗法を用いて、単回帰を実行出来ました。 すべてのデータが完全に乗る直線を描くことは出来ませんので、どうしても誤差が出ます。
#
# 最小化しているのは、誤差の2乗和でした。ですので、全体の誤差が分かれば、それをサンプルの数で割って、平方根をとることで、ちょうど標準偏差のようなイメージで、平均誤差を計算できます。
#
# numpy.linalg.lstsqのドキュメント（英語）

result = np.linalg.lstsq(X,Y)

result

error_total = result[1]
rmse = np.sqrt(error_total/len(x))  #root mean squere error

print('平均二乗誤差の平方根={:0.2f}'.format(rmse[0]))

#
# 平均二乗誤差は、標準偏差に対応するので、95%の確率で、この値の2倍以上誤差が広がることは無いと結論付けあれます。 正規分布の性質を思い出したい方は、こちらを参照.
#
# Thus we can reasonably expect a house price to be within $13,200 of our line fit.
#
# ## Step 6: scikit-learnを使った重回帰分析
# それでは、重回帰へと話を進めましょう。 説明変数が1つだけだと単回帰ですが、重回帰は複数の説明変数を同時に扱うことができます。
#
# scikit-learnの線形回帰ライブラリを利用します。 linear regression library
#
# sklearn.linear_model.LinearRegressionクラスは、データを元にモデルを作り、予測値を返すことができます。 モデルを作る時には、fit()メソッドを呼び、予測をするときは、predict()メソッドを使います。 今回は重回帰モデルを使いますが、他のモデルも同じように、fitとpredictメソッドを実装しているところが、scikit-learnの便利なところです。

import sklearn
from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

#説明変数の用意
X_multi = boston_df.drop('Price',1)

X_multi.head()

X_multi.shape

#重回帰は説明変数が複数
Y_target = boston_df.Price

lreg.fit(X_multi, Y_target)

lreg.intercept_ #y=ax+bのb

len(lreg.coef_) #係数の数

# 単回帰の時は、直線だったので、係数aと切片bはともに1つでした。今は、切片は1つですが、係数が13個あります。これは13個変数がある式になっている事を意味しています。
#
# y=b+a1x1+a2x2+⋯+a13x13
#  
# 実際に求められた係数を見ていきましょう。

coeff_df = DataFrame(boston_df.columns)

coeff_df.columns = ['Features']

coeff_df['Coefficient Estimate'] = pd.Series(lreg.coef_)

coeff_df

# ## Step 7: 学習（Training）と検証（Validation）
# ここまではすべてのデータを使って来ましたが、一部のデータを使って、モデルを作り、残りのデータを使って、モデルを検証するということができます。
#
# サンプルをどのように分けるかが問題ですが、scikit-learnに便利な関数 train_test_split があるので、使って見ましょう。
#
# サンプルを学習用のtrainと検証用のtestに分けてくれます。追加のパラメータを渡せば、割合も調整できます。 詳しくはこちら

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_multi, boston_df.Price)

Y_train.head()

Y_test.head()

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

lreg = LinearRegression()

lreg.fit(X_train, Y_train)

pred_train = lreg.predict(X_train)
pred_train

pred_test = lreg.predict(X_test)

np.mean((Y_train - pred_train)**2)

np.mean((Y_test - pred_test)**2)

# +
train = plt.scatter(pred_train, (pred_train - Y_train), c='b', alpha=.5)
test = plt.scatter(pred_test, (pred_test-Y_test),c='r',alpha=.5)
plt.hlines(y=0,xmin=1.0,xmax=50)

plt.legend((train,test),('Training','Test'), loc='lower left')
plt.title('Residual plots')
# -


