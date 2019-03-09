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

# # 探索的データ解析チュートリアル

# +
# 必要ライブラリのインポート
import pandas as pd

# pandasの列最大表示数を指定
pd.options.display.max_columns = 32

# matplotlib
import matplotlib.pyplot as plt

# numpy
import numpy as np

# %matplotlib inline
# -

# CSVファイルをPandasのDataFrame形式で読み込み
Iris = pd.read_csv('Iris.csv')

# ## 基本的なデータ探索

Iris.head()

Iris.tail()

Iris.describe()

# <img src="petal_sepal_label.png" height="80" width="120" />

# アヤメの種類ごとの特徴量のデータ分散を確認してみる
Irisdes = Iris.groupby(['Species'])
round(Irisdes.describe(),2)

# info()を使ってIrisデータフレームをみてみる
Iris.info()

# ## カラムの値を確認する

# 特定のカラムのユニークな値を出力する
Iris.Species.unique()

# value_counts()でSpeciesのカラムの情報をみてみよう
Iris['Species'].value_counts()

# ## カラムをデータフレームから削除する

# データフレームからIdのカラムを削除する
Iris = Iris.drop(['Id'],axis=1)
Iris.head()

# ## データの可視化

# ### ヒストグラム

# Irisのデータフレームのヒストグラムを作成
Iris.hist(bins=10, figsize=(20,15), color='teal')

# ## 特徴量ごとに重ねたヒストグラムを作って確認

# +
# ヒストグラムを作成
fig = plt.figure(figsize=(25,10))

p1 = fig.add_subplot(2,2,1)
p1.hist(Iris.PetalLengthCm[Iris.Species == 'Iris-setosa'], bins=10, alpha=.4)
p1.hist(Iris.PetalLengthCm[Iris.Species == 'Iris-versicolor'], bins=10, alpha=.4)
p1.hist(Iris.PetalLengthCm[Iris.Species == 'Iris-virginica'], bins=10, alpha=.4)
plt.title('Petal Length Cm')
plt.xlabel('Cm Measurement')
plt.ylabel('Count')
labels = ["Iris-setosa","Iris-versicolor","Iris-virgiica"]
plt.legend(labels)

p2 = fig.add_subplot(2,2,2)
p2.hist(Iris.PetalWidthCm[Iris.Species== 'Iris-setosa'], bins=10, alpha=.4)
p2.hist(Iris.PetalWidthCm[Iris.Species== 'Iris-versicolor'], bins=10, alpha=.4)
p2.hist(Iris.PetalWidthCm[Iris.Species== 'Iris-virginica'], bins=10, alpha=.4)
plt.title('Petal Width Cm')
plt.xlabel('Cm Measurement')
plt.ylabel('Count')
labels = ["Iris-setosa","Iris-versicolor","Iris-virgiica"]
plt.legend(labels)

p3 = fig.add_subplot(2,2,3)
p3.hist(Iris.SepalLengthCm[Iris.Species=='Iris-setosa'], bins=10, alpha=.4)
p3.hist(Iris.SepalLengthCm[Iris.Species=='Iris-versicolor'], bins=10, alpha=.4)
p3.hist(Iris.SepalLengthCm[Iris.Species=='Iris-virginica'], bins=10, alpha=.4)
plt.title('Sepal Length Cm')
plt.xlabel('Cm Measurement')
plt.ylabel('Count')
labels = ["Iris-setosa","Iris-versicolor","Iris-virgiica"]
plt.legend(labels)

p4 = fig.add_subplot(2,2,4)
p4.hist(Iris.SepalWidthCm[Iris.Species=='Iris-setosa'], bins=10, alpha=.4)
p4.hist(Iris.SepalWidthCm[Iris.Species=='Iris-versicolor'], bins=10, alpha=.4)
p4.hist(Iris.SepalWidthCm[Iris.Species=='Iris-virginica'], bins=10, alpha=.4)
plt.title('Sepal Width Cm')
plt.xlabel('Cm Measurement')
plt.ylabel('Count')
labels = ["Iris-setosa","Iris-versicolor","Iris-virgiica"]
plt.legend(labels)

plt.subplots_adjust(wspace=.1, hspace=.3)
plt.show()
# -

# ## 散布図（スキャタープロット）

# ヒストグラムと散布図の行列を作成
from pandas.plotting import scatter_matrix
x = scatter_matrix(Iris, alpha=1, figsize=(20,10), diagonal='hist')
# たった一行のコードで、このように各特徴量も散布図のマトリックス（行列）が作成できてしまいます。

# ## データフレームのフィルタリング

# データフレームの行インデックスが6-20までを表示
Iris[6:20]

# 行インデックス6-20の間で3等間隔にフィルター
Iris[6:20:3]

# カラムに値を指定してフィルタリング
Iris[Iris.Species=='Iris-setosa'][0:10]

# ２つのカラム&条件を指定してフィルタリング
Iris[(Iris.Species=='Iris-setosa') & (Iris.SepalLengthCm > 5.5)]

#  カラムをSpeciesとSpealLengthCmのみ表示する
Iris[['Species','SepalLengthCm']][0:10]

# ## データフレームの順番の並び替え（ソート）

#　並べ替え
Iris.sort_values('SepalLengthCm', axis=0, ascending=False)[0:10]

# ## 外れ値の確認

Iris.isnull().any()

# まとめ
# 如何でしたでしょうか？今回は、データサイエンティスト入門として探索的データ解析（EDA）の初歩的な内容をまとめました。
#
# 次のステップとして、次はより実践的なデータを触りながら機械学習入門をしてみては如何でしょうか？1時間〜3時間程度で行える初心者向けチュートリアルを公開していますので、是非挑戦してみてください。
#
# 初心者向けの機械学習入門チュートリアル
#
# 【Kaggle初心者入門編】タイタニック号で生き残るのは誰？
# Amazon SageMakerを使って銀行定期預金の見込み顧客を予測【SageMaker ＋XGBoost 機械学習初心者チュートリアル】
#
# 機械学習をすでに触ったことがある方はこちらもオススメ
#
# 初心者のための畳み込みニューラルネットワーク（MNISTデータセット + Kerasを使ってCNNを構築）
#
# 以上となります！最後までお付き合いくださいして、ありがとうございます。
