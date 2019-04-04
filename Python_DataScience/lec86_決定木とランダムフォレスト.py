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

# https://towardsdatascience.com/enchanted-random-forest-b08d418cb411
# 決定木

# ランダムフォレストはアンサンブル学習法の一つです。アンサンブル学習法はいくつかの分類機を集めて構成されるものですが、ここでは決定木が使われます。木が集まるから森というわけです。
# ランダムフォレストの方が新しく、応用範囲が広い

# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

# ダミーデータを作成する
from sklearn.datasets import make_blobs

# centers中心店の数、　cluster_stdデータのばらつき具合の指定
X, y = make_blobs(n_samples=500, centers=4, random_state=8, cluster_std=2.4)

X

y

plt.figure(figsize=(10,10))
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='jet')

from sklearn.tree import DecisionTreeClassifier


# 決定木を描画する
# SplitしてSplitしてという図を描く
def visualize_tree(classifier, X, y, boundaries=True,xlim=None, ylim=None):
    '''
    決定木の可視化をします。
    INPUTS: 分類モデル, X, y, optional x/y limits.
    OUTPUTS: Meshgridを使った決定木の可視化
    '''
    # fitを使ったモデルの構築
    classifier.fit(X, y)
    
    # 軸を自動調整
    if xlim is None:
        xlim = (X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    if ylim is None:
        ylim = (X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)

    x_min, x_max = xlim
    y_min, y_max = ylim
    
    
    # meshgridをつくります。
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 分類器の予測をZとして保存
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # meshgridを使って、整形します。
    Z = Z.reshape(xx.shape)
    
    # 分類ごとに色を付けます。
    plt.figure(figsize=(10,10))
    plt.pcolormesh(xx, yy, Z, alpha=0.2, cmap='jet')
    
    # 訓練データも描画します。
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='jet')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)        
    
    def plot_boundaries(i, xlim, ylim):
        '''
        境界線を描き込みます。
        '''
        if i < 0:
            return

        tree = classifier.tree_
        
        # 境界を描画するために、再帰的に呼び出します。
        if tree.feature[i] == 0:
            plt.plot([tree.threshold[i], tree.threshold[i]], ylim, '-k')
            plot_boundaries(tree.children_left[i],
                            [xlim[0], tree.threshold[i]], ylim)
            plot_boundaries(tree.children_right[i],
                            [tree.threshold[i], xlim[1]], ylim)
        
        elif tree.feature[i] == 1:
            plt.plot(xlim, [tree.threshold[i], tree.threshold[i]], '-k')
            plot_boundaries(tree.children_left[i], xlim,
                            [ylim[0], tree.threshold[i]])
            plot_boundaries(tree.children_right[i], xlim,
                            [tree.threshold[i], ylim[1]])
    
    if boundaries:
        plot_boundaries(0, plt.xlim(), plt.ylim())

clf = DecisionTreeClassifier(max_depth=2, random_state=0)

visualize_tree(clf,X,y)

#深さを変えてみる
clf4 = DecisionTreeClassifier(max_depth=4,random_state=0)

visualize_tree(clf4,X,y)

#　過学習が起こる→防ぐため(完全に防ぐわけではない)にランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

clfr = RandomForestClassifier(n_estimators=100, random_state=0)

visualize_tree(clfr,X,y,boundaries=False)

# ランダムフォレストは応用範囲が広い
# 分類ではなく回帰に使ってみる
from sklearn.ensemble import RandomForestRegressor

# +
x = 10 * np.random.rand(100)

def sin_model(x, sigma=0.2):
    '''
    大きな波＋小さな波＋ノイズからなるダミーデータです。
    '''
    noise = sigma * np.random.randn(len(x))
    
    return np.sin(5 * x) + np.sin(0.5 * x) + noise


# -

#xからyを計算
y = sin_model(x)

y

#plotします
plt.figure(figsize=(16,8))
plt.errorbar(x,y,0.1,fmt='o')

#回帰
xfit = np.linspace(0,10,1000)

rfr = RandomForestRegressor(100)

rfr.fit(x[:,None],y)

yfit = rfr.predict(xfit[:,None])

yfit

ytrue = sin_model(xfit,0)

# +
plt.figure(figsize=(16,8))

plt.errorbar(x,y,0.1,fmt='o')
plt.plot(xfit,yfit,'-r')
plt.plot(xfit,ytrue,'-k', alpha=0.5)
