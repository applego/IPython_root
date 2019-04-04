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

# 非常に高い予測精度が得られました。
#
# デフォルトでは、RBFカーネルが使われています。
#
# それぞれのカーネルの違いをscikit-learnのドキュメントに詳しく載っています。
#
# これを自分で作る方法を書いておきますので、興味がある方はやってみてください。

# +
from sklearn import svm

#図示できるのが二次元までなので、変数を２つに絞ります。
X = iris.data[:,:2]
Y = iris.target
# -

#SVMの正則化パラメータ
C = 1.0

#SVC with a Liner Kernel
svc = svm.SVC(kernel='linear', C=C).fit(X,Y)

# Gaussian Radial Bassis Function
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X,Y)

# SVC with 3rd degree polynomial
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X,Y)

# SVC Linear
lin_svc = svm.LinearSVC(C=C).fit(X,Y)

# +
# step size
h = 0.02

# X軸の最大最小
x_min = X[:,0].min() - 1
x_max = X[:,0].max() +1

#　Y軸の最大最小
y_min = X[:,1].min() -1
y_max = X[:,1].max() +1

#meshgridを作ります。
xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
# -

titles = ['SVC with linear kernel',
         'LinearSVC(linear kernel)',
         'SVC with RBF kernel',
         'SVC with polynomial(degree 3) kernel']

# +
for i, clf in enumerate((svc, lin_svc, rbf_svc,poly_svc)):
    
    #境界線を描画します
    plt.figure(figsize=(15,15))
    plt.subplot(2,2,i+1)
    
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx,yy,Z,cmap=plt.cm.terrain,alpha=0.5,linewidths=0)
    
    plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Dark2)
    
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
    
plt.show()
# -


