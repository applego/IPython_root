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

# +
import numpy as np
import pandas as pd 
from pandas import Series,DataFrame

import math

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline
# -

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn import metrics

import statsmodels.api as sm


#シグモイド関数、ロジスティック関数
def logistic(t):
    return 1.0/(1+math.exp(-1.0*t))


t = np.linspace(-6,6,500)

t

y = np.array([logistic(ele) for ele in t])

plt.plot(t,y)
plt.title('Logistic Function')

#不倫するか（自己申告
df = sm.datasets.fair.load_pandas().data

df


#不倫の有無を01で
def affair_check(x):
    if x != 0:
        return 1
    else:
        return 0


df['Had_Affair'] = df['affairs'].apply(affair_check)

df

df.groupby('Had_Affair').mean()

sns.countplot('age',data=df.sort_values('age'), hue='Had_Affair',palette='coolwarm')

sns.countplot('yrs_married',data=df.sort_values('yrs_married'),hue='Had_Affair',palette='coolwarm')

sns.countplot('children',data=df.sort_values('children'),hue='Had_Affair',palette='coolwarm')

sns.countplot('educ',data=df.sort_values('educ'),hue='Had_Affair',palette='coolwarm')

sns.scatterplot('yrs_married','children',data=df)

#ロジスティック回帰のためのデータの前処理
#カテゴリー型のデータ→数値の大小に意味がない→ダミー変数を使用（pandasにget_dummies
occ_dummies = pd.get_dummies(df.occupation)

hus_occ_dummies = pd.get_dummies(df.occupation_husb)

occ_dummies.head()

occ_dummies.columns = ['occ1','occ2','occ3','occ4','occ5','occ6']
hus_occ_dummies.columns = ['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']

X = df.drop(['occupation','occupation_husb','Had_Affair'],axis=1)

dummies = pd.concat([occ_dummies,hus_occ_dummies],axis=1)

X = pd.concat([X,dummies],axis=1)

X.head()

Y = df.Had_Affair

Y.tail()

X = X.drop('occ1',axis=1)

X = X.drop('hocc1',axis=1)

X = X.drop('affairs', axis=1)

X.head()

Y

Y.values

np.ravel(Y)

Y = np.ravel(Y)

#sklearnを使ってロジスティック回帰
log_model = LogisticRegression()

log_model.fit(X,Y)

log_model.score(X,Y) #精度　XでYをどのくらいの精度で予測できるか　モデル作成に使ったデータを使っても以下の数字

coeff_df = DataFrame([X.columns, log_model.coef_[0]]).T

coeff_df

X_train, X_test ,Y_train ,Y_test = train_test_split(X,Y)

log_model2 = LogisticRegression()

log_model2.fit(X_train,Y_train)

class_predict = log_model2.predict(X_test)

class_predict

metrics.accuracy_score(Y_test, class_predict)


