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

import pandas as pd 
from pandas import Series, DataFrame

titanic_df = pd.read_csv('lec56_train.csv')

titanic_df.head()

titanic_df.info()

# ## データ解析の目標
# このデータから有用な知見を得るために、明確な目標があったほうがよいでしょう。いくつか、具体的な問いを設定してみます。
# 1. タイタニック号の乗客はどのような人たちだったのか？
# 2. それぞれの乗客はデッキにいたか？またそれは客室の種類とどのような関係にあったか？
# 3. 乗客は主にどこからきたのか？
# 4. 家族連れか、単身者か？
#
# これらの基本的な問の後に、さらに深くデータ解析を進めます。
#
# 5.　沈没からの生還者には、どのような要因があったのか？ 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

sns.countplot('Sex', data=titanic_df)

sns.countplot('Sex', data=titanic_df, hue='Pclass')

sns.countplot('Pclass', data=titanic_df, hue='Sex')


# +
def male_female_child(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex
    
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child, axis=1)
# -

titanic_df.head(10)

sns.countplot('Pclass', data=titanic_df, hue='person')

titanic_df['Age'].hist(bins=70)

titanic_df['Age'].mean()

titanic_df['person'].value_counts()

#カーネル密度推定 kdeplot 想定される
fig = sns.FacetGrid(titanic_df, hue='Sex', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df, hue='person', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

titanic_df.head()

deck = titanic_df['Cabin'].dropna()

deck

type(deck)

levels = []
for level in deck:
    levels.append(level[0])

levels

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']

cabin_df

sns.countplot('Cabin', data=cabin_df,palette='winter_d', order=sorted(set(levels)))

cabin_df = cabin_df[cabin_df.Cabin != 'T']

sns.countplot('Cabin', data=cabin_df,palette='summer', order=sorted(set(cabin_df.Cabin)))

titanic_df.head()

sns.countplot('Embarked', data=titanic_df, hue='Pclass')

from collections import Counter

Counter(titanic_df.Embarked)

titanic_df.Embarked.value_counts()

titanic_df['Alone'] = titanic_df.Parch + titanic_df.SibSp
titanic_df['Alone']

titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone']==0] = 'Alone'

titanic_df.head()

sns.countplot('Alone',data=titanic_df, palette='Blues')

titanic_df['Survivor'] = titanic_df.Survived.map({0:'no', 1:'yes'})

sns.countplot('Survivor', data=titanic_df, palette='Set1')

sns.factorplot('Pclass', 'Survived', data=titanic_df, order=[1,2,3])

sns.factorplot('Pclass', 'Survived', hue='person', data=titanic_df, order=[1,2,3],aspect=2)

sns.lmplot('Age', 'Survived', data=titanic_df)

sns.lmplot('Age', 'Survived', hue='Pclass',data=titanic_df, palette='winter', hue_order=[1,2,3])

generations = [10,20,40,60,80]
sns.lmplot('Age', 'Survived', hue='Pclass',data=titanic_df, palette='winter', 
           hue_order=[1,2,3], x_bins=generations)

generations = [10,20,40,60,80]
sns.lmplot('Age', 'Survived', hue='Sex',data=titanic_df, palette='winter', 
            x_bins=generations)

# ## 次の問い
# 1. 乗客が居たデッキは生存率と関係があるか？また、その答えは感覚的な答えと合うだろうか？
# 2. 家族連れであることは、事故からの生存率を上げているだろうか？

titanic_df.head(1)

levels_andId = []
for level in deck:
    levels_andId.append(level.PassengerId,level[0])




