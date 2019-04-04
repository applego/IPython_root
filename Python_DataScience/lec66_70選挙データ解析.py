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

# 選挙のデータ解析（世論調査と寄付）
# このレクチャーでは、2012年のアメリカ大統領選挙について扱います。その内容にあまり詳しくない方は、以下が参考になると思います。 https://ja.wikipedia.org/wiki/2012%E5%B9%B4%E3%82%A2%E3%83%A1%E3%83%AA%E3%82%AB%E5%90%88%E8%A1%86%E5%9B%BD%E5%A4%A7%E7%B5%B1%E9%A0%98%E9%81%B8%E6%8C%99
#
# 基本的には民主党のオバマ候補と、共和党のロムニー候補の争いで、オバマ候補が勝利しました。
#
# 最初は、世論調査結果のデータを扱います。以下のような問題を設定してみましょう。
#
# 1.) どのような人達が調査対象だったか？
# 2.) 調査結果は、どちらの候補の有利を示しているか？
# 3.) 態度未定の人達が世論調査に与えた影響は？
# 4.) また、態度未定の人たちの動向は？
# 5.) 投票者の気持ちは、時間ともにどう変化したか？
# 6.) 討論会の影響を世論調査の結果から読み取ることができるか？
#
# 2つ目のデータセットについては、後半で。

# +
import pandas as pd 
from pandas import Series, DataFrame
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline
# -

import requests
from io import StringIO

# +
# データのURLです。
url = "http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv"

# requestsをつかってデータをtextとして取得します。
source = requests.get(url).text

# StringIOを使ってpandasのエラーを防ぎます。
poll_data = StringIO(source) 
# -

poll_data

poll_df = pd.read_csv(poll_data)

poll_df.info()

poll_df.head()

poll_df.tail()

# ちょっと分かりにくいので、世論調査の主体とその支持政党をまとめて見ます。
poll_df[['Pollster','Partisan','Affiliation']].sort_values('Pollster').drop_duplicates()

sns.countplot('Affiliation', data=poll_df)

sns.countplot('Affiliation', data=poll_df, hue='Population')

sns.countplot('Affiliation', data=poll_df, hue='Population', order=['Rep', 'Dem','None'])

avg = pd.DataFrame(poll_df.mean())

avg

avg.drop('Number of Observations', axis=0, inplace=True)

avg

std = pd.DataFrame(poll_df.std())
std.drop('Number of Observations', axis=0, inplace=True)

std

avg.plot(yerr=std, kind='bar', legend=False)

poll_avg = pd.concat([avg,std], axis=1)
poll_avg.columns = ['Average', 'STD']

poll_avg

poll_df.head()

poll_df.plot(x='End Date', y=['Obama','Romney','Undecided'], marker='o',linestyle='')

from datetime import datetime

poll_df['Difference'] = (poll_df.Obama - poll_df.Romney)/100

poll_df

poll_df = poll_df.groupby(['Start Date'],as_index=False).mean()

poll_df.head()

fig = poll_df.plot('Start Date', 'Difference', figsize=(12,4), marker='o',linestyle='-',color='purple')

#10/3, 10/11, 10/22
poll_df[poll_df['Start Date'].apply(lambda x:x.startswith('2012-10'))]

fig = poll_df.plot('Start Date', 'Difference', figsize=(12,4), marker='o',linestyle='-',color='purple', xlim=(325,352))
plt.axvline(x=326, linewidth=4, color='gray')
plt.axvline(x=333, linewidth=4, color='gray')
plt.axvline(x=343, linewidth=4, color='gray')

# # 寄付のデータ
# 話題を変えて、両陣営への寄付に関するデータを分析していくことにします。
#
# これまでで一番大きなデータセット（約150MB)になります。ここからダウンロード出来ます , Notebookが起動しているフォルダと同じ場所に保存しておきましょう。
#
# このデータは、次の視点から分析を進めることにします。
#
# 1. ) 寄付の金額とその平均的な額
# 2. ) 候補者ごとの寄付の違い
# 3. ) 民主党と共和党での寄付の違い
# 4. ) 寄付した人々の属性について
# 5. ) 寄付の総額になんらかのパターンがあるか？

donor_df = pd.read_csv('lec66_Election_Donor_Data.csv')

donor_df.info()

donor_df.head()

donor_df.contb_receipt_amt.value_counts()

donor_df.contb_receipt_amt.value_counts().shape

don_mean = donor_df.contb_receipt_amt.mean()
don_std = donor_df.contb_receipt_amt.std()
print('平均{:0.2f} 標準偏差{:0.2f}'.format(don_mean, don_std))

top_donor = donor_df['contb_receipt_amt'].copy()
top_donor.sort_values()
top_donor

top_donor = top_donor[top_donor > 0]
top_donor.sort_values()

top_donor.value_counts().head(10)

com_don = top_donor[top_donor < 2500]
com_don.hist(bins=100)

#所属政党ごとに
candidates = donor_df.cand_nm.unique()

candidates

# 所属政党の辞書です。
party_map = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}
donor_df['Party'] = donor_df.cand_nm.map(party_map)

donor_df.head()

donor_df = donor_df[donor_df.contb_receipt_amt > 0]

#候補者ごとの寄付額
donor_df.groupby('cand_nm')['contb_receipt_amt'].count()

donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()

cand_amount = donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()

cand_amount.plot(kind='bar')

#政党ごとの寄付額
donor_df.groupby('Party')['contb_receipt_amt'].sum().plot(kind='bar')

#寄付した人の職業
occupation_df = donor_df.pivot_table('contb_receipt_amt',
                                    index='contbr_occupation',
                                    columns='Party', aggfunc='sum')

occupation_df.head()

occupation_df.shape

occupation_df = occupation_df[occupation_df.sum(1) > 1000000]

occupation_df.shape

occupation_df.plot(kind='bar')

occupation_df.plot(kind='barh',figsize=(10,12),cmap='seismic')

#CEOをまとめる
occupation_df

occupation_df.drop(['INFORMATION REQUESTED PER BEST EFFORTS','INFORMATION REQUESTED'],axis=0,inplace=True)

occupation_df

occupation_df.loc['CEO'] = occupation_df.loc['CEO'] + occupation_df.loc['C.E.O.']

occupation_df.drop('C.E.O.',inplace=True)

occupation_df.plot(kind='barh', figsize=(10,12),cmap='seismic')


