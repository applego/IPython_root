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
import matplotlib.pylab as plt

import os
import datetime

# +
#現在の西暦を取得
now = datetime.datetime.now()  #ex => datetime.datetime(2017, 7, 1, 23, 15, 34, 309309)
thisyear = now.year -1
slicedthisYear = str(thisyear)[2:4]
#slicedthisYear = "{0:02d}".format(thisyear)

#exsistDataYear = thisyear - 2009 + 1
# -

# ## 打者データの取得

#urlをリスト形式で取得
df_all = [] #各要素に各年のデータが入る
years = range(int(slicedthisYear),13,-1)

years

# +
urls = []

#URLを入力：2017年だけ命名規則が違う
for year in years:
#     if(year == int(slicedthisYear)):
#         urls.append('http://baseball-data.com/stats/hitter-all/avg-1.html')
#     else:
    urls.append('http://baseball-data.com/'+ "{0:02d}".format(year)+'/stats/hitter-all/avg-1.html')

#データをURLから取得
for url in urls:
    print('取得URL：'+url)
    df = pd.io.html.read_html(url)
    df = df[0]
    df_all.append(df)

# +
# url18 = 'http://baseball-data.com/'+ "{0:02d}".format(18)+'/stats/hitter-all/avg-1.html'
# print(url18)
# df = pd.io.html.read_html(url18)
# -

df_all[0]

df_all[1]

df_all[2]

# ## 選手IDの作成

name_list = []
dic = {}
for i in range(len(df_all)):
    name_list.extend(df_all[i]['選手名'])
name_list = list(set(name_list))
for i,name in enumerate(name_list):
    dic[name] = i

#選手IDの付与
for i in range(len(df_all)):
    df_all[i]['ID'] = -1
    for j in range(len(df_all[i])):
        df_all[i].loc[j,'ID'] = dic[df_all[i].loc[j,'選手名']]
    df_all[i].index = df_all[i]['ID']
    df_all[i] = df_all[i].drop('ID',axis=1)

#indexかぶりを除去
for i in range(len(df_all)):
    doubled_index = []
    count = df_all[i].index.value_counts()
    for j in count.index:
        if(count.loc[j]>1):
            doubled_index.append(j)
    df_all[i] = df_all[i].drop(doubled_index)

# +
#df_m =　pd.DataFrame(data=None,index=None,columns=None)

# +
#データ統合
#カラム名に年を付ける
for i in range(len(df_all)):
    for col_name in df_all[i].columns:
        df_all[i] = df_all[i].rename(columns = {col_name:col_name+"20"+"{0:02d}".format(years[i])})

df_m = pd.concat(df_all,axis=1)
# -

df_m.head()

#データの確認
'''
最後にデータの確認をします。今回は2017年の勝利数上位２０人の最近９年間の勝利数を見たいと思います。
'''
data = '打率'
data_col = ['選手名2017']
for col in df_m.columns:
    if '打率' in col:
        data_col.append(col)
df_2017 = pd.concat(df_all,axis=1)
df_2017 = df_2017.sort_values('打率2017',ascending=False)
df_2017 = df_2017[data_col]
df_2017.head(20)

# +
# print(df_m)
# if(not os.path.exists(os.path.dirname(os.path.abspath(__file__))+"\\"+os.path.basename(__file__)+"_output")):
#     os.mkdir(os.path.dirname(os.path.abspath(__file__))+"\\"+os.path.basename(__file__)+"_output")

# +
# df_m.head(20).to_csv(os.path.dirname(os.path.abspath(__file__))+"\\"+os.path.basename(__file__)+"_output"+"\\top20highAverageOf2017_winsoflast9ave"+"_"+datetime.datetime.now().strftime('%Y%m%d%H%M%S') +".csv",encoding='utf-8')
#df_all.to_csv("C:\\private\\myPrograms\\baseball\\プロ野球をPythonで分析するためにデータ集めでしたこと\\get_pitchers_data_output\\all_pitcher_last9years.csv",sep=',',index=True,encoding='utf-8')
#endregion
# -

type(df_all)

type(df_m)

df_all[0]

df_alldata = pd.DataFrame(df_all)

df_alldata.head()

df_alldata.info()

df_alldata.describe()

df_m.head()

# +
# scipyの統計パッケージも使います。
from scipy import stats

# 描画のためのライブラリです。
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ブラウザ内に画像を埋め込むための命令です
# %matplotlib inline
# -

#plt.hist(df_m.打率2018,bins=100)
years

for year in years:
    label = '打率20{}'.format(year)
    print(label)
    plt.hist(df_m[label],normed=True,alpha=.25,bins=20)

sns.jointplot(df_m.打率2018,df_m.打率2017,kind='hex')

sns.rugplot(df_m.打率2018)
plt.hist(df_m.打率2018,alpha=.3)

sns.kdeplot(df_m.打率2018)
sns.kdeplot(df_m.打率2017)
sns.kdeplot(df_m.打率2016)
sns.kdeplot(df_m.打率2015)
sns.kdeplot(df_m.打率2014)

plt.hist(df_m.OPS2018,cumulative=True,bins=20)

#df_m['RC272018'][df_m['RC272018']=='∞'] = np.nan
sns.jointplot(df_m.打率2018,df_m.OPS2018)

#df_m['RC272018'][df_m['RC272018']=='∞'] = np.nan
sns.jointplot(df_m.打率2018,df_m.OPS2018,kind='kde')

df_m.RC272018
sns.distplot(df_m.OPS2018.dropna())

sns.boxplot(data=[df_m.OPS2017,df_m.OPS2018])

sns.violinplot(data=df_m.OPS2017,inner='stick')

df_m.columns

# +
#変更前のデータフレーム名をdfとする
# df = df.rename(columns={'A':'id', 'B':'gender', 'C':'generation'})
# -

df_m.columns = df_m.columns.str.replace("順位", "rank")

df_m.columns = df_m.columns.str.replace("選手名", "player_name")

df_m.columns = df_m.columns.str.replace("チーム", "team")

df_m.columns = df_m.columns.str.replace("打率", "batting_average")

df_m.columns = df_m.columns.str.replace("試合", "numOfGames")

df_m.columns = df_m.columns.str.replace("numOfBatting", "plate_appearances")

df_m.columns = df_m.columns.str.replace("打数", "atbat")

df_m.columns = df_m.columns.str.replace("安打", "hits")

df_m.columns = df_m.columns.str.replace("本塁打", "HRｓ")

df_m.columns = df_m.columns.str.replace("打点", "RBI")

df_m.columns = df_m.columns.str.replace("四球", "walks")

df_m.columns = df_m.columns.str.replace("死球", "deadball")

df_m.columns = df_m.columns.str.replace("三振", "strike_out")

df_m.columns = df_m.columns.str.replace("犠打", "sacrifice_bunt")

df_m.columns = df_m.columns.str.replace("併殺打", "double_play")

df_m.columns = df_m.columns.str.replace("出塁率", "On_base_percentage")

df_m.columns = df_m.columns.str.replace("長batting_average", "Slugging_percentage")

df_m.info()

df_m.to_csv('npb_batter_2014-2018_1.csv')

from pandas import DataFrame, Series
dframe1 = DataFrame(df_m['OPS2018'])
dframe1.head()

dframe2 = DataFrame(df_m['OPS2017'])
dframe2.head()

dframe1 = pd.merge(dframe1,dframe2,on='ID')
dframe1.head()

df_m.replace('∞',np.inf)

df_m.to_csv('npb_batter_2014-2018_2.csv')

df_m2 = pd.read_csv('npb_batter_2014-2018_2.csv')

df_m2.head()

plt.plot(df_m2.RC272014)
plt.plot(df_m2.RC272015)
plt.plot(df_m2.RC272016)
plt.plot(df_m2.RC272017)
plt.plot(df_m2.RC272018)
plt.legend()
plt.show()

group_dena = df_m2['batting_average2014'].groupby(df_m2['team2014'])
group_dena

group_dena.mean()

for name,group in df_m2.groupby('team2015'):
    print('This is the {} group'.format(name))
    print(group)
    print('\n')

#クロス集計表
pd.crosstab(df_m2.team2014,df_m2.walks2014,margins=True)

# plt.hist(df_m2.groupby('team2018').RC272018,normed=True,alpha=0.2,bins=20)
plt.hist(df_m2.RC272018,density=True,alpha=0.3,bins=20)
plt.hist(df_m2.RC272017,density=True,alpha=0.3,bins=20)
plt.hist(df_m2.RC272016,density=True,alpha=0.3,bins=20)
plt.hist(df_m2.RC272015,density=True,alpha=0.3,bins=20)
plt.hist(df_m2.RC272014,density=True,alpha=0.3,bins=20)

sns.jointplot('OPS2018','RC272018',df_m2,kind='kde')

sns.jointplot('batting_average2018','RC272018',df_m2,kind='kde')

# それぞれのグラフのスタイルを変更することもできます。
sns.distplot(df_m2.batting_average2018,bins=25,
             kde_kws={'color':'indianred','label':'KDE PLOT'},
             hist_kws={'color':'blue','label':"HISTOGRAM"})

#散布
sns.lmplot('batting_average2018','RC272018',df_m2,
          scatter_kws={'marker':'o','color':'indianred'},
          line_kws={'linewidth':1,'color':'blue'})

sns.lmplot('ID','walks2018',df_m2)

sns.lmplot('batting_average2018','RC272018',df_m2,hue='team2018')

df_m2.tail(20)



df_m2.head(50)

# +
fig, (axis1,axis2) = plt.subplots(1,2,sharey=True)

sns.regplot('walks2018','OPS2018',df_m2,ax=axis1)
sns.violinplot(y='RC272018',x='walks2018',data=df_m2.sort_values('On_base_percentage2015'),ax=axis2)
# -

sns.heatmap(DataFrame(df_m2.RC272018))


