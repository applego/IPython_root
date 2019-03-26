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

df_m3 = pd.read_csv('npb_batter_2014-2018_2.csv')

df_m3.head()

df_m3.replace('日本ハム','Fighters')

df_m3 = df_m3.replace('日本ハム','Fighters')

df_m3 = df_m3.replace({'オリックス':'Orix','ヤクルト':'Yakult','中日':'Doragons','巨人':'Giants','阪神':'Tigers','広島':'Carp','楽天':'Rakuten','ソフトバンク':'Softbank','ロッテ':'Lotte','西武':'Lions'})

df_m3.columns = df_m3.columns.str.replace('盗塁','steal')
df_m3.head()

# +
namelist = []
for index, row in df_m3.iterrows():
#     print(type(row.player_name2018))
    if(type(row.player_name2018) is str):
        namelist.append(row.player_name2018)
    else:
        if(type(row.player_name2017) is str):
            namelist.append(row.player_name2017)
        else:
            if(type(row.player_name2016) is str):
                namelist.append(row.player_name2016)
            else:
                if(type(row.player_name2015) is str):
                    namelist.append(row.player_name2015)
                else:
                    if(type(row.player_name2014) is str):
                        namelist.append(row.player_name2014)
                    else:
                        namelist.append('')
    
#     row.player_name2018.isnull()
#     print(index)
#     print('~~~~~~')

#     print(type(row))
#     print(row)
#     print('------')

#     print('======\n')
# -

namelist = [i.replace('\u3000',' ') for i in namelist]

namelist
df_m3['player_name'] = DataFrame(namelist)

df_m3.drop(['player_name2018','player_name2017','player_name2016','player_name2015','player_name2014'],axis=1,inplace=True)
df_m3.head()

#タイタニック参考
df_m3.info()

sns.countplot('team2018',data=df_m3)

sns.countplot('team2017',data=df_m3)

# df_m3.to_csv('npb_batter_2014-2018_3.csv')
df_m3 = pd.read_csv('npb_batter_2014-2018_3.csv')

df_m3.RC272018.hist(bins=70)

df_m3.RC272018.mean()

fig = sns.FacetGrid(df_m3,hue='team2018',aspect=4)
fig.map(sns.kdeplot,'OPS2018',shade=True)
maxest = df_m3['OPS2018'].max()
fig.set(xlim=(0,maxest))
fig.add_legend()

#OPS2018 kdeplot
fig = sns.FacetGrid(df_m3,hue='team2018',aspect=4)
fig.map(sns.kdeplot,'OPS2018',shade=True)
# maxest = df_m3['OPS2018'].max()
fig.set(xlim=(0,2))
fig.add_legend()

#HRs kdeplot
fig = sns.FacetGrid(df_m3,hue='team2018',aspect=4)
fig.map(sns.kdeplot,'HRｓ2018',shade=True)
maxest = df_m3['HRｓ2018'].max()
fig.set(xlim=(0,maxest))
fig.add_legend()

#On_base_percentage2018 kdeplot
fig = sns.FacetGrid(df_m3,hue='team2018',aspect=4)
fig.map(sns.kdeplot,'On_base_percentage2018',shade=True)
maxest = df_m3['On_base_percentage2018'].max()
fig.set(xlim=(0,maxest))
fig.add_legend()

df_m3.groupby('team2018').OPS2018.mean()

plt.plot(df_m3.groupby('team2018').OPS2018.mean())

df_m3.head(1)

sns.lmplot('batting_average2018','OPS2018',data=df_m3,palette='hls')

sns.lmplot('batting_average2018','OPS2018',hue='team2018',data=df_m3,palette='hls')

df_m3.head()

Y_over3wari = []

for index, row in df_m3.iterrows():
    if(row.batting_average2018 > 0.3):
        Y_over3wari.append(1)
    else:
        Y_over3wari.append(0)

Y_over3wari = DataFrame(Y_over3wari)

Y_over3wari

#ロジスティック回帰参考
df_logi2018 = df_m3.loc[:, df_m3.columns.str.endswith('2018')]

df_logi2018.head()


def check_over3wari(x):
    if x>=0.3:
        return 1
    else:
        return 0


df_logi2018['over3wari'] = df_logi2018['batting_average2018'].apply(check_over3wari)

df_logi2018 = pd.concat([df_logi2018,df_m3['player_name']],axis=1)

df_logi2018

#犠牲バントと打率３割超えか
sns.countplot('sacrifice_bunt2018',data=df_logi2018.sort_values('sacrifice_bunt2018'),hue='over3wari',palette='coolwarm')

#三振数と打率３割超えか
sns.countplot('strike_out2018',data=df_logi2018.sort_values('strike_out2018'),hue='over3wari',palette='coolwarm')

df_logi2018.groupby('over3wari').mean()

#目的変数「Had_Affair」を削除します。
X = df_logi2018.drop(['over3wari'],axis=1)

Y = np.ravel(Y_over3wari)

Y

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
log_model = LogisticRegression()

X.head(1)

team_dummier = pd.get_dummies(df_logi2018['team2018'])
team_dummier.head()

# X = X.drop('player_name',axis=1)
X = pd.concat([X,team_dummier],axis=1)

X = X.drop('team2018',axis=1)
X.head()

X = X.drop('batting_average2018',axis=1)
X.head()

test = X.dropna()

X.info()

test.info()

test2018 = df_logi2018.dropna()

test2018.head()

team_dummies = pd.get_dummies(df_logi2018.dropna()['team2018'])

X = test2018.drop(['player_name','team2018','over3wari'],axis=1)

X.columns

X.info()

Y = df_logi2018.dropna()['over3wari']

Y =Y.values

# +
# X = pd.concat([X,team_dummies],axis=1)
# X.info()
# -

X.info()

team_dummies.info()

X.shape

Y.shape

log_model.fit(X,Y)

log_model.score(X,Y)

Y.mean()

coeff_df = DataFrame([X.columns, log_model.coef_[0]]).T
coeff_df.columns=['col1','col2']
coeff_df

coeff_df.plot(x='col1',y='col2',kind='bar')

#打率と打率３割超えか→係数高くない笑
sns.countplot('batting_average2018',data=df_logi2018.sort_values('batting_average2018'),hue='over3wari',palette='coolwarm')

# +
# おなじく、train_test_splitを使います。
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# 新しいモデルを作ります。
log_model2 = LogisticRegression()

# 学習用のデータだけでモデルを鍛えます。
log_model2.fit(X_train, Y_train)

# +
# テスト用データを使って、予測してみましょう。
class_predict = log_model2.predict(X_test)

# もう一つ、性能の評価用に
from sklearn import metrics

# 精度を計算してみます。
print(metrics.accuracy_score(Y_test,class_predict))
# -

X_drophits = X.drop(['hits2018'],axis=1)

log_model3_drophits = LogisticRegression()

log_model3_drophits.fit(X_drophits,Y)

log_model3_drophits.score(X_drophits,Y)

# +
# おなじく、train_test_splitを使います。
X_train, X_test, Y_train, Y_test = train_test_split(X_drophits, Y)

# 新しいモデルを作ります。
log_model_dh = LogisticRegression()

# 学習用のデータだけでモデルを鍛えます。
log_model_dh.fit(X_train, Y_train)

# +
# テスト用データを使って、予測してみましょう。
class_predict = log_model_dh.predict(X_test)

# 精度を計算してみます。
print(metrics.accuracy_score(Y_test,class_predict))
# -

X.head(1)

# 打席数１００以上で

X_over100daseki = X[X['plate_appearances2018']>100]
X_over100daseki.info()

X_over100daseki.plate_appearances2018.mean()

X_over100daseki.shape

Y_over100daseki = df_logi2018.dropna()[df_logi2018['plate_appearances2018']>100]['over3wari']

Y_over100daseki.head()

Y_over100daseki.shape

type(Y_over100daseki)

Y_over100daseki = np.ravel(Y_over100daseki)
Y_over100daseki

log_model_over100daseki = LogisticRegression()

log_model_over100daseki.fit(X_over100daseki,Y_over100daseki)

log_model_over100daseki.score(X_over100daseki,Y_over100daseki)

Y_over100daseki.mean()

coeff_df_over100daseki = DataFrame([X_over100daseki.columns,log_model_over100daseki.coef_[0]]).T
coeff_df_over100daseki.columns=['col1','col2']
coeff_df_over100daseki

coeff_df_over100daseki.plot(x='col1',y='col2',kind='bar')

# drop hits2018

X_over100daseki_dh = X_over100daseki.drop('hits2018',axis=1)
X_over100daseki_dh.head(1)

log_model_over100daseki_dh = LogisticRegression()

log_model_over100daseki_dh.fit(X_over100daseki_dh,Y_over100daseki)

log_model_over100daseki_dh.score(X_over100daseki_dh,Y_over100daseki)

coeff_df_over100daseki_dh = DataFrame([X_over100daseki_dh.columns,log_model_over100daseki_dh.coef_[0]]).T
coeff_df_over100daseki_dh.columns = ['col1','col2']
coeff_df_over100daseki_dh

coeff_df_over100daseki_dh.plot(x='col1',y='col2',kind='bar')

X_over100daseki_dh_drcxr = X_over100daseki_dh.drop(['RC272018','XR272018'],axis=1)

log_model_over100daseki_dh_drcxr = LogisticRegression()

log_model_over100daseki_dh_drcxr.fit(X_over100daseki_dh_drcxr,Y_over100daseki)

log_model_over100daseki_dh_drcxr.score(X_over100daseki_dh_drcxr,Y_over100daseki)

coeff_df_over100daseki_dh_drcxr = DataFrame([X_over100daseki_dh_drcxr.columns,log_model_over100daseki_dh_drcxr.coef_[0]]).T
coeff_df_over100daseki_dh_drcxr.columns = ['col1','col2']
coeff_df_over100daseki_dh_drcxr

coeff_df_over100daseki_dh_drcxr.plot(x='col1',y='col2',kind='bar')

# 打率が関係するのは消す

X_over100daseki_dh_drcxr_andmore = X_over100daseki_dh_drcxr.drop(['rank2018','batting_average2018','Slugging_percentage2018','OPS2018','On_base_percentage2018'],axis=1)

X_over100daseki_dh_drcxr_andmore.head()

log_model_over100daseki_dh_drcxr_andmore = LogisticRegression()

log_model_over100daseki_dh_drcxr_andmore.fit(X_over100daseki_dh_drcxr_andmore,Y_over100daseki)

log_model_over100daseki_dh_drcxr_andmore.score(X_over100daseki_dh_drcxr_andmore,Y_over100daseki)

coeff_df_over100daseki_dh_drcxr_andmore = DataFrame([X_over100daseki_dh_drcxr_andmore.columns,log_model_over100daseki_dh_drcxr_andmore.coef_[0]]).T

coeff_df_over100daseki_dh_drcxr_andmore.columns = ['col1','col2']
coeff_df_over100daseki_dh_drcxr_andmore

coeff_df_over100daseki_dh_drcxr_andmore.plot(x='col1',y='col2',kind='bar')

# +
# おなじく、train_test_splitを使います。
X_train, X_test, Y_train, Y_test = train_test_split(X_over100daseki_dh_drcxr_andmore, Y_over100daseki)

# 新しいモデルを作ります。
log_model_over100daseki_dh_drcxr_andmore_split = LogisticRegression()

# 学習用のデータだけでモデルを鍛えます。
log_model_over100daseki_dh_drcxr_andmore_split.fit(X_train, Y_train)

# +
# テスト用データを使って、予測してみましょう。
class_predict = log_model_over100daseki_dh_drcxr_andmore_split.predict(X_test)

# 精度を計算してみます。
print(metrics.accuracy_score(Y_test,class_predict))
# -

# 適合率、再現率

# +
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, f1_score

y_true = [0,0,0,1,1,1]
y_pred = [1,0,0,1,1,1]
confmat = confusion_matrix(y_true, y_pred, labels=[1, 0])

print (confmat)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print(precision)
print(recall)
# -


