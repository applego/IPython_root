# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

import numpy as np
from pandas import Series, DataFrame
import pandas as pd

import webbrowser
website = 'http://en.wikipedia.org/wiki/NFL_win-loss_records'
webbrowser.open(website)

nfl_frame = pd.read_clipboard()
nfl_frame

npb_frame = pd.read_clipboard(header=None)
npb_frame

tmp_header = pd.read_clipboard()
tmp_header

nfl_frame.columns

npb_frame.columns

nfl_frame['First NFL Season']

npb_frame[1]

nfl_frame.Team

DataFrame(nfl_frame,columns=['Team','First NFL Season','Total Games'])

DataFrame(npb_frame,columns=[0,1])

DataFrame(nfl_frame,columns=['Team','First NFL Season','Total Games','Stadium'])

nfl_frame.head(3)

nfl_frame.tail(3)

npb_frame.ix[30][:]

nfl_frame['Stadium'] = "Levi's Stadium"
nfl_frame

nfl_frame["Stadium"] = np.arange(6)
nfl_frame

stadiums = Series(["Levi's Stadium","AT&T Stadium"],index=[4,0])
stadiums

nfl_frame['Stadium'] = stadiums
nfl_frame

del nfl_frame['Stadium']
nfl_frame

# +
data = {'City':['SF','LA','NYC'],
       'Population':[837000,3880000,8400000]}

city_frame = DataFrame(data)
city_frame
# -

# # NPB　return

npb_frame.head()

npb_frame.columns=["順位","選手","チーム","打率","試合","打席","打数","得点","安打","二塁打","三塁打","本塁打","塁打","打点","盗塁","盗塁刺","犠打","犠飛","四球","故意四球","死球","三振","併殺打","長打率","出塁率"]

npb_frame.head()

npb_frame.tail()

npb_frame.loc[30,'長打率'] = '.355'

npb_frame.loc[30,'出塁率'] = '0.301'

ser1 = npb_frame['打率']
ser1

ser1.describe()

ser1.rank()

ser1[ser1>0.3]

import seaborn as sns

#sns.rugplot(npb_frame.打率)
sns.distplot(npb_frame.打率, rug=True,bins=30,color='blue')

sns.kdeplot(npb_frame.打率, label='average')

sns.boxplot(npb_frame.打率)

# # wOBA算出方法
# Introduction 04. 「打者の評価（weighted On-Base Average）」
#
# 得点への貢献度を重要視、相関関係を向上
#
# 得点への貢献度を重要視、相関関係を向上
# 打者の評価指標は歴史上多く提唱されてきたところで、一般的にはセイバーメトリクスの指標としてOPSが有名だ。だが現在MLBのセイバーメトリクス界隈ではwOBA（weighted On-Base Average）が主流となっている。
# wOBA（NPB版）=｛0.692×（四球−故意四球）＋0.73×死球＋0.966×失策出塁＋0.865×単打＋1.334×二塁打＋1.725×三塁打＋2.065×本塁打｝÷（打数＋四球−故意四球＋死球＋犠飛）
#
# wOBAは四死球・単打・二塁打・三塁打・本塁打それぞれに得点期待値から得点の価値を割り振り、それによって打者を評価する指標だ。出塁率と同じ感覚で扱えるようになっており、平均値は.330くらいになる。なお、盗塁を加味して計算する場合もある。
#
# 一言で表現すればwOBAは打者が打席あたりにどれだけ得点の増加に貢献する打撃をしているかを表す指標であり、四死球を評価に含める点、さらに長打の価値を区別する点で打率よりも的確に打者の得点への貢献度を表す指標といえる。OPSとの比較では、各項目への加重が得点への影響度の実態に即しているという点でより優れている。
#
# wOBAを見る際のポイントとしては、それが出塁率のように打席あたりの率という形をしていることと、それ自体が直接に得点の単位をしているわけではないということだ。つまり同じwOBAでも打席数が多いほうが創出した得点の絶対量は多いことになる。得点との関係で言えば、各項目の加重の比率は得点価値に即したものとなっているが、例えばwOBAが.400だからといって「1打席で0.4点を創出する」といった読み方ができるわけではない。
#
# 具体的に得点数の意味でどれだけ貢献をしているかを把握するためには、wOBAに少し手を加えてwRAA（weighted Runs Above Average）に変換する必要がある。

ser_slugave = Series(npb_frame.長打率)
ser_slugave = ser_slugave.astype('float')

ser_onbaseper = Series(npb_frame.出塁率)
ser_onbaseper = ser_onbaseper.astype('float')

ser_ops = ser_onbaseper + ser_slugave

npb_frame['ops'] = ser_ops
npb_frame.head()

tes1 = npb_frame.loc[0]
tes1['打率']

# wOBA（NPB版）=
# ｛0.692×（四球−故意四球）＋0.73×死球＋0.966×失策出塁＋0.865×単打＋1.334×二塁打＋1.725×三塁打＋2.065×本塁打｝
# ÷（打数＋四球−故意四球＋死球＋犠飛）

wOBA_list = []
for i in np.arange(len(npb_frame)):
    ser = Series(npb_frame.ix[i])
#失策出塁が適当（３で固定）なのでDeltaと合わない
    woba = (0.9692*(ser['四球']-ser['故意四球'])+0.73*ser['死球']+0.966*3+0.865*(ser['安打']-ser['二塁打']-ser['三塁打']-ser['本塁打'])+1.334*ser['二塁打']+1.725*ser['三塁打']+2.065*ser['本塁打'])/(ser['打数']+ser['四球']-ser['故意四球']+ser['死球']+ser['犠飛'])
    wOBA_list.append(woba)

print(wOBA_list)
type(wOBA_list[0])
# ser_woba =　Series(wOBA_list)
ser_woba = Series(wOBA_list)
npb_frame['wOBA_失策出塁3'] = ser_woba

npb_frame.to_csv('lec15_mergedata.csv')

# # WAR算出方法
# ## 野手
# ### Fangraphs版
# Beyond the Boxscoreのジェフ・アベールの解説に沿って[9]、2008年のマット・ホリデイを例に大まかな算出の流れを記す。各指標の詳細は、FanGraphs Sabermetrics Libraryを参照。


