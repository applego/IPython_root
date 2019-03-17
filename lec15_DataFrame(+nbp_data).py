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

#tmp_header = pd.read_clipboard()
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






