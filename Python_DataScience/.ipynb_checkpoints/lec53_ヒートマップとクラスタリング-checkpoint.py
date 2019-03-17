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

import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
# -

flight_dframe = sns.load_dataset('flights')
flight_dframe.head()

flight_dframe = flight_dframe.pivot('month','year','passengers')

flight_dframe

sns.heatmap(flight_dframe)

sns.heatmap(flight_dframe, annot=True, fmt='d')

sns.heatmap(flight_dframe, center=flight_dframe.loc['January',1955])

# +
f, (axis1, axis2) = plt.subplots(2,1)

yearly_flights = flight_dframe.sum()

years = pd.Series(yearly_flights.index.values)
years = pd.DataFrame(years)

flights = pd.Series(yearly_flights.values)
flights = pd.DataFrame(flights)

year_dframe = pd.concat((years,flights),axis=1)
year_dframe.columns = ['Year', 'Flights']

sns.barplot('Year', y='Flights', data=year_dframe, ax=axis1)
sns.heatmap(flight_dframe, cmap='Blues', ax=axis2, cbar_kws={'orientation':'horizontal'})
# -

# ### 自分でやってみる

# +
import numpy as np
from numpy.random import randn
import pandas as pd 

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# %matplotlib inline
# -

flight_dframe = sns.load_dataset('flights')

flight_dframe.head()

flight_dframe = flight_dframe.pivot('month','year','passengers')
flight_dframe

sns.heatmap(flight_dframe, cmap='YlGn')

sns.heatmap(flight_dframe,annot=True,fmt='d',cmap='Reds')

sns.heatmap(flight_dframe,cmap='binary',center=flight_dframe.loc['January',1955])

# +
f, (axis1,axis2) = plt.subplots(2,1)

yearly_flights = flight_dframe.sum()
#print(yearly_flights)
#print(yearly_flights.index.values)
years = pd.Series(yearly_flights.index.values)
years = pd.DataFrame(years)

flights = pd.Series(yearly_flights.values)
flights = pd.DataFrame(flights)
#plt.hist(flights)
year_dframe = pd.concat((years,flights), axis=1)
year_dframe.columns = ['Year','Flights']

sns.barplot('Year',y='Flights', data=year_dframe,ax=axis1)
sns.heatmap(flight_dframe,cmap='YlGn', ax=axis2, cbar_kws={'orientation':'horizontal'})
# -

print('yearly_flights')
yearly_flights.head()

print('year')
years.head()

print('flights')
flights.head()

print('year_dframe')
year_dframe.head()

#クラスタリング　似たものが近くに
sns.clustermap(flight_dframe)

sns.clustermap(flight_dframe, col_cluster=False)

sns.clustermap(flight_dframe,row_cluster=False)

sns.clustermap(flight_dframe, standard_scale=1)

sns.clustermap(flight_dframe, standard_scale=0)

sns.clustermap(flight_dframe, z_score=1)


