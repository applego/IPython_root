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

import numpy as np
import pandas as pd
from pandas import DataFrame

dframe = DataFrame({'k1':['X','X','Y','Y','Z'],
                   'k2':['alpha','beta','alpha','beta','alpha'],
                   'dataset1':np.random.randn(5),
                   'dataset2':np.random.randn(5)})
dframe

group1 = dframe['dataset1'].groupby(dframe['k1'])

group1

group1.mean()

cities = np.array(['NY','LA','LA','NY','NY'])
month = np.array(['JAN','FEB','JAN','FEB','JAN'])
#それぞれグループ化
dframe['dataset1'].groupby([cities,month]).mean()

dframe.groupby('k1').mean()

dframe

dframe.groupby(['k1','k2']).mean()

dataset2_group = dframe.groupby(['k1','k2'])[['dataset2']]

dataset2_group.mean()

dframe.groupby(['k1']).size()

for name, group in dframe.groupby('k1'):
    print('This is the {} group'.format(name))
    print(group)
    print('\n')

for (k1,k2), group in dframe.groupby(['k1','k2']):
    print('Ke1 = {} Key2 = {}'.format(k1,k2))
    print(group)
    print('\n')

gr = dframe.groupby('k1')

gr.get_group('X')

group_dict = dict(list(dframe.groupby('k1')))

group_dict['X']

group_dict_axis1 = dict(list(dframe.groupby(dframe.dtypes,axis=1)))

group_dict_axis1


