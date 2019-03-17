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
from pandas import Series, DataFrame

animals = DataFrame(np.arange(4*4).reshape((4,4)),
                   columns=['W','X','Y','Z'],
                   index=['Dog','Cat','Bird','Mouse'])

animals

animals.ix[1:2,['W','Y']] = np.nan

animals

behavior_map = {'W':'bad','X':'good','Y':'bad','Z':'good'} #辞書型

animals_col = animals.groupby(behavior_map, axis=1)

animals_col.sum()

behavior_series = Series(behavior_map)

behavior_series

animals.groupby(behavior_series, axis=1).count()

animals

animals.groupby(len).sum() #文字サイズでグループ化

け
