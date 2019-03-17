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
from pandas import DataFrame

np.random.seed(12345)

dframe = DataFrame(np.random.randn(1000,4))
#　同じシード→同じ乱数

dframe.head()

type(np.random.randn(100,3))

dframe.tail()

dframe.describe()

col = dframe[0]
col.head()

col[np.abs(col)>3]

np.abs(dframe)>3

dframe[(np.abs(dframe)>3).any(1)]

# 3以上を外れ値と考える
np.sign(dframe)

dframe[np.abs(dframe)>3] = np.sign(dframe)
