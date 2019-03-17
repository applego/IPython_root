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

# +
# このセルの内容は、理解できなくても大丈夫です。
# ただサンプルになるデータを作りたいだけですので、出来たデータをどのような加工するかに注目してください。

import pandas.util.testing as tm
tm.N = 3

# ちょっとした関数を作ります。
def unpivot(frame):
    N, K = frame.shape
    data = {'value' : frame.values.ravel('F'),
            'variable' : np.asarray(frame.columns).repeat(N),
            'date' : np.tile(np.asarray(frame.index), K)}
    return DataFrame(data, columns=['date', 'variable', 'value'])

# DataFrameを作ります。
dframe = unpivot(tm.makeTimeDataFrame())
# -

dframe

dframe_piv = dframe.pivot('date','variable','value')
dframe_piv


