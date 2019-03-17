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

ser1 = Series([2, np.nan, 4, np.nan, 6, np.nan],
             index=['Q','R','S','T','U','V'])
ser1

ser2 = Series(np.arange(len(ser1), dtype=np.float64),
              index=['Q','R','S','T','U','V'])
ser2

Series(np.where(pd.isnull(ser1),ser2,ser1) ,index=ser1.index)

ser1.combine_first(ser2)

dframe_odds = DataFrame({'X':[1., np.nan, 3., np.nan],
                        'Y':[np.nan, 5., np.nan, 7.],
                        'Z':[np.nan, 9., np.nan,11.]})
dframe_odds

dframe_evens = DataFrame({'X':[2.,4., np.nan, 6.,8.],
                        'Y':[np.nan, 10.,12.,14.,16.]})
dframe_evens


