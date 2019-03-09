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

arr1 = np.arange(9).reshape((3,3))
arr1

np.concatenate([arr1,arr1],axis=1)

np.concatenate([arr1,arr1],axis=0)

ser1 = Series([0,1,2], index=['T','U','V'])
ser2 = Series([3,4], index=['X','Y'])

ser1

ser2

pd.concat([ser1,ser2])

pd.concat([ser1,ser2,ser1])

pd.concat([ser1,ser2,ser1],axis=1,sort=False)

pd.concat([ser1,ser2,ser1],axis=1,sort=True)

pd.concat([ser1,ser2],keys=['cat1','cat2'])

pd.concat([ser1,ser2],keys=['cat1','cat2'],axis=1,sort=False)

dframe1 = DataFrame(np.random.randn(4,3), columns=['X','Y','Z'])
dframe1

dframe2 = DataFrame(np.random.randn(3,3), columns=['Y','Q','X'])
dframe2

pd.concat([dframe1,dframe2],sort=False)

pd.concat([dframe1,dframe2],sort=False,axis=1)

pd.concat([dframe1,dframe2],sort=False,ignore_index=True)

pd.concat([dframe2,dframe1],sort=False,ignore_index=True)


