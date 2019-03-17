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

dframe1 = DataFrame(np.arange(8).reshape((2,4)),
                    index=pd.Index(['LA','SF'], name='city'),
                    columns=pd.Index(['A','B','C','D'],name='letter'))

dframe1

dframe1.stack()

type(dframe1.stack())

dframe_st = dframe1.stack()
dframe_st

dframe_st.unstack()

type(dframe_st)

type(dframe_st.unstack())

dframe_st.unstack(level=0)

dframe_st.unstack('city')

dframe_st.unstack(level=1)

dframe_st.unstack('letter')

ser1 = Series([0,1,2], index=['Q','X','Y'])
ser2 = Series([4,5,6], index=['X','Y','Z'])

dframe = pd.concat([ser1,ser2], keys=['Alpha','Beta'])
dframe

dframe.unstack()

dframe.unstack().stack()
