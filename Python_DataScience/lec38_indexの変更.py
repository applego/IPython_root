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

dframe = DataFrame(np.arange(12).reshape((3,4)),
                  index=['NY','LA','SF'],
                  columns=['A','B','C','D'])
dframe

str.lower('A')

dframe.index.map(str.lower)

dframe.index = dframe.index.map(str.lower)
dframe

str.title('udemy is good')

dframe.rename(index=str.title, columns=str.lower)

dframe

dframe.rename(index={'ny':'NEW YORK'},
             columns={'A':'ALPHA'})

dframe.rename(index={'ny':'NEW YORK'}, inplace=True)
dframe


