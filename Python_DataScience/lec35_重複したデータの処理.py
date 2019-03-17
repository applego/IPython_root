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

from pandas import DataFrame

dframe = DataFrame({'key1':['A']*2+['B']*3,
                   'key2':[2,2,2,3,3]})
dframe

dframe.duplicated()

dframe.drop_duplicates()

dframe.drop_duplicates(['key1'])

dframe

dframe.drop_duplicates(['key1'],ta=True)


