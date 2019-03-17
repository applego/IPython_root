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

dframe = DataFrame({'city':['Alma','Brian Head','Fox Park'],
                   'altitude':[3158,3000,2762]})
dframe

state_map = {'Alma':'Colorado','Brian Head':'Utah','Fox Park':'Wyoming'}

dframe['state'] = dframe['city'].map(state_map)

dframe


