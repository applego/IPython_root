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

import pandas as pd

# +
from io import StringIO

data = """Sample Animal Intelligence
1 Dog Dumb
2 Dog Dumb
3 Cat    Smart
4 Cat  Smart
5 Dog  Smart
6 Cat  Smart"""
dframe = pd.read_table(StringIO(data),sep='\s+')
# -

dframe

pd.crosstab(dframe.Animal, dframe.Intelligence)

pd.crosstab(dframe.Animal, dframe.Intelligence, margins=True)


