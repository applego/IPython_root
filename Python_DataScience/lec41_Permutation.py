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

dframe = DataFrame(np.arange(4 * 4).reshape((4,4)))
dframe

blender = np.array([0,3,2,1])

dframe.take(blender)

blender = np.random.permutation(4)#順列
blender

dframe.take(blender)

box = np.array(['A','B','C'])

box

print(len(box))
shaker = np.random.randint(0,len(box),size=10)

shaker

hand_grabs = box.take(shaker)

hand_grabs


