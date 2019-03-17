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

# +
import numpy as np
from numpy.random import randn

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
# -

tips = sns.load_dataset('tips')
#チップ　レストラン飲食代別心付けのデータ

tips.head()

sns.lmplot('total_bill', 'tip', tips)

sns.lmplot('total_bill', 'tip', tips,
          scatter_kws={'marker':'o', 'color':'indianred'},
          line_kws={'linewidth':1, 'color':'blue'})

sns.lmplot('total_bill', 'tip', tips, order=4,
          scatter_kws={'marker':'o', 'color':'indianred'},
          line_kws={'linewidth':1, 'color':'blue'})

sns.lmplot('total_bill','tip',tips,fit_reg=False)

tips['tip_pect'] = 100*(tips['tip']/tips['total_bill'])
tips.head()

sns.lmplot('size','tip_pect', tips)

sns.lmplot('size','tip_pect', tips, x_jitter=0.2)

sns.lmplot('size','tip_pect', tips, x_estimator=np.mean)

sns.lmplot('total_bill', 'tip_pect', tips, hue='sex', markers=['x','o']) #hue 色調、傾向

sns.lmplot('total_bill', 'tip_pect', tips, hue='day') #hue 色調、傾向

sns.lmplot('total_bill', 'tip_pect', tips, lowess=True, line_kws={'color':'black'}) 

sns.regplot('total_bill', 'tip_pect', tips)

fig, (axis1, axis2) = plt.subplots(1,2, sharey=True)
sns.regplot('total_bill', 'tip_pect', tips, ax=axis1)
sns.violinplot(y='tip_pect', x='size', data=tips.sort_values('size'), ax=axis2)


