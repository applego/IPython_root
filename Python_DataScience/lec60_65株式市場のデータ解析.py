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

# # 株式市場のデータ解析
# 株のデータを解析して、未来の株価が分かったら大金持ちになれるかも知れません。 それはさておき、Pythonと周辺ライブラリを使うと、株価データのような、時系列データの解析も比較的簡単に行う事ができます。
#
# ## 次のような課題について考えて行くことにしましょう。
#
# 1. ) 株価の時間による変化を見てみる。
# 2. ) 日ごとの変動を可視化する。
# 3. ) 移動平均を計算する
# 4. ) 複数の株価の終値の相関を計算する
# 5. ) 複数の株価の変動の関係を見る
# 6. ) 特定の株のリスクを計算する
# 7. ) シミュレーションを使った未来の予測
#
# ## 株価データの基本
# pandasを使って株価のデータを扱う基本を学んで行きましょう。

# +
import pandas as pd 
from pandas import Series, DataFrame
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline
# -

from pandas_datareader import DataReader

from datetime import datetime

tech_list = ['AAPL','GOOG','MSFT','AMZN']

end = datetime.now()
start = datetime(end.year-1,end.month,end.day)

for stock in tech_list:
    globals()[stock] = DataReader(stock, 'yahoo',  start, end)

type(AAPL )

AAPL.describe()

AAPL.info()

AAPL['Adj Close'].plot(legend=True, figsize=(10,4))

AAPL['Volume'].plot(legend=True, figsize=(10,4))

#移動平均線 10日平均
ma_day = [10,20,50]
for ma in ma_day:
    column_name = 'MA {}'.format(ma)
    #     AAPL[column_name] = pd.rolling_mean(AAPL['Adj Close'], ma)
    AAPL[column_name] = AAPL['Adj Close'].rolling(ma).mean()

AAPL.head()

AAPL[['Adj Close', 'MA 10','MA 20', 'MA 50']].plot(subplots=False,figsize=(10,4))

#昨日の終値との比較
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

AAPL.head()

AAPL['Daily Return'].plot(figsize=(10,4),legend=True, linestyle='--', marker='o')

sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color='purple')

AAPL['Daily Return'].hist(bins=100)

closing_df = DataReader(['AAPL','GOOG','MSFT','AMZN'],'yahoo', start, end)['Adj Close']

closing_df.head()

tech_rets = closing_df.pct_change()

tech_rets.head()

sns.jointplot('GOOG','GOOG', tech_rets, kind='scatter', color='seagreen')

sns.jointplot('GOOG','MSFT', tech_rets, kind='scatter', color='seagreen')

# # 10年分にしてみる

start = datetime(end.year-10,end.month,end.day)
closing_df2 = DataReader(['AAPL','GOOG','MSFT','AMZN'],'yahoo', start, end)['Adj Close']

closing_df2.head()

sns.jointplot('GOOG','GOOG',data=tech_rets,kind='scatter',color='purple')

sns.jointplot('GOOG','MSFT',data=tech_rets,kind='scatter',color='purple')

# 2つの会社の株価の変化率は相当関係があることがわかります。pearsonrは相関係数(正確には、ピアソン積率相関係数）ですが、0.52と正に相関していることを示しています。
#
# url - https://ja.wikipedia.org/wiki/%E7%9B%B8%E9%96%A2%E4%BF%82%E6%95%B0
#
# 相関係数について、感覚的な理解を助けてくれる図を紹介しておきます。

from IPython.display import SVG
SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')

# 2社の間の比較は、色々な組み合わせを考える事が出来ますが、Seabornを使うと、このような比較をすべてのパターンについて、簡単にやってくれます。 それが、sns.pairplot() です。

sns.pairplot(tech_rets.dropna())

returns_fig = sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)

returns_fig = sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter, color='green')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)

returns_fig = sns.PairGrid(closing_df2)
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)

sns.heatmap(tech_rets.corr(), annot=True)

#リターンとリスク
rets = tech_rets.dropna()

rets.head()

plt.scatter(rets.mean(), rets.std(), alpha=.5, s=np.pi*20)

# +
plt.scatter(rets.mean(), rets.std(), alpha=.5, s=np.pi*20)
plt.ylim([0.01,0.030])
plt.xlim([-0.005, 0.01])

plt.xlabel('Expected returns')
plt.ylabel('Risk')

for label, x,y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label,xy=(x,y), xytext=(0,50),
                textcoords = 'offset points', ha = 'right', va='bottom',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3'))
# -

#value at risk
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')

rets['AAPL'].quantile(0.05)

days = 365
dt = 1/days
mu = rets.mean()['GOOG']
sigma = rets.std()['GOOG']


def stock_monte_carlo(start_price, days, mu,sigma):
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1, days):
        shock[x] = np.random.normal(loc=mu*dt, scale=sigma * np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1]*(drift[x]+shock[x]))
    return price


GOOG.head()

# +
start_price = GOOG.iloc[0,5]

for num in range(5):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis')
# -

runs = 10000
simulations = np.zeros(runs)
np.set_printoptions(threshold=5)
for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]

plt.hist(simulations, bins=200)

# +
q = np.percentile(simulations, 1)
plt.hist(simulations, bins=200)

plt.figtext(0.6, 0.8, s='Start price: {:0.2f}'.format(start_price))
plt.figtext(0.6, 0.7, 'mean final price: {:0.2f}'.format(simulations.mean()))
plt.figtext(0.6, 0.6, 'VaR(0.99): {:0.2f}'.format(start_price-q))
plt.figtext(0.15, 0.6, 'q(0.99): {:0.2f}'.format(q))

plt.axvline(x=q, linewidth=4,color='r')
# -

# シミュレーションで、グーグルの株価のVaRを計算することができました。1年という期間、99%の信頼区間でのVaRは、1株（526.4ドル）あたり、18.38ドルであることがわかります。99%の可能性で、損失はこれ以内に収まる計算になるわけです。
#
# お疲れ様でした。ひとまず、株価のデータ解析を終えることができました。 追加の課題をいくつか考える事ができます。
#
# 1.) このレクチャーで学んだVaRを計算する2つの方法を、ハイテク株では無い銘柄に適用してみましょう。
#
# 2.) 実際の株価でシミュレーションを行い、リスクの予測やリターンについて検証してみましょう。過去のデータから現在の株価を予測することで、これが出来るはずです。
#
# 3.) 関連のある銘柄同士の値動きに注目してみましょう。ペアトレードという投手法が実際にありますが、ここに繋がる知見を得られるかも知れません。


