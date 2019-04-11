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

# 二項分布（Binomial distribution）は離散分布の一種です
# まずは例題
#
# プレイヤーAとプレイヤーBの2人が、バスケットボールをします。Aは1ゲームで平均11回シュートをして、平均的な成功率は72％です。一方、Bは15回シュートをしますが、平均的に48%しか決まりません。
#
# 問1: プレイヤーAが平均的な1試合で6回シュートを決める確率は？
#
# 問2: おなじく、プレイヤーBが1試合で6回シュートを決める確率は？
#
# 以下の条件が満たされれば、この問題を二項分布を使って考える事ができます。
#
# 1.) 全体がn回の連続した試行からなる
# 2.) それぞれの試行は、相互に排他的な2つの事象からなる（例えば成功と失敗）
# 3.) 成功の確率がpならば、失敗の確率は1-p
# 4.) それぞれの試行は独立
# 二項分布の確率質量関数は、以下のようになります。
#
# Pr(X=k)=C(n,k)pk(1−p)n−k
#  
# nは試行の回数、kは成功の数、pは成功の確率、1-pは失敗の確率ですが、しばしばqと書かれます。
#
# n回試行して、k回成功する確率は、
# pk
#  
# また、n-k回失敗する確率は
# (1−p)n−k
#  
# n回の試行で、k回の成功がどこにくるかわかりませんが、この並べ方は
# C(n,k)
#  
# 通りあります。 これらをすべて掛け合わせれば、n回中k回成功する確率が求まるわけです。
#
# C(n,k) は組み合わせです。実際の計算は次のような式で表現できます。
#
# C(n,k)=n!k!(n−k)!
#  
# 例題を解いてみましょう¶

# +
p_A = .72
n_A = 11

k=6

#import scipy.misc as sc
import scipy.special as sc

comb_A = sc.comb(n_A,k)

answer_A = comb_A * (p_A**k)*((1-p_A)**(n_A-k))

answer_A = 100 * answer_A

p_B = .48
n_B = 15
comb_B = sc.comb(n_B,k)
answer_B = 100 * comb_B * (p_B**k)*((1-p_B)**(n_B-k))

print('Aが平均的な試合で６回シュートを決める確率は{:0.2f}%'.format(answer_A))
print('')
print('Bが平均的な試合で６回シュートを決める確率は{:0.2f}%'.format(answer_B))

# +
#9回
k=9

comb_A = sc.comb(n_A,k)
comb_B = sc.comb(n_B,k)

answer_A = 100 * comb_A*(p_A**k)*((1-p_A)**(n_A-k))
answer_B = 100 * comb_B*(p_B**k)*((1-p_B)**(n_B-k))

print('Aが平均的な試合で６回シュートを決める確率は{:0.2f}%'.format(answer_A))
print('')
print('Bが平均的な試合で６回シュートを決める確率は{:0.2f}%'.format(answer_B))

# +
# 平均値です。
mu_A = n_A *p_A
mu_B = n_B *p_B

# 標準偏差を計算しましょう。
sigma_A = ( n_A *p_A*(1-p_A) )**0.5
sigma_B = ( n_B *p_B*(1-p_B) )**0.5

print('プレイヤーAは1試合で、平均{:0.1f}回±{:0.1f}シュートを決めます。'.format(mu_A,sigma_A))
print('\n')
print('プレイヤーBは1試合で、平均{:0.1f}回±{:0.1f}シュートを決めます。'.format(mu_B,sigma_B))

# +
from scipy.stats import binom

mean,var= binom.stats(n_A,p_A)

print(mean)
print(var**0.5)

# +
import numpy as np

# 10回と、表の確率0.5をセットします。
n=10
p=0.5

x = range(n+1)

# 二項分布の確率質量関数をから、実際の確率を計算できます。
Y = binom.pmf(x,n,p)

Y

# +
import matplotlib.pyplot as plt
# %matplotlib inline

# プロットします。
plt.plot(x,Y,'o')

# y=1.08はタイトルが少し上に行くようにするためです。
plt.title('Binomial Distribution PMF: 10 coin Flips, Odds of Success for Heads is p=0.5',y=1.08)

#軸にもタイトルが付けられます。
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
# -


