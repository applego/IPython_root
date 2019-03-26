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

#
# # 機械学習入門
#
# このセクションは、SciKit Learn を使った実践的な機械学習の入門コースです。
#
# インストールが済んでいない場合は、以下のいずれかの方法でインストールしておいてください。
#
# pip install scikit-learn
#
# conda install scikit-learn
#
# ## このコースで学べること
# 機械学習アルゴリズムには、数学的な背景があります。各レクチャーは、できるだけ分かり易くこの背景を説明したあと、実際にPythonのコードを実行する流れになります。
#
# 機械学習に関する話題は広いので、すべてを網羅することは出来ませんが、レクチャーを終えれば、いろいろな方法論についての知識が付きます。
#
# ## 機械学習に関する資料
# 1. ) SciKit Learnのドキュメントは、英語になってしまいますが秀逸です。
#
# SciKit Learn Tutorial
#
# 2. ) 数少ない日本語の入門コースです 技術評論社のページ
#
# 3. ) sasという会社さんのページですが、よくまとまっています。機械学習
#
# 4. ) 英語になってしまいますが、Andrew Ng先生の講義です。
#
# notes
#
# Coursera Video
#
# Pythonやscikit learnは使っていませんが、数学的な背景についてはよい入門になっています。

# # 学習とテスト
#
# - 学習（Training)
#     - 説明変数を使って、目的変数をうまく予測できるモデルを作る
# - テスト（Test）
#     - 説明変数だけを使って、目的変数を予測
# - 分類（Classification)
#     - 目的変数が離散的（クラス１、クラス２、・・・）
# - 回帰（Regression)
#     - 目的変数が連続的な値

# # 学習の種類
# - 教師あり学習（Supervised learning)
#     - 目的変数を学習に利用してモデルを作る
# - 教師なし学習（Unsupervised learning)
#     - 説明変数だけを使って、サンプルの分類などをする


