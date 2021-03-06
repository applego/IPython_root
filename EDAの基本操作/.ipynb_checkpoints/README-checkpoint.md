https://www.codexa.net/basic-exploratory-data-analysis-with-python/

# 【データサイエンティスト入門編】探索的データ解析（EDA）の基礎操作をPythonを使ってやってみよう
~Codexa~

- 探索的データ解析
- Explanaotry Data Analysis(EDA)

データの特徴を探求し、構造を理解することを目的としたデータサイエンスの最初の一歩です。

まずはデータに触れてみて、データを視覚化したり、データのパターンを探したり、特徴やターゲットの県警性/相関性を感じとるのが目的

問題を解決する前に、どのようなデータセットを扱っているのか、どのような状況にあるのかを、しっかりと理解するのが重要であり、「単セク的データ解析（EDA)」はまさしくそれを目的とした作業となります。

今回のチュートリアルでは、データサイエンティスト入門として、探索的データ解析で頻繁に使われる基本的な関数などを紹介させていただきます。利用するデータセットは、機械学習入門者であれば一度は目にしたことがる「アヤメ（Ires Dataset)」のデータセットを使いましょう。

### なぜ探索的データ解析が重要なのか？
データに対して「仮設」を立てて、最終的に*予測モデルを構築する*のですが、そのプロセスにおいて「探索的データ解析」が重要になる。

有名なドイツの哲学者「アルトゥル・ショーペンハウアー」の引用ですが、「金を探し求めている錬金術師たちは、金よりも価値の高い多くのものを発見しました」と残していますが、まさに探索的データ解析はデータサイエンティストによって、データをより深く理解して「データよりもかちの高いもの」を見つけるための作業なのです。

### 探索的データ解析で使うツールとは？
1. Pandas
2. Matplotlib
3. Numpy

### 「アヤメデータセット」
機械学習の初学者が分類問題を練習する際に広く使われる、とても有名なデータセット。

#### 特徴量（説明変数）
- Sepal Length：がく片の長さ(cm)
- Sepal Eidth:がく片の幅(cm)
- Petal Length:花弁の長さ(cm)
- Petal Eidth:花弁の幅(cm)

アヤメには下記の３種類あり、上記の特徴量からどの種類のアヤメに属するかを分類予測する。
- Iris-Setosa
- Iris-Versicolor
- Ires-Virginica

各50データずつ

### 探索的データ解析チュートリアル
以下はipynbで


