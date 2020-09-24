# ApproximateSS
ベクトル近似信念伝搬法(VAMP)に基づく近似的stability selection法のJulia実装

# quick start
stability selectionをデフォルトパラメータで実行するためには
```
rvamp(A, y, λ, family, covariance_type)
```
`family`の部分は、線形回帰とロジスティック回帰に応じて`Normal()`か`Binomial()`で指定する。

`covariance_type`はVAMPで使う共分散行列の構造を、`ApproximateSS.Diagonal()`か`ApproximateSS.DiagonalRestricted()`から指定する。前者の場合は、サイトごとに異なる二次モーメントの共役変数を指定する方法で、後者の場合はサイトごとに一定の二次モーメントの共役変数を指定する方法になっている(self-averaging rVAMP)。

ダンピング係数`dumping`、反復回数`t_max`、収束基準`tol`、stability selectionのパラメータ`pw, w`、切片の有無`intercept`を指定したければ
```
rvamp(
        A, y, λ, family, covariance_type, 
        dumping=0.8, t_max=50, tol=1.0e-6, pw=0.5, w=2.0, intercept=false
    )
```
という具合。

# メモ
* 一回目の実行はコンパイルが走るので遅い
* 手元のmac miniだと、データ数100-300, パラメータ数10000くらいまでは共分散の構造によらず数秒で終わる
* 線形回帰とロジスティック回帰のみ実装済
* intercept は Diagonal版だけ。（自己平均版には実装してない）
* 観測数がパラメータ数より少ない場合にしか動きません (n < p)

# see also
* [Semi-analytic approximate stability selection for correlated data in generalized linear models](https://iopscience.iop.org/article/10.1088/1742-5468/ababff/meta)
    - この手法の導出を説明している論文。実データへの適用例もある。
