# total-perspective-vortex

42 の課題：脳コンピュータインターフェース（BCI）。運動イメージの脳波（EEG）から、
手を動かしているのか足なのかを、MNE と scikit-learn で当てる。

## ファイルの置き場所

| パス | 中身 |
|---|---|
| `mybci.py` | 入口（`train`、`predict`、引数なしで全被験者の評価） |
| `src/visualize.py` | 生の脳波とフィルタ後の脳波の図 |
| `src/tpv/data.py` | physionet のデータを MNE で読み込む |
| `src/tpv/preprocessing.py` | フィルタと特徴の取り出し |
| `src/tpv/csp.py` | 自分で書く次元削減（sklearn の transformer） |
| `src/tpv/pipeline.py` | sklearn の `Pipeline` |
| `src/tpv/stream.py` | 録画を流れるデータとして再生する |

## 使い方

```
make install
make visualize SUBJECT=4 RUN=14
make train SUBJECT=4 RUN=14
make predict SUBJECT=4 RUN=14
make eval
```

`make help` で一覧が出る。

## データ（PhysioNet EEG Motor Movement/Imagery）

`src/tpv/data.py` の `eegbci.load_data(subject, runs)`：

- `subject: int`：被験者。1〜109。
- `runs: list[int]`：記録1回分（run）の番号。1〜14。EDF ファイルのパスのリストを返す。

1人の被験者が 14 回記録している。最初の2回は1分のベースライン、そのあとは
2分のタスク4種類を3回ずつ。

| run | 課題 |
|---|---|
| 1 | ベースライン（目を開けて休む） |
| 2 | ベースライン（目を閉じて休む） |
| 3, 7, 11 | 左拳か右拳を動かす |
| 4, 8, 12 | 左拳か右拳を動かすのを想像する |
| 5, 9, 13 | 両拳か両足を動かす |
| 6, 10, 14 | 両拳か両足を動かすのを想像する |

イベントの印（アノテーション）の意味は run の種類で変わるので、run の番号を見て
ラベルに直す。

| 印 | run 3, 4, 7, 8, 11, 12 | run 5, 6, 9, 10, 13, 14 |
|---|---|---|
| T0 | 休み | 休み |
| T1 | 左拳 | 両拳 |
| T2 | 右拳 | 両足 |

課題は「手か足か」なので、使う run は 5, 6, 9, 10, 13, 14（想像のほうは 6, 10, 14）。
サンプリング周波数 160 Hz、64 チャンネル。

run の構成とイベントの印の意味は PhysioNet のページで確認した。run と課題の対応表は
記憶で書いた。使う前に PhysioNet / MNE のドキュメントで確かめること。

## 参考

[scikit-learn pipelineによる前処理を概観する](https://qiita.com/suikabar/items/88d1fb702ed289a5a3ee)

[MNE-Python の使い方 運動想起の解析](https://qiita.com/ponpopon_pon/items/32e931058841acc7dee5)

[Pythonで脳波解析：Python MNEのチュートリアル](https://qiita.com/sentencebird/items/035ba0c48569f06e3a42)

[PythonでEDFファイルを読み込む方法](https://qiita.com/tchkwkzk/items/0e4e6cb1e29436c843c9)

[Overview of MEG/EEG analysis with MNE-Python](https://mne.tools/stable/auto_tutorials/intro/10_overview.html)
