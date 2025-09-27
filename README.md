# BitSpec-GCN: 化学構造からのマススペクトル予測

## 概要

このリポジトリは、化学構造（MOLファイル）から電子イオン化マススペクトル（EI-MS）を予測するための機械学習プロジェクト「BitSpec-GCN」の実装です。

Graph Convolutional Network (GCN) を用いて分子のグラフ構造から特徴を抽出し、BitNet技術に基づく1.58ビット量子化を適用したデコーダでマススペクトルを生成します。これにより、軽量かつ高速な予測モデルの実現を目指します。

この実装は、提供された[要件定義書](requirements-doc.md)、[設計書](design-doc.md)、[仕様書](specification-doc.md)に完全に基づいています。

## 主な技術

- **フレームワーク**: PyTorch, PyTorch Geometric
- **化学情報学**: RDKit
- **モデルアーキテクチャ**:
    - **エンコーダ**: Graph Convolutional Network (GCN)
    - **デコーダ**: BitNet (カスタムの1.58ビット量子化 `BitLinear` レイヤー)
- **損失関数**: 重み付きコサイン類似度損失 (Weighted Cosine Loss)

## プロジェクト構造

```
.
├── config.yaml            # プロジェクト全体の設定ファイル
├── data/                    # データ処理関連のモジュール
│   ├── msp_parser.py      # MSPファイルパーサー
│   ├── mol_loader.py      # MOLファイルローダー
│   └── preprocessor.py    # グラフデータへの前処理
├── evaluation/            # 評価指標と可視化ツール
│   ├── metrics.py         # 評価指標の計算
│   └── visualizer.py      # スペクトル比較プロットの生成
├── main.py                # 訓練・評価パイプラインを実行するメインスクリプト
├── models/                # モデルアーキテクチャ
│   ├── bitlinear.py       # カスタム量子化レイヤー
│   ├── gcn_encoder.py     # GCNエンコーダ
│   └── bitnet_decoder.py  # BitNetデコーダ
├── training/              # 訓練関連のモジュール
│   ├── loss.py            # カスタム損失関数
│   └── trainer.py         # 訓練ループ管理クラス
├── mol_files/             # 入力データ (MOLファイル)
├── NIST17.MSP             # 入力データ (MSPファイル)
├── requirements.txt       # Pythonの依存関係リスト
└── README.md              # このファイル
```

## 使い方

### 1. セットアップ

まず、リポジトリをクローンし、必要な依存関係をインストールします。

```bash
# 依存関係のインストール
pip install -r requirements.txt

# PyTorch Geometricの関連ライブラリをインストール
# (環境に応じてCUDAバージョンを調整してください: +cu121, +cu118, or +cpu)
pip install torch_geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```
**注意**: このプロジェクトは `numpy<2.0` に依存しています。`requirements.txt` のバージョン指定をご確認ください。

### 2. 設定の確認

`config.yaml` ファイルで、訓練エポック数、バッチサイズ、学習率、ファイルパスなどの設定を確認・調整できます。

```yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

data:
  msp_path: "NIST17.MSP"
  mol_dir: "mol_files"
  cache_dir: "cache"
```

### 3. 訓練と評価の実行

以下のコマンドで、データの前処理、モデルの訓練、およびテストデータセットでの評価まで、すべてのパイプラインを実行します。

```bash
python main.py
```

- **前処理**: 初回実行時には、データが前処理され、結果が `cache/` ディレクトリに保存されます。次回以降の実行は、このキャッシュを読み込むため高速になります。
- **訓練**: 訓練中、最良のモデルのチェックポイントが `results/checkpoints/` に保存されます。
- **評価**: 訓練完了後、保存された最良モデルを使ってテストデータセットでの評価が自動的に行われます。評価指標が出力され、予測スペクトルの比較プロットが `results/evaluation_plots/` に保存されます。

## 評価結果について

現在の実装と提供されたデータセット（訓練データ80件）で100エポックの訓練を行った結果、テストデータセットにおける**平均コサイン類似度**は `0.1045` となりました。

これは、要件定義書にある成功基準（0.85）には達していませんが、ソフトウェアとしてのパイプラインは完全に機能しています。モデルの精度を向上させるには、より大規模なデータセットでの訓練や、ハイパーパラメータのさらなるチューニングが今後の課題となります。