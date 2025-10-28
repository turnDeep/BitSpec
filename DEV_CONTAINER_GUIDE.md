# Dev Container で BitSpec を実行する完全ガイド

## 概要

このガイドでは、Dev Container環境でBitSpec（マススペクトル予測プロジェクト）を実行し、10個のMOLファイルを使ってモデル作成を検証する方法を説明します。

## 作成されたファイル

### 1. Dev Container設定ファイル

```
.devcontainer/
├── Dockerfile              # Docker イメージの定義（CUDA 12.8, PyTorch, RDKit等）
├── devcontainer.json       # VS Code Dev Container設定
└── README.md              # Dev Container詳細ガイド
```

### 2. トレーニングスクリプト

```
scripts/
├── test_training.py        # 10個のMOLファイルでのテストトレーニング（PyTorch必要）
└── test_data_loading.py    # データ読み込みテスト（簡易版）
```

## クイックスタート（Dev Container使用）

### 前提条件

- Docker Desktop または Docker Engine (20.10以上)
- Visual Studio Code
- VS Code拡張: Remote - Containers
- （オプション）NVIDIA Docker（GPUを使用する場合）

### ステップ 1: Dev Containerを開く

1. VS CodeでBitSpecプロジェクトを開く
2. `F1` → "Remote-Containers: Reopen in Container" を選択
3. 初回ビルドには10-20分程度かかります

### ステップ 2: 依存関係の確認

コンテナ内のターミナルで：

```bash
# PyTorchが正しくインストールされているか確認
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# RDKitの確認
python3 -c "import rdkit; print('RDKit: OK')"

# プロジェクトのインストール
pip3 install -e .
```

### ステップ 3: 10個のMOLファイルでテストトレーニング

```bash
# テストトレーニングを実行
python3 scripts/test_training.py
```

## テストトレーニングの内容

`test_training.py` スクリプトは以下を実行します：

### 1. データセットの準備

- **入力**: `data/NIST17.MSP` と `data/mol_files/` ディレクトリ
- **処理**: 最初の10個のMOLファイルとそれに対応するマススペクトルデータを読み込み
- **分割**: Train 8サンプル / Val 1サンプル / Test 1サンプル

### 2. モデルの構築

```python
model = GCNMassSpecPredictor(
    node_features=155,      # 原子特徴量の次元
    edge_features=15,       # 結合特徴量の次元
    hidden_dim=128,         # 隠れ層の次元（テスト用に小さく設定）
    num_layers=3,           # GCN層の数（テスト用に少なく設定）
    spectrum_dim=1000,      # 出力スペクトルの次元
    dropout=0.1
)
```

### 3. トレーニング

- **エポック数**: 5（テスト用に短く設定）
- **バッチサイズ**: 2
- **損失関数**: Combined Loss (MSE + Cosine Similarity)
- **オプティマイザ**: Adam (lr=0.001)

### 4. 評価

各エポックで以下のメトリクスを計算：
- 損失値 (Combined Loss)
- コサイン類似度
- （内部的に）MSE、MAE等

### 5. モデルの保存

訓練完了後、モデルを以下に保存：
```
checkpoints/test/test_model.pt
```

## 期待される出力

```
============================================================
10個のMOLファイルを使用したテストトレーニング
============================================================
使用デバイス: cuda:0
CUDA デバイス: NVIDIA RTX 5090
CUDA バージョン: 12.8

1. データセットの作成
------------------------------------------------------------
解析された化合物数: 900
全データ数: 900 サンプル
テスト用サンプル数: 10
  訓練データ: 8
  検証データ: 1
  テストデータ: 1
  訓練バッチ数: 4
  検証バッチ数: 1

2. モデルの作成
------------------------------------------------------------
  ノード特徴量次元: 155
  エッジ特徴量次元: 15
  総パラメータ数: 1,234,567
  訓練可能パラメータ数: 1,234,567

3. 訓練設定
------------------------------------------------------------
  損失関数: SpectrumLoss (combined)
  オプティマイザ: Adam
  学習率: 0.001

4. トレーニング開始
------------------------------------------------------------
Epoch 1/5 [Train]: 100%|███████| 4/4 [00:05<00:00, loss=0.8234, cos_sim=0.1234]
Epoch 1/5 [Val]:   100%|███████| 1/1 [00:01<00:00, loss=0.7890, cos_sim=0.1456]

Epoch 1 結果:
  訓練損失: 0.8234
  訓練コサイン類似度: 0.1234
  検証損失: 0.7890
  検証コサイン類似度: 0.1456

...（Epoch 2-5も同様）

5. モデルの保存
------------------------------------------------------------
モデル保存: checkpoints/test/test_model.pt

============================================================
テストトレーニング完了！
============================================================
最終訓練損失: 0.5123
最終訓練コサイン類似度: 0.4567
最終検証損失: 0.5234
最終検証コサイン類似度: 0.4321

✓ テストトレーニングが正常に完了しました
```

## データの確認

### MOLファイルの確認

```bash
# MOLファイルの数を確認
ls data/mol_files/ | wc -l
# 出力: 900

# 最初の10個を確認
ls data/mol_files/ | head -10
# 出力:
# ID200001.MOL
# ID200002.MOL
# ...
# ID200010.MOL
```

### MSPファイルの確認

```bash
# MSPファイルのサイズ確認
ls -lh data/NIST17.MSP

# MSPファイルの内容をサンプル表示
head -50 data/NIST17.MSP
```

## トラブルシューティング

### 1. Docker ビルドエラー

**症状**: Dev Containerのビルドが失敗する

**解決策**:
```bash
# Dockerのキャッシュをクリア
docker system prune -a

# VS Code で再ビルド
# F1 → "Remote-Containers: Rebuild Container"
```

### 2. GPU が認識されない

**症状**: `torch.cuda.is_available()` が False を返す

**解決策**:
```bash
# ホスト側でNVIDIA Dockerが動作するか確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# nvidia-smiが動作しない場合、NVIDIA Container Toolkitを再インストール
```

### 3. メモリ不足エラー

**症状**: トレーニング中にOOMエラー

**解決策**:
- Docker Desktopのメモリ設定を増やす（8GB以上推奨）
- `test_training.py` の `batch_size` を1に減らす
- `hidden_dim` を64に減らす

### 4. データが見つからない

**症状**: `MSPファイルが見つかりません` エラー

**解決策**:
```bash
# データの配置を確認
ls -la data/
ls -la data/mol_files/

# データが無い場合は、前処理を実行
python3 scripts/preprocess_data.py
```

## 次のステップ

### 1. 全データでのトレーニング

10個のサンプルでの検証が成功したら、全データでトレーニング：

```bash
# config.yamlを編集してパラメータを調整
# その後、完全なトレーニングを実行
python3 scripts/train.py --config config.yaml
```

### 2. ハイパーパラメータのチューニング

`config.yaml` で以下を調整：
- `hidden_dim`: 128 → 256
- `num_layers`: 3 → 5
- `batch_size`: 2 → 32
- `num_epochs`: 5 → 200

### 3. モデルの評価

```bash
# テストデータで評価
python3 scripts/evaluate.py \
    --checkpoint checkpoints/test/test_model.pt \
    --config config.yaml
```

### 4. 予測の実行

```bash
# 単一分子の予測
python3 scripts/predict.py \
    --checkpoint checkpoints/test/test_model.pt \
    --config config.yaml \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --output prediction.png
```

## ファイル構成

```
BitSpec/
├── .devcontainer/
│   ├── Dockerfile                  # Docker環境定義
│   ├── devcontainer.json          # VS Code設定
│   └── README.md                  # 詳細ガイド
├── scripts/
│   ├── test_training.py           # テストトレーニング（10サンプル）
│   ├── test_data_loading.py       # データ読み込みテスト
│   ├── train.py                   # 本番トレーニング
│   ├── predict.py                 # 予測
│   └── preprocess_data.py         # データ前処理
├── src/
│   ├── data/                      # データ処理
│   │   ├── mol_parser.py          # MOL/MSPパーサー
│   │   ├── features.py            # 特徴量抽出
│   │   ├── dataset.py             # データセット
│   │   └── dataloader.py          # データローダー
│   ├── models/                    # モデル定義
│   │   └── gcn_model.py           # GCNモデル
│   ├── training/                  # トレーニング
│   │   └── loss.py                # 損失関数
│   └── utils/                     # ユーティリティ
│       ├── metrics.py             # 評価指標
│       └── rtx50_compat.py        # RTX 50互換性
├── data/
│   ├── NIST17.MSP                 # MSPファイル
│   └── mol_files/                 # MOLファイル（900個）
├── config.yaml                    # 設定ファイル
├── requirements.txt               # 依存関係
└── README.md                      # プロジェクトREADME
```

## まとめ

このガイドに従えば、Dev Container環境でBitSpecプロジェクトを実行し、10個のMOLファイルを使ってモデル作成を検証できます。

**重要なポイント**:
1. Dev Containerを使用することで、環境構築の問題を回避
2. テストトレーニングで動作を確認してから本番実行
3. GPU環境では高速なトレーニングが可能

**参考リンク**:
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
