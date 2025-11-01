# BitSpec - AI-based Mass Spectrum Prediction for GC-MS

化学構造からマススペクトルを予測する深層学習システム。Graph Convolutional Network (GCN)を使用して、MOLファイルやSMILES文字列から質量スペクトルを生成します。

## 特徴

- **Graph Convolutional Network**: 分子グラフの構造情報を効果的に学習
- **MOL/MSP対応**: MOLファイルとNIST MSP形式の完全サポート
- **RTX 50シリーズ対応**: 最新のNVIDIA RTX 50シリーズGPU (sm_120) に最適化
- **Mixed Precision Training**: FP16混合精度訓練による高速化とメモリ効率化
- **複合損失関数**: MSE、コサイン類似度、KLダイバージェンスを組み合わせた損失
- **Attention Pooling**: 重要な分子部分構造に注目する機構
- **Dev Container対応**: 環境構築を簡素化するDev Container設定

## システム要件

- Python 3.10+
- CUDA 12.8+
- PyTorch 2.7.0+
- RTX 50シリーズGPU (推奨) またはその他のCUDA対応GPU
- Docker (Dev Container使用時)

## インストール

### 方法1: Dev Container (推奨)

```bash
# Visual Studio Codeで開く
# F1 → "Remote-Containers: Reopen in Container"
# 全ての依存関係が自動的にインストールされます
```

詳細は [DEV_CONTAINER_GUIDE.md](DEV_CONTAINER_GUIDE.md) を参照してください。

### 方法2: ローカルインストール

```bash
# リポジトリのクローン
git clone https://github.com/turnDeep/BitSpec.git
cd BitSpec

# PyTorch 2.7.0+ (CUDA 12.8対応)
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch>=2.7.0

# 依存関係のインストール
pip install -r requirements.txt

# パッケージのインストール
pip install -e .
```

## プロジェクト構造

```
BitSpec/
├── config.yaml              # 設定ファイル
├── requirements.txt         # 依存関係
├── setup.py                # パッケージ設定
├── README.md               # このファイル
├── DEV_CONTAINER_GUIDE.md  # Dev Containerガイド
├── .devcontainer/          # Dev Container設定
├── data/
│   ├── NIST17.MSP          # NIST MSPファイル
│   ├── mol_files/          # MOLファイルディレクトリ
│   └── processed/          # 前処理済みデータ
├── checkpoints/            # モデルチェックポイント
├── src/
│   ├── data/              # データ処理
│   │   ├── mol_parser.py  # MOL/MSPパーサー
│   │   ├── features.py    # 分子特徴量抽出
│   │   ├── dataset.py     # データセット
│   │   └── dataloader.py  # データローダー
│   ├── models/            # モデル定義
│   │   └── gcn_model.py   # GCNモデル
│   ├── training/          # トレーニング
│   │   └── loss.py        # 損失関数
│   └── utils/             # ユーティリティ
│       ├── metrics.py     # 評価メトリクス
│       └── rtx50_compat.py # RTX 50互換性
└── scripts/
    ├── preprocess_data.py      # データ前処理
    ├── train.py                # トレーニング
    ├── predict.py              # 推論
    ├── test_training.py        # テストトレーニング
    └── test_data_loading.py    # データ読み込みテスト
```

## クイックスタート

### 1. テストトレーニング (10サンプル)

環境が正しくセットアップされているか確認:

```bash
python scripts/test_training.py
```

このスクリプトは10個のMOLファイルを使用して小規模なトレーニングを実行し、全ての機能が動作することを確認します。

### 2. データの準備

NIST MSP形式のデータとMOLファイルを準備:

```
data/
├── NIST17.MSP           # マススペクトルデータ
└── mol_files/           # 対応するMOLファイル
    ├── ID200001.MOL
    ├── ID200002.MOL
    └── ...
```

データの前処理 (オプション):

```bash
python scripts/preprocess_data.py \
    --input data/NIST17.MSP \
    --output data/processed \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

### 3. トレーニング

```bash
python scripts/train.py --config config.yaml
```

設定のカスタマイズは `config.yaml` で行えます:

```yaml
model:
  node_features: 157      # 原子特徴量の次元
  edge_features: 16       # 結合特徴量の次元
  hidden_dim: 256         # 隠れ層の次元
  num_layers: 5           # GCN層の数
  dropout: 0.1

training:
  batch_size: 32
  num_epochs: 200
  learning_rate: 0.001
  use_amp: true           # Mixed Precision Training

gpu:
  use_cuda: true
  mixed_precision: true
  compile: true           # torch.compile使用
  rtx50:
    enable_compat: true   # RTX 50対応を有効化
```

### 4. 予測

#### 単一分子の予測 (SMILES)

```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --output prediction.png
```

#### MOLファイルからの予測

```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --mol_file data/mol_files/ID200001.MOL \
    --output prediction.png
```

#### バッチ予測

```bash
# smiles.txtに各行1つのSMILES文字列を記載
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --batch_file smiles.txt \
    --output batch_predictions/
```

#### MSP形式でエクスポート

```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --export_msp
```

## Pythonスクリプトでの使用

```python
from src.models.gcn_model import GCNMassSpecPredictor
from scripts.predict import MassSpectrumPredictor

# 予測器の初期化
predictor = MassSpectrumPredictor(
    checkpoint_path='checkpoints/best_model.pt',
    config_path='config.yaml'
)

# SMILES文字列から予測
spectrum = predictor.predict_from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O')

# 有意なピークの検出
peaks = predictor.find_significant_peaks(spectrum, threshold=0.05, top_n=20)
print(f"Top 10 peaks: {peaks[:10]}")

# 可視化
predictor.visualize_prediction(
    smiles='CC(=O)OC1=CC=CC=C1C(=O)O',
    save_path='aspirin_spectrum.png'
)

# MSP形式でエクスポート
predictor.export_to_msp(
    smiles='CC(=O)OC1=CC=CC=C1C(=O)O',
    output_path='aspirin.msp',
    compound_name='Aspirin'
)
```

## モデルアーキテクチャ

```
Input (MOL/SMILES) → Molecular Graph → GCN Layers → Attention Pooling → MLP → Mass Spectrum
                                          ↓
                                   Feature Extraction
                                   (Residual + BatchNorm)
```

### 詳細

- **入力**: 分子グラフ
  - ノード特徴量: 157次元 (原子番号、次数、形式電荷、キラリティ、水素数、混成軌道、芳香族性、環情報)
  - エッジ特徴量: 16次元 (結合タイプ、立体化学、共役、環情報)
- **GCN層**: 5層のGraph Convolutional層
  - 各層にResidual接続とBatch Normalizationを適用
  - 活性化関数: ReLU
  - ドロップアウト: 0.1
- **Attention Pooling**: グラフレベル表現の生成
- **出力層**:
  - Multi-Layer Perceptron (MLP)
  - 出力: 1000次元のスペクトル (m/z 0-999)
  - 活性化関数: Sigmoid (強度を0-1に正規化)

## 損失関数

複合損失関数を使用:

```python
Loss = α·MSE + β·(1 - CosineSimilarity) + γ·KL_Divergence
```

- **MSE損失** (α=1.0): ピーク強度の二乗誤差
- **コサイン類似度損失** (β=1.0): スペクトル形状の類似性
- **KLダイバージェンス損失** (γ=0.1): 分布の類似性

重みは `config.yaml` で調整可能。

## 評価メトリクス

- **Cosine Similarity**: スペクトル形状の類似度 (主要指標)
- **Pearson Correlation**: ピーク強度の相関係数
- **MSE/MAE**: 予測誤差
- **Top-K Accuracy**: 主要ピークの一致率

## RTX 50シリーズ対応

このプロジェクトはNVIDIA RTX 50シリーズGPU (Blackwell, sm_120) に完全対応しています:

- **PyTorch 2.7.0+**: sm_120アーキテクチャの公式サポート
- **CUDA 12.8+**: 最新CUDA Toolkitによる最適化
- **Mixed Precision Training**: FP16による高速化
- **torch.compile**: JITコンパイルによる更なる高速化
- **互換性レイヤー**: 必要に応じてsm_90エミュレーション

詳細は `src/utils/rtx50_compat.py` を参照。

## データ形式

### NIST MSP形式

```
Name: Aspirin
InChIKey: BSYNRYMUTXBXSQ-UHFFFAOYSA-N
Formula: C9H8O4
MW: 180
ID: 200001
Num peaks: 15
41 100.0
55 50.0
69 25.0
...
180 999.0

```

### MOLファイル

標準のMOL V2000/V3000形式に対応。`data/mol_files/` ディレクトリにID付きで配置:

```
data/mol_files/
├── ID200001.MOL
├── ID200002.MOL
└── ...
```

MSPファイルのIDとMOLファイル名のIDが対応している必要があります。

## トラブルシューティング

### GPU が認識されない

```bash
# CUDAの確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# RTX 50対応の確認
python -c "from src.utils.rtx50_compat import setup_gpu_environment; setup_gpu_environment()"
```

### メモリ不足エラー

`config.yaml` でバッチサイズを調整:

```yaml
training:
  batch_size: 16  # 32から16に減らす
```

または `hidden_dim` を減らす:

```yaml
model:
  hidden_dim: 128  # 256から128に減らす
```

### データが見つからない

```bash
# データの配置を確認
ls -la data/NIST17.MSP
ls -la data/mol_files/ | head

# MOLファイルとMSPのIDマッピングを確認
python scripts/test_mol_nist_mapping.py
```

## コンソールスクリプト

パッケージインストール後、以下のコマンドが使用可能:

```bash
ms-train --config config.yaml         # トレーニング
ms-predict --checkpoint model.pt ...  # 予測
ms-evaluate --checkpoint model.pt ... # 評価
```

## 開発ツール

```bash
# コードフォーマット
black src/ scripts/

# 型チェック
mypy src/

# テスト実行
pytest
```

## 参考文献

- **NEIMS**: Neural EI-MS Prediction for Unknown Compound Identification
- **ICEBERG/SCARF**: MIT Mass Spectrum Prediction
- **Massformer**: Graph Transformer for Small Molecule Mass Spectra Prediction

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずIssueを開いて変更内容を議論してください。

## お問い合わせ

- **GitHub Issues**: https://github.com/turnDeep/BitSpec/issues
- **プロジェクトURL**: https://github.com/turnDeep/BitSpec

## 更新履歴

- **v1.0.0** (2024): 初回リリース
  - GCNベースのマススペクトル予測モデル
  - RTX 50シリーズ対応
  - MOL/MSP完全サポート
  - Dev Container対応
