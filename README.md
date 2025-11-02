# BitSpec - AI-based Mass Spectrum Prediction for GC-MS

化学構造からマススペクトルを予測する深層学習システム。Graph Convolutional Network (GCN)を使用して、MOLファイルやSMILES文字列から質量スペクトルを生成します。

## 特徴

- **Graph Convolutional Network**: 分子グラフの構造情報を効果的に学習
- **MOL/MSP対応**: MOLファイルとNIST MSP形式の完全サポート
- **RTX 50シリーズ対応**: 最新のNVIDIA RTX 50シリーズGPU (sm_120) に最適化
- **Mixed Precision Training**: FP16混合精度訓練による高速化とメモリ効率化
- **Modified Cosine Loss**: Neutral lossを考慮したコサイン類似度損失関数
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
├── config_pretrain.yaml     # 事前学習設定ファイル
├── requirements.txt         # 依存関係
├── setup.py                # パッケージ設定
├── README.md               # このファイル
├── DEV_CONTAINER_GUIDE.md  # Dev Containerガイド
├── PRETRAINING_PROPOSAL.md # 事前学習提案
├── TECHNICAL_SUMMARY.md    # 技術サマリー
├── QUICK_REFERENCE.md      # クイックリファレンス
├── ANALYSIS_INDEX.md       # 分析インデックス
├── ANALYSIS_PRETRAINING_INFRASTRUCTURE.md # 事前学習インフラ分析
├── .devcontainer/          # Dev Container設定
├── data/
│   ├── NIST17.MSP          # NIST MSPファイル
│   ├── mol_files/          # MOLファイルディレクトリ
│   └── processed/          # 前処理済みデータ
├── checkpoints/            # モデルチェックポイント
│   ├── pretrain/          # 事前学習モデル
│   └── finetune/          # ファインチューニングモデル
├── src/
│   ├── data/              # データ処理
│   │   ├── mol_parser.py  # MOL/MSPパーサー
│   │   ├── features.py    # 分子特徴量抽出
│   │   ├── dataset.py     # データセット
│   │   ├── dataloader.py  # データローダー
│   │   └── pcqm4mv2_loader.py # PCQM4Mv2データローダー
│   ├── models/            # モデル定義
│   │   └── gcn_model.py   # GCNモデル + 事前学習ヘッド
│   ├── training/          # トレーニング
│   │   └── loss.py        # 損失関数
│   └── utils/             # ユーティリティ
│       ├── metrics.py     # 評価メトリクス
│       └── rtx50_compat.py # RTX 50互換性
└── scripts/
    ├── preprocess_data.py      # データ前処理
    ├── train.py                # トレーニング
    ├── pretrain.py             # PCQM4Mv2事前学習
    ├── finetune.py             # ファインチューニング
    ├── predict.py              # 推論
    ├── test_training.py        # テストトレーニング
    ├── test_data_loading.py    # データ読み込みテスト
    └── test_mol_nist_mapping.py # MOL-NISTマッピングテスト
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
  node_features: 48       # 原子特徴量の次元（実装最適化済み）
  edge_features: 6        # 結合特徴量の次元（実装最適化済み）
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

## PCQM4Mv2事前学習とファインチューニング

BitSpecは、大規模分子データセット（PCQM4Mv2）での事前学習をサポートしています。量子化学的性質（HOMO-LUMO gap）を学習することで、より少ないEI-MSデータでも高性能なモデルを構築できます。

### 1. PCQM4Mv2での事前学習

```bash
# PCQM4Mv2データセット（約370万分子）でGCNバックボーンを事前学習
python scripts/pretrain.py --config config_pretrain.yaml
```

事前学習では以下を学習します：
- **HOMO-LUMO gap予測**: 分子の電子的性質を理解
- **分子グラフ表現**: 汎用的な化学構造の特徴抽出

`config_pretrain.yaml` の主要設定：

```yaml
pretraining:
  dataset: "PCQM4Mv2"
  data_path: "data/"
  task: "homo_lumo_gap"
  batch_size: 256
  num_epochs: 100
  learning_rate: 0.001

model:
  node_features: 48    # BitSpecと同じ特徴量次元
  edge_features: 6
  hidden_dim: 256
  num_layers: 5
```

### 2. EI-MSタスクへのファインチューニング

```bash
# 事前学習済みモデルをNIST EI-MSデータでファインチューニング
python scripts/finetune.py --config config_pretrain.yaml
```

ファインチューニングの戦略：

```yaml
finetuning:
  pretrained_checkpoint: "checkpoints/pretrain/pretrained_backbone.pt"

  # オプション1: バックボーン全体を凍結（予測ヘッドのみ学習）
  freeze_backbone: false

  # オプション2: 下位N層のみ凍結
  freeze_layers: 0

  # 異なる学習率の適用
  backbone_lr: 0.0001  # 事前学習済み層は低学習率
  head_lr: 0.001       # 新規層（spectrum_predictor）は高学習率
```

### 事前学習の効果

- **データ効率の向上**: 少ないEI-MSデータでも高性能
- **汎化性能の向上**: 未知の分子に対するロバスト性
- **学習の安定化**: より良い初期重みで収束が早い

詳細は [PRETRAINING_PROPOSAL.md](PRETRAINING_PROPOSAL.md) を参照してください。

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
  - ノード特徴量: 48次元 (原子番号12D、次数8D、形式電荷8D、キラリティ5D、水素数6D、混成軌道7D、芳香族性1D、環情報1D)
  - エッジ特徴量: 6次元 (結合タイプ4D、共役1D、環情報1D)
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

Modified Cosine Lossを使用:

```python
ModifiedCosineLoss = 1 - (CosineSimilarity + ShiftedMatching) / 2
```

- **通常のコサイン類似度**: スペクトル形状の基本的な類似性を評価
- **Shifted Matching**: プリカーサーイオンの質量差を考慮したピークシフトマッチング
  - Neutral loss（中性損失）を考慮してピークの対応関係を評価
  - 許容誤差（tolerance）により柔軟なピークマッチングを実現

この損失関数により、分子の構造的類似性とフラグメンテーションパターンの両方を効果的に学習できます。

許容誤差は `config.yaml` で調整可能:
```yaml
training:
  loss_tolerance: 0.1  # m/z単位での許容誤差
```

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
- **PCQM4Mv2**: Quantum Chemistry Structures and Properties of 134 kilo Molecules (OGB-LSC 2022)
- **Transfer Learning with GNNs**: "Transfer learning with graph neural networks for improved molecular property prediction" (Nature Communications, 2024)
- **Atom-level Pretraining**: "Pretraining graph transformers with atom-in-a-molecule quantum properties for improved ADMET modeling" (J. Cheminformatics, 2025)

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずIssueを開いて変更内容を議論してください。

## お問い合わせ

- **GitHub Issues**: https://github.com/turnDeep/BitSpec/issues
- **プロジェクトURL**: https://github.com/turnDeep/BitSpec

## 更新履歴

- **v1.2.0** (2025-11): PCQM4Mv2事前学習対応
  - PCQM4Mv2データセットでの事前学習機能追加
  - ファインチューニングスクリプト実装
  - 転移学習のための凍結戦略サポート
  - マルチタスク事前学習ヘッド追加
  - PretrainHead/MultiTaskPretrainHead実装
  - 詳細な技術文書追加（TECHNICAL_SUMMARY.md等）

- **v1.1.0** (2025-11): 特徴量最適化
  - 原子特徴量を157次元→48次元に最適化
  - 結合特徴量を16次元→6次元に最適化
  - MOL-NISTマッピングの厳密化
  - ModifiedCosineLossに統一

- **v1.0.0** (2024): 初回リリース
  - GCNベースのマススペクトル予測モデル
  - RTX 50シリーズ対応
  - MOL/MSP完全サポート
  - Dev Container対応
