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
    ├── train_pipeline.py       # 統合パイプライン（ダウンロード→事前学習→ファインチューニング）★推奨★
    ├── pretrain.py             # PCQM4Mv2事前学習
    ├── finetune.py             # ファインチューニング
    ├── train.py                # スクラッチからのトレーニング
    ├── predict.py              # 推論
    ├── preprocess_data.py      # データ前処理
    ├── test_training.py        # テストトレーニング
    ├── test_data_loading.py    # データ読み込みテスト
    └── test_mol_nist_mapping.py # MOL-NISTマッピングテスト
```

## クイックスタート

**推奨**: PCQM4Mv2事前学習とファインチューニングを使用したトレーニング

### 1. 統合パイプラインの実行（推奨）

**完全なワークフロー（PCQM4Mv2ダウンロード→事前学習→ファインチューニング）を1コマンドで実行:**

```bash
python scripts/train_pipeline.py --config config_pretrain.yaml
```

このコマンドは以下を自動的に実行します:
1. **PCQM4Mv2データセット（約370万分子）のダウンロード**
2. **事前学習**: 分子の量子化学的性質（HOMO-LUMO gap）を学習
3. **ファインチューニング**: NIST EI-MSデータでマススペクトル予測に特化

**既にPCQM4Mv2をダウンロード済みの場合:**

```bash
python scripts/train_pipeline.py --config config_pretrain.yaml --skip-download
```

**デバッグ用（小さなサブセットで高速テスト）:**

```bash
python scripts/train_pipeline.py --config config_pretrain.yaml --pretrain-subset 10000
```

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

### 3. 個別ステップの実行（オプション）

統合パイプラインの代わりに、各ステップを個別に実行することも可能です:

#### 3a. PCQM4Mv2事前学習

```bash
python scripts/pretrain.py --config config_pretrain.yaml
```

#### 3b. EI-MSファインチューニング

```bash
python scripts/finetune.py --config config_pretrain.yaml
```

#### 3c. スクラッチからの学習（事前学習なし）

```bash
python scripts/train.py --config config.yaml
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

## トレーニング戦略

### PCQM4Mv2事前学習とファインチューニング（推奨）

BitSpecは、大規模分子データセット（PCQM4Mv2、約370万分子）での事前学習を**基本戦略**としています。量子化学的性質（HOMO-LUMO gap）を学習することで、より少ないEI-MSデータでも高性能なモデルを構築できます。

### 統合パイプラインの使用（最も簡単）

**1コマンドで完全なワークフロー:**

```bash
python scripts/train_pipeline.py --config config_pretrain.yaml
```

**パイプラインのステップ:**
1. ✅ **PCQM4Mv2自動ダウンロード**: OGBライブラリ経由で約370万分子をダウンロード
2. ✅ **事前学習**: HOMO-LUMO gap予測タスクでGCNバックボーンを学習
3. ✅ **ファインチューニング**: NIST EI-MSデータでマススペクトル予測に特化
4. ✅ **モデル保存**: `checkpoints/finetune/best_finetuned_model.pt`

**オプションフラグ:**

```bash
# ダウンロードをスキップ（既にダウンロード済み）
python scripts/train_pipeline.py --config config_pretrain.yaml --skip-download

# 事前学習をスキップ（スクラッチから学習）
python scripts/train_pipeline.py --config config_pretrain.yaml --skip-pretrain

# デバッグ用（10,000サンプルのみで事前学習）
python scripts/train_pipeline.py --config config_pretrain.yaml --pretrain-subset 10000

# ファインチューニングのみ実行
python scripts/train_pipeline.py --config config_pretrain.yaml --skip-download --skip-pretrain
```

### 個別ステップの実行

パイプラインの各ステップを個別に実行することも可能です:

#### ステップ1: PCQM4Mv2事前学習

```bash
python scripts/pretrain.py --config config_pretrain.yaml
```

事前学習では以下を学習します：
- **HOMO-LUMO gap予測**: 分子の電子的性質を理解
- **分子グラフ表現**: 汎用的な化学構造の特徴抽出

#### ステップ2: EI-MSファインチューニング

```bash
python scripts/finetune.py --config config_pretrain.yaml
```

ファインチューニングの戦略（`config_pretrain.yaml`）：

```yaml
finetuning:
  pretrained_checkpoint: "checkpoints/pretrain/pretrained_backbone.pt"
  freeze_backbone: false  # バックボーン全体を凍結しない
  freeze_layers: 0        # 下位N層のみ凍結（0で凍結なし）
  backbone_lr: 0.0001     # 事前学習済み層は低学習率
  head_lr: 0.001          # 新規層（spectrum_predictor）は高学習率
```

### 事前学習の効果

- **データ効率の向上**: 少ないEI-MSデータでも高性能
- **汎化性能の向上**: 未知の分子に対するロバスト性
- **学習の安定化**: より良い初期重みで収束が早い
- **転移学習**: 量子化学の知識をマススペクトル予測に活用

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

Weighted Cosine Similarity Lossを使用:

```python
WeightedCosineLoss = 1 - cosine_similarity(pred, target)
```

- **コサイン類似度**: スペクトル形状の類似性を評価
- EI-MSスペクトル予測に適した設計
- 将来的にNIST標準の重み付け（m/z重み、強度重み）を追加可能

**注意**: EI-MSでは、MS/MSと異なり、Shifted Matching（Neutral loss考慮）は適用しません。
EI-MSはイオン化と同時にフラグメンテーションが発生するため、明確なプリカーサー-フラグメント関係が存在しないためです。

この損失関数により、NIST Mass Spectral Libraryと一貫性のある予測を実現します。

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

- **v1.3.0** (2025-11): 統合パイプライン追加
  - `train_pipeline.py`: PCQM4Mv2ダウンロード→事前学習→ファインチューニングの統合スクリプト
  - PCQM4Mv2自動ダウンロード機能（OGBライブラリ経由）
  - README.mdを事前学習とファインチューニングを基本戦略とする内容に更新
  - 1コマンドで完全なワークフローを実行可能に

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
  - WeightedCosineLossに統一（EI-MS専用設計）

- **v1.0.0** (2024): 初回リリース
  - GCNベースのマススペクトル予測モデル
  - RTX 50シリーズ対応
  - MOL/MSP完全サポート
  - Dev Container対応
