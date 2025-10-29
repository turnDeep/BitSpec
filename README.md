# Mass Spectrum Prediction from Chemical Structures

化学構造からマススペクトルを予測する深層学習システム。Graph Convolutional Network (GCN)を使用して、SMILES文字列やMOLファイルから質量スペクトルを生成します。

## 特徴

- **Graph Convolutional Network**: 分子グラフの構造情報を効果的に学習
- **RTX 50シリーズ対応**: 最新のNVIDIA RTX 50シリーズGPUに最適化
- **Mixed Precision Training**: 高速化とメモリ効率化
- **複合損失関数**: MSE、コサイン類似度、KLダイバージェンスを組み合わせ
- **Attention Pooling**: 重要な分子部分構造に注目

## システム要件

- Python 3.8+
- CUDA 12.8+
- PyTorch 2.7+ (nightly)
- RTX 50シリーズGPU (推奨)

## インストール
```bash
# リポジトリのクローン
git clone https://github.com/yourusername/mass-spectrum-prediction.git
cd mass-spectrum-prediction

# 依存関係のインストール
pip install -r requirements.txt

# パッケージのインストール
pip install -e .
```

## Claude Codeの使用

このプロジェクトはClaude Codeに対応しています。Claude Codeを使用すると、AIアシスタントを使ってプロジェクトの開発や管理を効率化できます。

### 利用可能なコマンド

Claude Code環境内では、以下のスラッシュコマンドが使用できます:

- `/setup` - 開発環境のセットアップ
- `/preprocess` - データの前処理
- `/train` - モデルのトレーニング
- `/predict` - 予測の実行
- `/test` - テストの実行

### 使用方法

Claude Code環境を使用している場合は、直接スラッシュコマンドを実行できます:

```
/setup          # 環境をセットアップ
/preprocess     # データを前処理
/train          # モデルをトレーニング
/predict        # 予測を実行
```

または、Pythonスクリプトを直接実行することもできます:

```bash
# 環境セットアップ
pip install -r requirements.txt
pip install -e .

# データ前処理
python scripts/preprocess_data.py --input data/raw/nist_data.msp --output data/processed

# トレーニング
python scripts/train.py --config config.yaml

# 予測
python scripts/predict.py --checkpoint checkpoints/best_model.pt --config config.yaml --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

プロジェクト固有の質問や開発タスクについても、Claude Codeに直接質問できます。

## データの準備

NIST MSP形式のデータを準備します:
```bash
python scripts/preprocess_data.py \
    --input data/raw/nist_data.msp \
    --output data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

## トレーニング
```bash
python scripts/train.py --config config.yaml
```

### 設定のカスタマイズ

`config.yaml`を編集してハイパーパラメータを調整できます:
```yaml
model:
  hidden_dim: 256
  num_layers: 5
  dropout: 0.2

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
```

## 予測

### 単一分子の予測
```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --output prediction.png
```

### バッチ予測
```bash
# SMILES.txtに各行1つのSMILES文字列を記載
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --batch_file smiles.txt \
    --output batch_predictions/
```

### MSP形式でエクスポート
```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --export_msp
```

## Pythonスクリプトでの使用
```python
from src.models.gcn_model import MassSpectrumGCN
from scripts.predict import MassSpectrumPredictor

# 予測器の初期化
predictor = MassSpectrumPredictor(
    checkpoint_path='checkpoints/best_model.pt',
    config_path='config.yaml'
)

# 予測
spectrum = predictor.predict_from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O')

# ピークの検出
peaks = predictor.find_significant_peaks(spectrum, threshold=0.05, top_n=20)
print(f"Top 10 peaks: {peaks[:10]}")

# 可視化
predictor.visualize_prediction(
    smiles='CC(=O)OC1=CC=CC=C1C(=O)O',
    save_path='aspirin_spectrum.png'
)
```

## モデルアーキテクチャ
```
Input (SMILES) → Molecular Graph → GCN Layers → Global Pooling → MLP → Mass Spectrum
                                       ↓
                                  Attention
```

- **入力**: 分子グラフ（ノード特徴量: 44次元、エッジ特徴量: 12次元）
- **GCN層**: 5層のGraph Convolutional層（Residual接続付き）
- **Attention Pooling**: 重要なノードに注目
- **出力**: 1000次元のスペクトル（m/z 0-999）

## 評価メトリクス

- **Cosine Similarity**: スペクトル形状の類似度
- **Pearson Correlation**: ピーク強度の相関
- **MSE/MAE**: 予測誤差
- **Top-K Accuracy**: 主要ピークの一致率

## プロジェクト構造
```
mass-spectrum-prediction/
├── config.yaml              # 設定ファイル
├── requirements.txt         # 依存関係
├── setup.py                # パッケージ設定
├── README.md               # このファイル
├── data/
│   ├── raw/                # 生データ
│   └── processed/          # 前処理済みデータ
├── checkpoints/            # モデルチェックポイント
├── src/
│   ├── models/            # モデル定義
│   ├── features/          # 特徴量化
│   ├── data/              # データローダー
│   └── utils/             # ユーティリティ
└── scripts/
    ├── preprocess_data.py # データ前処理
    ├── train.py           # トレーニング
    └── predict.py         # 推論
```

## 参考文献

- NEIMS: Neural EI-MS Prediction
- ICEBERG/SCARF: MIT MS-Pred
- Massformer: Graph Transformer for Mass Spectrum

## ライセンス

MIT License

## お問い合わせ

Issue: https://github.com/yourusername/mass-spectrum-prediction/issues
