# NExtIMS v4.2 インストールガイド

## 概要

NExtIMS v4.2のインストール方法を説明します。開発モードと通常モードの両方に対応しています。

## システム要件

### ハードウェア

| 項目 | 推奨 | 最小 |
|------|------|------|
| **GPU** | RTX 5070 Ti (16GB) | RTX 3060 (12GB) |
| **RAM** | 32GB | 16GB |
| **CPU** | Ryzen 7700 (8コア) | 4コア以上 |
| **ストレージ** | 100GB+ SSD | 50GB HDD |

### ソフトウェア

- **Python**: 3.10, 3.11, 3.12
- **CUDA**: 12.8+ (RTX 50シリーズの場合)
- **OS**: Linux (Ubuntu 22.04+推奨), Windows 10/11, macOS

## インストール方法

### 方法1: 開発モード（推奨）

開発やコントリビューション向けのインストール方法です。

#### 1. リポジトリのクローン

```bash
git clone https://github.com/turnDeep/NExtIMS.git
cd NExtIMS
```

#### 2. 仮想環境の作成（推奨）

```bash
# venv使用
python -m venv venv
source venv/bin/activate  # Linux/macOS
# または
venv\Scripts\activate     # Windows

# または conda使用
conda create -n nextims python=3.10
conda activate nextims
```

#### 3. 開発モードでインストール

```bash
pip install -e .
```

**開発モードの利点**:
- ✅ ソースコードの変更が即座に反映
- ✅ デバッグが容易
- ✅ コントリビューションに最適

#### 4. インストール確認

```bash
# バージョン確認
python -c "import sys; sys.path.insert(0, 'src'); from models import QCGN2oEI_Minimal; print('✅ Installation successful')"

# コマンドラインツールの確認
nextims-train --help
nextims-evaluate --help
nextims-predict --help
nextims-predict-batch --help
```

### 方法2: 通常モード

ユーザーとして使用する場合のインストール方法です。

#### GitHubから直接インストール

```bash
pip install git+https://github.com/turnDeep/NExtIMS.git
```

#### ローカルから通常インストール

```bash
git clone https://github.com/turnDeep/NExtIMS.git
cd NExtIMS
pip install .
```

### 方法3: requirements.txtのみ（最小構成）

パッケージインストールなしで依存関係のみインストールする場合：

```bash
pip install -r requirements.txt
```

**注意**: この方法ではコマンドラインツール（`nextims-train`等）は使用できません。

## コマンドラインツール

setup.pyをインストールすると、以下のコマンドが使用可能になります：

### 1. nextims-train

GNNモデルの訓練

```bash
nextims-train \
    --nist-msp data/NIST17.MSP \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --output models/qcgn2oei_minimal_best.pth \
    --epochs 300 \
    --batch-size 32
```

### 2. nextims-evaluate

モデルの評価

```bash
nextims-evaluate \
    --model models/qcgn2oei_minimal_best.pth \
    --nist-msp data/NIST17.MSP \
    --visualize \
    --benchmark \
    --output-dir results/evaluation
```

### 3. nextims-predict

単一分子のスペクトル予測

```bash
nextims-predict \
    --smiles "CCO" \
    --model models/qcgn2oei_minimal_best.pth \
    --output ethanol_prediction.png
```

### 4. nextims-predict-batch

バッチ予測

```bash
nextims-predict-batch \
    --input molecules.csv \
    --model models/qcgn2oei_minimal_best.pth \
    --output predictions.csv \
    --batch-size 64
```

## 依存関係の詳細

### 必須パッケージ

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| torch | nightly (CUDA 12.8) | ディープラーニングフレームワーク |
| torch-geometric | >=2.5.0 | グラフニューラルネットワーク |
| rdkit | >=2023.9.1 | 化学構造処理 |
| numpy | >=1.24.0 | 数値計算 |
| pandas | >=2.0.0 | データ処理 |
| h5py | >=3.10.0 | HDF5ファイル処理 |
| pyyaml | >=6.0 | 設定ファイル読み込み |
| tqdm | >=4.66.0 | プログレスバー |

### 可視化パッケージ

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| matplotlib | >=3.7.0 | スペクトルプロット |
| seaborn | >=0.13.0 | 統計的可視化 |
| plotly | >=5.18.0 | インタラクティブ可視化 |

### 開発ツール

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| pytest | >=7.4.0 | ユニットテスト |
| black | >=23.12.0 | コードフォーマット |
| flake8 | >=7.0.0 | Linter |
| mypy | >=1.8.0 | 型チェック |

## RTX 50シリーズ特有の設定

### PyTorch Nightly Build

RTX 5070 Tiなど、Blackwellアーキテクチャ（sm_120）の場合、PyTorch nightly buildが必須です。

```bash
# CUDA 12.8対応のPyTorch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### CUDA確認

```bash
# CUDA利用可能か確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

期待される出力:
```
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5070 Ti
```

## トラブルシューティング

### エラー1: "No module named 'torch'"

**原因**: PyTorchがインストールされていない

**解決策**:
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### エラー2: "No module named 'rdkit'"

**原因**: RDKitがインストールされていない

**解決策**:
```bash
pip install rdkit
# または conda経由
conda install -c conda-forge rdkit
```

### エラー3: "CUDA out of memory"

**原因**: GPUメモリ不足

**解決策**:
```bash
# バッチサイズを削減
nextims-train --batch-size 16  # デフォルトは32

# または config.yaml を編集
training:
  batch_size: 16
```

### エラー4: "command not found: nextims-train"

**原因**: setup.pyが正しくインストールされていない

**解決策**:
```bash
# 開発モードで再インストール
pip install -e .

# またはPATHを確認
which nextims-train

# または直接スクリプトを実行
python scripts/train_gnn_minimal.py --help
```

### エラー5: "ImportError: cannot import name 'QCGN2oEI_Minimal'"

**原因**: パッケージが正しくインストールされていない

**解決策**:
```bash
# 再インストール
pip uninstall nextims
pip install -e .

# またはPYTHONPATHを設定
export PYTHONPATH="${PYTHONPATH}:/path/to/NExtIMS/src"
```

## アンインストール

### 開発モード

```bash
pip uninstall nextims
```

### 依存関係も含めて完全削除

```bash
pip uninstall nextims
pip uninstall -r requirements.txt
```

### 仮想環境ごと削除

```bash
# venvの場合
deactivate
rm -rf venv

# condaの場合
conda deactivate
conda env remove -n nextims
```

## 環境変数の設定（オプション）

### データパスの設定

```bash
export NEXTIMS_DATA_DIR="/path/to/data"
export NEXTIMS_MODEL_DIR="/path/to/models"
export NEXTIMS_OUTPUT_DIR="/path/to/outputs"
```

### CUDAデバイスの指定

```bash
export CUDA_VISIBLE_DEVICES=0  # GPU 0のみ使用
```

### ログレベルの設定

```bash
export NEXTIMS_LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## Docker環境でのインストール

Dockerfileが提供されている場合：

```bash
# イメージのビルド
docker build -t nextims:v4.2 .

# コンテナの起動
docker run --gpus all -it nextims:v4.2 bash

# コンテナ内で確認
nextims-train --help
```

## 開発者向けセットアップ

### 追加の開発ツールをインストール

```bash
pip install -e ".[dev]"  # setup.pyに[dev]セクションがある場合
```

### pre-commitフックの設定

```bash
pip install pre-commit
pre-commit install
```

### テストの実行

```bash
# すべてのテストを実行
pytest tests/

# 特定のテストを実行
pytest tests/test_models.py

# カバレッジ付き
pytest --cov=src tests/
```

## 次のステップ

インストール後の推奨手順：

1. **クイックスタート**
   ```bash
   # クイックスタートガイドを参照
   cat QUICKSTART.md
   ```

2. **データの準備**
   ```bash
   # NIST17データ構造のセットアップ
   cat docs/NIST17_DATA_STRUCTURE.md
   ```

3. **設定ファイルの確認**
   ```bash
   # config.yamlの確認
   cat config.yaml
   ```

4. **Phase 0: BDE環境構築**
   ```bash
   # BDEキャッシュの生成
   # （詳細は仕様書参照）
   ```

5. **Phase 2: 訓練開始**
   ```bash
   nextims-train \
       --nist-msp data/NIST17.MSP \
       --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
       --output models/qcgn2oei_minimal_best.pth
   ```

## 参考資料

- **README.md**: プロジェクト概要
- **QUICKSTART.md**: 5分で始めるガイド
- **docs/spec_v4.2_minimal_iterative.md**: 技術仕様書
- **docs/CONFIG_v4.2_CHANGES.md**: 設定変更ガイド
- **docs/NIST17_DATA_STRUCTURE.md**: データ構造ガイド
- **docs/PREDICTION_GUIDE.md**: 推論使用ガイド

## サポート

問題が発生した場合：

1. **Issue報告**: https://github.com/turnDeep/NExtIMS/issues
2. **ドキュメント**: https://github.com/turnDeep/NExtIMS/tree/main/docs
3. **FAQ**: `docs/CONFIG_v4.2_CHANGES.md` のFAQセクション

---

**最終更新**: 2025-12-03
**バージョン**: v4.2.0
**対象Python**: 3.10, 3.11, 3.12
