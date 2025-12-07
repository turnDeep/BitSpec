# DGL Installation for CUDA 12.8 (Blackwell / sm_120)

## 概要

このドキュメントでは、CUDA 12.8 (Blackwell / sm_120) 環境で DGL (Deep Graph Library) をソースからビルド・インストールする手順を説明します。

## 問題の背景

BonDNet のトレーニングスクリプト (`scripts/train_bondnet_bde_db2.py`) を実行すると、以下のエラーが発生します:

```
Alternative training failed: No module named 'dgl'
```

このエラーの直接的な原因は、DGL がインストールされていないことですが、根本原因は以下の通りです:

1. **CUDA 12.8 (Blackwell / sm_120) サポートの欠如**: 現在の DGL ソースコードのビルド設定ファイル (`cmake/modules/CUDA.cmake`) には、最新の Blackwell アーキテクチャ (sm_120) の定義が含まれていません。

2. **不適切なインストール方法**: `pip install git+https://github.com/dmlc/dgl.git` というコマンドは、DGL のような C++ コアを持つライブラリのインストールには不十分です。DGL は CMake を使って C++ ライブラリ (`libdgl.so`) を手動でビルドしてから、Python バインディングをインストールする必要があります。

## 解決策

### 自動インストール（推奨）

提供されているインストールスクリプトを使用して、DGL を自動的にインストールします。

```bash
# Dockerコンテナ内で実行
bash scripts/install_dgl_cuda_12_8.sh
```

このスクリプトは以下の処理を自動的に実行します:

1. DGL リポジトリのクローン
2. 必要なサブモジュールの初期化
3. CUDA 12.8 / sm_120 サポートの追加
4. C++ ライブラリのビルド
5. Python パッケージのインストール
6. インストールの検証

### 手動インストール

より詳細な制御が必要な場合は、以下の手順で手動インストールを行うことができます。

#### 1. 前提条件の確認

```bash
# CUDA 12.8 がインストールされていることを確認
nvcc --version

# PyTorch with CUDA support がインストールされていることを確認
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### 2. 環境変数の設定

```bash
export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 3. DGL リポジトリのクローン

```bash
# 作業ディレクトリに移動
cd /tmp

# Shallow clone で高速ダウンロード
git clone --depth=1 https://github.com/dmlc/dgl.git
cd dgl
```

#### 4. 必要なサブモジュールの初期化

```bash
git submodule update --init --depth=1 \
    third_party/dmlc-core \
    third_party/dlpack \
    third_party/cccl \
    third_party/cuco
```

#### 5. CUDA 12.8 / sm_120 サポートの追加

##### 方法 A: パッチファイルを使用（推奨）

```bash
# プロジェクトルートから修正済みファイルをコピー
cp <NExtIMS_PROJECT_ROOT>/patches/dgl/CUDA.cmake cmake/modules/CUDA.cmake
```

##### 方法 B: 手動で修正

`cmake/modules/CUDA.cmake` ファイルを編集し、以下の変更を適用します:

**変更1**: sm_120 サポートを追加（22行目付近の `endif()` の後に追加）

```cmake
# Add support for CUDA 12.8 / sm_120 (Blackwell)
if (CUDA_VERSION VERSION_GREATER_EQUAL "12.8")
  list(APPEND dgl_known_gpu_archs "120")
  set(dgl_cuda_arch_ptx "120")
endif()
```

**変更2**: Blackwell をアーキテクチャ名リストに追加（108行目付近）

```cmake
# 変更前
set(__archs_names "Kepler" "Maxwell" "Pascal" "Volta" "Turing" "Ampere" "Ada" "Hopper" "All" "Manual")

# 変更後
set(__archs_names "Kepler" "Maxwell" "Pascal" "Volta" "Turing" "Ampere" "Ada" "Hopper" "Blackwell" "All" "Manual")
```

**変更3**: Blackwell のアーキテクチャ設定を追加（159-161行目の Hopper の後に追加）

```cmake
elseif(${CUDA_ARCH_NAME} STREQUAL "Blackwell")
  set(__cuda_arch_bin "120")
  set(__cuda_arch_ptx "120")
```

#### 6. C++ ライブラリのビルド

```bash
mkdir -p build
cd build

cmake -DUSE_CUDA=ON \
      -DUSE_OPENMP=ON \
      -DCUDA_ARCH_NAME=Manual \
      -DCUDA_ARCH_BIN="12.0" \
      -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
      ..

# ビルド（10-20分程度かかります）
make -j$(nproc)
```

**注意**: `-DCUDA_ARCH_BIN="12.0"` は、DGL の正規表現処理により `120` (sm_120) に変換されます。

#### 7. Python パッケージのインストール

```bash
cd ../python
pip install -e .
```

#### 8. インストールの検証

```bash
python -c "import dgl; print(f'DGL Version: {dgl.__version__}')"
```

成功すると、DGL のバージョンが表示されます。

## トラブルシューティング

### エラー: `Cannot find CUDA`

**原因**: CMake が CUDA を検出できない。

**解決策**:
- `CUDA_TOOLKIT_ROOT_DIR` を明示的に指定
- 環境変数 `CUDA_HOME` が正しく設定されているか確認

```bash
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 ...
```

### エラー: `nvcc not found`

**原因**: CUDA コンパイラが PATH に含まれていない。

**解決策**: 環境変数を設定

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

### ビルドが遅い

**原因**: サブモジュールが大きい（特に `cccl`）。

**解決策**:
- Shallow clone (`--depth=1`) を使用（既に実施済み）
- ビルド時の並列度を調整: `make -j<コア数>`

### DGL のインポートに失敗

**原因**: Python パッケージが正しくインストールされていない。

**解決策**:
```bash
cd /tmp/dgl/python
pip uninstall dgl -y
pip install -e .
```

## BonDNet トレーニングの実行

DGL のインストールが完了したら、BonDNet トレーニングスクリプトを実行できます:

```bash
python scripts/train_bondnet_bde_db2.py \
    --data-dir data/processed/bondnet_training/ \
    --output models/bondnet_bde_db2_best.pth
```

## 参考資料

- [DGL 公式ドキュメント](https://docs.dgl.ai/)
- [DGL GitHub リポジトリ](https://github.com/dmlc/dgl)
- [CUDA Toolkit ドキュメント](https://docs.nvidia.com/cuda/)
- [CMake ドキュメント](https://cmake.org/documentation/)

## ライセンス

DGL は Apache License 2.0 の下でライセンスされています。
