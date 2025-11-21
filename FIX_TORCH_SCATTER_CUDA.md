# torch_scatter CUDA エラーの修正方法

## エラー内容

```
RuntimeError: Not compiled with CUDA support
```

このエラーは、`torch_scatter`がCUDA対応版でインストールされていない場合に発生します。

## 重要な注意事項

**PyTorch 2.9以降の場合**: torch_scatterの公式ホイールはまだPyTorch 2.8までしかサポートしていません。自動修正スクリプトは互換性のあるPyTorch 2.8のホイールを試しますが、失敗する場合は以下の選択肢があります：

1. **ランタイムフォールバックを使用**（推奨・最も簡単）: コードに既に実装されているネイティブPyTorch実装が自動的に使用されます
2. **ソースからビルド**: 下記の「方法3」を参照
3. **PyTorchをダウングレード**: PyTorch 2.8以下にダウングレードする

## 修正方法

### 方法1: 自動修正スクリプトを使用（推奨）

最も簡単な方法は、提供されている修正スクリプトを実行することです：

```bash
bash fix_torch_scatter.sh
```

このスクリプトは以下を自動的に行います：
- PyTorchとCUDAのバージョンを検出
- 適切なCUDA対応版の`torch_scatter`をインストール
- インストール後にCUDA機能をテスト

### 方法2: 手動でインストール

PyTorchとCUDAのバージョンを確認してから、適切なバージョンをインストールします：

#### ステップ1: 現在の環境を確認

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

#### ステップ2: 既存のtorch_scatterをアンインストール

```bash
pip uninstall -y torch-scatter torch-sparse torch-cluster
pip cache purge
```

#### ステップ3: CUDA対応版をインストール

PyTorchのバージョンに応じて、以下のコマンドを実行します：

**PyTorch 2.7.x + CUDA 12.8の場合（RTX 50シリーズ対応）:**
```bash
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

**PyTorch 2.7.x + CUDA 12.6の場合:**
```bash
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

**PyTorch 2.7.x + CUDA 11.8の場合（最も互換性が高い）:**
```bash
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
```

**PyTorch 2.6.x + CUDA 12.4の場合:**
```bash
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

**PyTorch 2.8.x + CUDA 12.9の場合:**
```bash
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

**その他のバージョン:**

一般的な形式: `https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html`

- `${TORCH_VERSION}`: PyTorchのバージョン（例: 2.8.0, 2.7.0, 2.6.0）
- `${CUDA_VERSION}`: CUDAのバージョン（例: cu129, cu128, cu126, cu121, cu118）

**注意**: PyTorch 2.9以降はまだサポートされていません。PyTorch 2.8のホイールを試すか、方法3（ソースからビルド）を使用してください。

#### ステップ4: インストールを確認

```bash
python -c "
import torch
import torch_scatter
print(f'torch_scatter: {torch_scatter.__version__}')

# CUDA機能テスト
if torch.cuda.is_available():
    x = torch.randn(10, 3).cuda()
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3]).cuda()
    result = torch_scatter.scatter_max(x, batch, dim=0)
    print('✓ CUDA機能が正常に動作しています!')
"
```

### 方法3: ソースからビルド（PyTorch 2.9以降の場合）

PyTorch 2.9以降を使用していて、ホイールが利用できない場合は、ソースからビルドできます：

#### 前提条件

- CUDA Toolkit（システムにインストールされている必要があります）
- C++コンパイラ（gcc, g++など）
- PyTorchが既にインストールされている

#### ビルド手順

```bash
# 1. ビルドに必要なパッケージをインストール
pip install ninja

# 2. torch_scatterをソースからビルド
pip install git+https://github.com/rusty1s/pytorch_scatter.git

# 3. torch_sparseをソースからビルド（オプション）
pip install git+https://github.com/rusty1s/pytorch_sparse.git

# 4. torch_clusterをソースからビルド（オプション）
pip install git+https://github.com/rusty1s/pytorch_cluster.git
```

#### ビルド確認

```bash
python -c "
import torch
import torch_scatter
print(f'torch_scatter: {torch_scatter.__version__}')

# CUDA機能テスト
if torch.cuda.is_available():
    x = torch.randn(10, 3).cuda()
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3]).cuda()
    result = torch_scatter.scatter_max(x, batch, dim=0)
    print('✓ CUDA機能が正常に動作しています!')
"
```

**注意**: ビルドには5〜15分かかる場合があります。エラーが発生した場合は、CUDA Toolkitが正しくインストールされていることを確認してください。

## トラブルシューティング

### エラー: "Detected that PyTorch and torch_scatter were compiled with different CUDA versions"

PyTorchとtorch_scatterのCUDAバージョンが一致していません。以下を確認してください：

1. PyTorchのCUDAバージョンを確認：
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. 出力に`+cu128`のようなサフィックスが含まれている場合、それと同じバージョンを使用してください

3. 一致するバージョンをインストール：
   ```bash
   pip uninstall -y torch-scatter
   pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
   ```

### エラー: "CUDA out of memory"

これは別の問題です。バッチサイズを減らすか、モデルのサイズを調整してください。

### PyTorch 2.9でホイールインストールが失敗する

PyTorch 2.9はまだtorch_scatterの公式ホイールでサポートされていません。以下の対処方法があります：

1. **ランタイムフォールバックを使用**（推奨）:
   - `src/models/teacher.py`には既にネイティブPyTorch実装のフォールバックが組み込まれています
   - torch_scatterのインストールに失敗しても、トレーニングは自動的にネイティブ実装を使用します
   - パフォーマンスはわずかに低下しますが、機能的には同等です

2. **ソースからビルド**（上記の方法3を参照）:
   ```bash
   pip install ninja
   pip install git+https://github.com/rusty1s/pytorch_scatter.git
   ```

3. **PyTorchをダウングレード**（推奨しない）:
   ```bash
   pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   bash fix_torch_scatter.sh
   ```

### Dockerコンテナを使用している場合

Dockerコンテナ内でスクリプトを実行する必要があります：

```bash
docker exec -it <container_name> bash fix_torch_scatter.sh
```

または、コンテナに入ってから実行：

```bash
docker exec -it <container_name> bash
cd /workspace  # または適切なディレクトリ
bash fix_torch_scatter.sh
```

## 参考リンク

- [PyTorch Geometric インストールガイド](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [torch_scatter GitHubリポジトリ](https://github.com/rusty1s/pytorch_scatter)
- [PyG Wheel Repository](https://data.pyg.org/whl/)

## 注意事項

- `torch_scatter`のインストールは、**必ずPyTorchをインストールした後**に行ってください
- PyTorchとtorch_scatterのCUDAバージョンは**完全に一致**する必要があります
- `--no-cache-dir`オプションを使用して、古いキャッシュが使われないようにしてください
