# torch_scatter CUDA エラーの修正方法

## エラー内容

```
RuntimeError: Not compiled with CUDA support
```

このエラーは、`torch_scatter`がCUDA対応版でインストールされていない場合に発生します。

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

**その他のバージョン:**

一般的な形式: `https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html`

- `${TORCH_VERSION}`: PyTorchのバージョン（例: 2.7.0, 2.6.0）
- `${CUDA_VERSION}`: CUDAのバージョン（例: cu128, cu126, cu121, cu118）

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
