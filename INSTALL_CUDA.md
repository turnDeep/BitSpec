# CUDA対応torch_scatterインストールガイド

## 問題の概要

`RuntimeError: Not compiled with CUDA support` エラーが発生する場合、`torch_scatter`がCPU版でインストールされているか、PyTorchとCUDAのバージョンが一致していません。

## 原因

`torch_scatter`、`torch_sparse`、`torch_cluster`などのPyTorch Geometric拡張ライブラリは、以下の条件を満たす必要があります：

1. **PyTorchのバージョンと厳密に一致**する必要がある
2. **CUDAのバージョンと一致**する必要がある
3. **事前にコンパイルされたホイール**を使用する必要がある（ソースからのビルドは推奨されない）

## 解決方法

### 方法1: 自動インストールスクリプト（推奨）

提供されたスクリプトを使用して、自動的に正しいバージョンをインストールします：

```bash
# Pythonコマンドを指定して実行
PYTHON_CMD=python bash install_torch_scatter_cuda.sh

# またはDockerコンテナ内の場合
PYTHON_CMD=/usr/bin/python bash install_torch_scatter_cuda.sh
```

このスクリプトは以下を自動的に実行します：
- 現在のPyTorchとCUDAバージョンを検出
- 適切なホイールURLを決定
- 既存の`torch_scatter`をアンインストール
- CUDA対応版を再インストール
- インストールの検証とテスト

### 方法2: 手動インストール

1. **PyTorchとCUDAのバージョンを確認**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

出力例：
```
PyTorch: 2.7.0
CUDA: 12.8
```

2. **既存のtorch_scatterをアンインストール**

```bash
pip uninstall -y torch-scatter torch-sparse torch-cluster
pip cache purge
```

3. **対応するCUDA版をインストール**

PyTorch 2.7.0 + CUDA 12.8の場合：
```bash
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

PyTorch 2.7.0 + CUDA 12.6の場合：
```bash
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

PyTorch 2.6.0 + CUDA 12.6の場合：
```bash
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
```

4. **インストールを確認**

```bash
python -c "
import torch
import torch_scatter
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'torch_scatter: {torch_scatter.__version__}')

# CUDA機能テスト
if torch.cuda.is_available():
    x = torch.randn(10, 3).cuda()
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3]).cuda()
    result = torch_scatter.scatter_max(x, batch, dim=0)
    print('✓ CUDA機能が正常に動作しています')
"
```

## GPU計算資源の最大活用

### 1. CUDA環境変数の最適化

```bash
# GPU可視性の設定（複数GPU環境の場合）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# CUDAアーキテクチャの明示的指定（RTX 50シリーズの場合）
export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"

# CUDAキャッシュの最適化
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB

# cuDNN最適化
export CUDNN_BENCHMARK=1
export CUDNN_DETERMINISTIC=0
```

### 2. PyTorchの最適化設定

Pythonコード内で以下を設定：

```python
import torch

# cuDNNベンチマークモードを有効化（入力サイズが固定の場合）
torch.backends.cudnn.benchmark = True

# TensorFloat-32を有効化（Ampere以降のGPU）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# メモリアロケーターの最適化
torch.cuda.empty_cache()  # 開始前にキャッシュをクリア
```

### 3. バッチサイズとWorker数の最適化

`config.yaml`で以下を調整：

```yaml
# GPUメモリに応じてバッチサイズを最大化
training:
  batch_size: 128  # GPUメモリが許す限り大きく（16GB: 64-128, 24GB: 128-256, 40GB: 256-512）

data:
  num_workers: 8  # CPUコア数の半分程度（データローディングの並列化）
  pin_memory: true  # GPUへのデータ転送を高速化
  persistent_workers: true  # Workerの再利用
```

### 4. Mixed Precision Training (AMP)

`config.yaml`でAMPを有効化：

```yaml
training:
  use_amp: true  # 自動混合精度（メモリ使用量を半減、速度1.5-2倍）
  grad_clip: 1.0  # 勾配クリッピング
```

### 5. GPU並列処理（複数GPU）

複数GPUがある場合：

```python
# DataParallel（簡単だが効率は中程度）
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# DistributedDataParallel（推奨、高効率）
# torchrun --nproc_per_node=4 scripts/train_teacher.py --config config.yaml
```

### 6. メモリ最適化

```python
# 勾配チェックポインティング（メモリ削減、速度はやや低下）
from torch.utils.checkpoint import checkpoint

# モデル定義内で
def forward(self, x):
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
```

### 7. コンパイルモード（PyTorch 2.0+）

```python
# torch.compileで計算グラフを最適化（PyTorch 2.0以降）
model = torch.compile(model, mode="reduce-overhead")  # または "max-autotune"
```

## RTX 50シリーズ特有の最適化

### Blackwellアーキテクチャ（sm_120）の活用

```bash
# アーキテクチャを明示的に指定
export TORCH_CUDA_ARCH_LIST="12.0"

# FP8サポート（RTX 50シリーズの新機能）
# PyTorch 2.7以降で自動的にサポート
```

### 大容量メモリの活用（32GB/48GB）

```yaml
# バッチサイズを大幅に増やす
training:
  batch_size: 512  # RTX 5090 (32GB)の場合
  gradient_accumulation_steps: 1  # 勾配累積は不要

# より大きなモデル
model:
  teacher:
    gnn_branch:
      hidden_dim: 512  # デフォルトより大きく
      num_layers: 8
```

## トラブルシューティング

### エラー: `CUDA out of memory`

```bash
# バッチサイズを減らす
training.batch_size: 64 → 32

# または勾配累積を使用
training.gradient_accumulation_steps: 2  # 実質的にバッチサイズを2倍に
```

### エラー: `cuDNN version mismatch`

```bash
# PyTorchを再インストール
pip uninstall torch torchvision torchaudio
pip install torch>=2.7.0 torchvision>=0.22.0 torchaudio>=2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### エラー: `RuntimeError: Detected that PyTorch and torch_scatter were compiled with different CUDA versions`

```bash
# install_torch_scatter_cuda.shを再実行
bash install_torch_scatter_cuda.sh
```

## 性能ベンチマーク

推奨設定での期待性能（RTX 5090 32GB基準）：

| 設定 | スループット | メモリ使用量 | 備考 |
|------|-------------|-------------|------|
| バッチサイズ64 + FP32 | ~200 samples/sec | ~12GB | 基本設定 |
| バッチサイズ128 + AMP | ~400 samples/sec | ~14GB | 推奨 |
| バッチサイズ256 + AMP + Compile | ~600 samples/sec | ~20GB | 最適化 |
| バッチサイズ512 + AMP + Compile | ~800 samples/sec | ~28GB | 最大活用 |

## 参考リンク

- [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [torch_scatter GitHub](https://github.com/rusty1s/pytorch_scatter)
- [PyTorch CUDA最適化ガイド](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
