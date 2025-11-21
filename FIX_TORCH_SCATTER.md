# torch_scatter CUDA エラーのクイックフィックス

## エラー内容

```
RuntimeError: Not compiled with CUDA support
```

このエラーは`torch_scatter`がCPU版でインストールされており、GPUで実行できない状態です。

## クイックフィックス（推奨）

### ステップ1: インストールスクリプトを実行

```bash
# Pythonコマンドを指定して実行
PYTHON_CMD=/usr/bin/python bash install_torch_scatter_cuda.sh
```

このスクリプトが自動的に：
- PyTorchとCUDAのバージョンを検出
- 適切なCUDA対応版`torch_scatter`をインストール
- インストールを検証

### ステップ2: トレーニングを再実行

```bash
# 標準設定で実行
python scripts/train_pipeline.py --config config.yaml

# またはGPU最適化版で実行
bash run_gpu_optimized.sh
```

## 詳細な手順

より詳しい情報やトラブルシューティングについては以下を参照：

- **[INSTALL_CUDA.md](INSTALL_CUDA.md)** - 詳細なインストールガイド
- **手動インストール方法**
- **GPU最適化設定**
- **トラブルシューティング**

## 手動修正（スクリプトが使えない場合）

### 1. PyTorchとCUDAのバージョン確認

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

### 2. 既存の torch_scatter をアンインストール

```bash
pip uninstall -y torch-scatter torch-sparse torch-cluster
pip cache purge
```

### 3. CUDA対応版をインストール

**PyTorch 2.7.0 + CUDA 12.8の場合:**
```bash
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

**PyTorch 2.7.0 + CUDA 12.6の場合:**
```bash
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

### 4. インストール確認

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

## GPU最適化設定

エラーを修正した後、さらにパフォーマンスを向上させるには：

### オプション1: 最適化スクリプトを使用

```bash
# 環境変数を設定して最適化実行
bash run_gpu_optimized.sh
```

### オプション2: GPU最適化設定を使用

```bash
# より大きなバッチサイズと最適化設定
python scripts/train_pipeline.py --config config_gpu_optimized.yaml
```

### オプション3: 環境変数を手動設定

```bash
# CUDA最適化
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"
export CUDNN_BENCHMARK=1

# 実行
python scripts/train_pipeline.py --config config.yaml
```

## バッチサイズの調整

GPUメモリに応じてバッチサイズを調整してください：

| GPU | VRAMサイズ | 推奨バッチサイズ |
|-----|-----------|----------------|
| RTX 4060 Ti | 16GB | 128 |
| RTX 5070 Ti | 16GB | 128 |
| RTX 4090 | 24GB | 256 |
| RTX 5080 | 24GB | 256 |
| RTX 5090 | 32GB | 512 |

`config.yaml` または `config_gpu_optimized.yaml` の以下の値を変更：

```yaml
training:
  teacher_pretrain:
    batch_size: 256  # ← ここを調整
```

## トラブルシューティング

### エラー: CUDA out of memory

バッチサイズを減らすか、勾配累積を使用：

```yaml
gpu:
  memory_optimization:
    gradient_accumulation_steps: 2  # 実質的にバッチサイズを半分に
```

### エラー: cuDNN version mismatch

PyTorchを再インストール：

```bash
pip uninstall torch torchvision torchaudio
pip install torch>=2.7.0 torchvision>=0.22.0 torchaudio>=2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

その後、`install_torch_scatter_cuda.sh`を再実行。

### エラー: Different CUDA versions detected

PyTorchと`torch_scatter`のCUDAバージョンが一致していません：

```bash
# install_torch_scatter_cuda.shを再実行
bash install_torch_scatter_cuda.sh
```

## サポート

問題が解決しない場合：

1. 詳細なログを確認：エラーメッセージ全体を保存
2. 環境情報を収集：
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   pip list | grep torch
   nvidia-smi
   ```
3. GitHubのIssueで報告

## 参考リンク

- [INSTALL_CUDA.md](INSTALL_CUDA.md) - 詳細なインストールガイド
- [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [torch_scatter GitHub](https://github.com/rusty1s/pytorch_scatter)
