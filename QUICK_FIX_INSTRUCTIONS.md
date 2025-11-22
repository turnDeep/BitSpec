# torch_scatter CUDA エラーのクイックフィックス

## 現状の診断

エラーログを分析した結果：

1. ✅ **コンテナは再ビルドされました** - `/opt/venv` が存在
2. ✅ **Python 3.11が使用されています**
3. ✅ **PyTorch Nightlyはインストールされています**
4. ❌ **torch_scatterがCUDA非対応でインストールされています**

## 問題の原因

Dockerfile のビルド中、PyTorch Nightly の動的バージョン（例: `2.7.0.dev20251122`）に対応する**ビルド済みホイールが PyG リポジトリに存在しない**ため、フォールバックでCPU版がインストールされた可能性があります。

## 解決方法

### オプション1: コンテナ内で修正スクリプトを実行（推奨）

コンテナ内でシェルを起動し、修正スクリプトを実行してください：

```bash
# VSCode Dev Container を使用している場合
# 既にコンテナ内のターミナルにいるはずです

# または、docker exec でコンテナに入る
# docker exec -it <container-name> /bin/bash

# 修正スクリプトを実行
bash /workspace/fix_torch_scatter_in_container.sh
```

このスクリプトは以下を実行します：
1. 現在のPyTorchとCUDAバージョンを確認
2. 既存のtorch_scatterを削除
3. PyTorchバージョンに合うビルド済みホイールからインストール
4. 失敗した場合は、ソースからCUDA対応ビルド
5. インストールを検証

### オプション2: Dockerfile を修正して再ビルド

より確実な方法として、Dockerfileを修正します：

**`.devcontainer/Dockerfile` の 110-115行目を以下に置き換えてください：**

```dockerfile
# torch_scatterなど拡張ライブラリをソースからCUDA対応ビルド
# ビルド済みホイールが見つからない場合に備えてソースビルドを強制
RUN TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])") && \
    CUDA_VERSION=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))") && \
    echo "Installing PyG extensions for PyTorch ${TORCH_VERSION} with ${CUDA_VERSION}" && \
    (pip install --no-cache-dir --no-build-isolation \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html || \
    (echo "⚠️  Wheels not found, building from source with CUDA support..." && \
     TORCH_CUDA_ARCH_LIST="9.0;12.0" FORCE_CUDA=1 \
     pip install --no-cache-dir --no-build-isolation \
     torch-scatter torch-sparse torch-cluster torch-spline-conv))
```

その後、コンテナを再ビルド：
```bash
# VSCode: Command Palette (Ctrl+Shift+P) → "Dev Containers: Rebuild Container"
```

### オプション3: 手動でインストール（デバッグ用）

コンテナ内で以下を実行：

```bash
# 既存パッケージを削除
pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv

# 環境変数を設定してソースからビルド
TORCH_CUDA_ARCH_LIST="9.0;12.0" \
FORCE_CUDA=1 \
pip install --no-cache-dir --no-build-isolation \
    torch-scatter torch-sparse torch-cluster torch-spline-conv

# 検証
python -c "
import torch
import torch_scatter
src = torch.randn(10, 5).cuda()
index = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1]).cuda()
out = torch_scatter.scatter(src, index, dim=0, reduce='sum')
print('✅ CUDA対応確認完了!')
"
```

## 検証方法

修正後、以下のコマンドで検証してください：

```bash
# GPU検証スクリプトが存在する場合
python /usr/local/bin/verify-gpu.py

# または手動で確認
python -c "
import torch
import torch_scatter
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'torch_scatter: {torch_scatter.__version__}')

# CUDA演算テスト
src = torch.randn(10, 5).cuda()
index = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1]).cuda()
out = torch_scatter.scatter(src, index, dim=0, reduce='sum')
print('✅ All tests passed!')
"
```

## トラブルシューティング

### ビルドに失敗する場合

1. **ninjaがインストールされているか確認**:
   ```bash
   pip install ninja
   ```

2. **CUDAツールキットが利用可能か確認**:
   ```bash
   nvcc --version
   ```

3. **メモリ不足の場合**:
   並列ビルド数を制限:
   ```bash
   MAX_JOBS=4 pip install --no-cache-dir torch-scatter
   ```

### それでも動作しない場合

PyTorch Nightlyではなく、安定版のPyTorch 2.6.0を使用することを検討してください。
`.devcontainer/Dockerfile` の 67-69行目を以下に変更：

```dockerfile
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

そして110-115行目を：
```dockerfile
RUN pip install --no-cache-dir --no-build-isolation \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
```

PyTorch 2.6.0でも sm_120（RTX 50シリーズ）をサポートしています。

## 次のステップ

修正完了後：
```bash
python scripts/train_pipeline.py --config config.yaml
```

でトレーニングを再実行してください。
