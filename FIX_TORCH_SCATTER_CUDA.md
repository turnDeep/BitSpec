# torch_scatter CUDA エラーの修正

## 問題の概要

トレーニングパイプラインで以下のエラーが発生：

```
RuntimeError: Not compiled with CUDA support
```

**エラー発生箇所**: `torch_scatter.scatter()` 関数呼び出し時

## 根本原因

1. **torch_scatter がCPU版でインストールされていた**
   - PyTorch Nightlyとtorch_scatterのCUDAバージョンミスマッチ
   - ビルド済みホイールが見つからず、ソースからビルド時にCUDAサポートが有効化されていなかった

2. **Dockerfile の問題点**
   - Python 3.10使用（PyTorch Nightlyとの互換性が低い）
   - 仮想環境を使用していない
   - `TORCH_CUDA_ARCH_LIST` 環境変数が未設定（sm_120サポートに必須）
   - PyTorchインストールの複雑なフォールバックロジックが失敗

## 解決策

### 主要な変更点

#### 1. Python 3.11への移行
```dockerfile
# Python 3.11（PyTorch Nightlyと互換性が良い）
python3.11 python3.11-dev python3.11-venv
```

#### 2. Python仮想環境の導入
```dockerfile
# 仮想環境の作成
RUN python3.11 -m venv /opt/venv

# システムPythonを仮想環境のものに置き換え
RUN ln -sf /opt/venv/bin/python /usr/bin/python && \
    ln -sf /opt/venv/bin/python /usr/bin/python3 && \
    ln -sf /opt/venv/bin/pip /usr/bin/pip && \
    ln -sf /opt/venv/bin/pip /usr/bin/pip3

ENV PATH="/opt/venv/bin:$PATH"
```

#### 3. CUDA環境変数の設定（sm_120対応）
```dockerfile
ENV TORCH_CUDA_ARCH_LIST="9.0;12.0" \
    CUDA_LAUNCH_BLOCKING=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### 4. PyTorch Nightlyのインストール方法改善
```dockerfile
# シンプルで確実なインストール
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 5. ビルドツールの事前インストール
```dockerfile
# 依存関係問題を回避
RUN pip install --no-cache-dir six hatchling wheel ninja
```

#### 6. torch_scatterのインストール改善
```dockerfile
# 動的にバージョンを検出してCUDA対応ホイールをインストール
RUN TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])") && \
    CUDA_VERSION=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))") && \
    echo "Installing PyG extensions for PyTorch ${TORCH_VERSION} with ${CUDA_VERSION}" && \
    pip install --no-cache-dir --no-build-isolation \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
```

## 参考リポジトリ

この修正は、以下のリポジトリで成功している設定を参考にしました：
- https://github.com/turnDeep/Computational-Chemistry-AI

同リポジトリでは同じRTX 5070ti（sm_120）環境で正常動作を確認済み。

## 検証方法

コンテナ起動後、以下のコマンドで環境を検証できます：

```bash
# GPU検証スクリプトの実行
python /usr/local/bin/verify-gpu.py
```

このスクリプトは以下を確認します：
- ✅ CUDA利用可能性
- ✅ sm_120（Blackwell）検出
- ✅ PyTorchバージョン情報
- ✅ GPU演算テスト
- ✅ torch_scatter CUDA対応確認

## 次のステップ

1. コンテナの再ビルド:
   ```bash
   # VSCode Dev Container: "Rebuild Container"
   # または
   docker-compose build --no-cache
   ```

2. 環境検証:
   ```bash
   python /usr/local/bin/verify-gpu.py
   ```

3. トレーニング再実行:
   ```bash
   python scripts/train_pipeline.py --config config.yaml
   ```

## 変更ファイル

- `.devcontainer/Dockerfile` - 完全書き換え
- `requirements.txt` - PyTorch Nightly指定に変更
- `FIX_TORCH_SCATTER_CUDA.md` - この文書（修正記録）

## 技術的詳細

### なぜPython 3.11が必要か？

PyTorch Nightlyは最新のPythonバージョンでテストされており、3.11との互換性が最も高い。特に、C++拡張モジュール（torch_scatterなど）のビルド時に問題が発生しにくい。

### なぜ仮想環境が必要か？

システムPythonとpipパッケージの競合を避け、確実に正しいPythonバージョンとパッケージが使用されるようにするため。参考リポジトリでも同じアプローチを採用。

### TORCH_CUDA_ARCH_LISTの意味

PyTorchとCUDA拡張モジュールがコンパイルされる際のターゲットGPUアーキテクチャを指定：
- `9.0` - Ada Lovelace (RTX 40シリーズ)
- `12.0` - Blackwell (RTX 50シリーズ、sm_120)

## 参考リンク

Web検索で調査した解決方法：

- [How to resolve Torch not compiled with CUDA enabled](https://www.educative.io/answers/how-to-resolve-torch-not-compiled-with-cuda-enabled)
- [torch-scatter RuntimeError Issue #394](https://github.com/rusty1s/pytorch_scatter/issues/394)
- [PyTorch and torch_scatter CUDA version mismatch](https://stackoverflow.com/questions/70008715/pytorch-and-torch-scatter-were-compiled-with-different-cuda-versions-on-google-c)
- [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [PyTorch Forums Discussion](https://discuss.pytorch.org/t/detected-that-pytorch-and-torch-scatter-were-compiled-with/125975)
