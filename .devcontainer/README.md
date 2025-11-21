# Dev Container での NExtIMS 実行方法

このディレクトリには、NExtIMS (Next Extended NEIMS) プロジェクトをDev Containerで実行するための設定ファイルが含まれています。

## 必要な環境

- Docker Desktop または Docker Engine (20.10以上)
- Visual Studio Code
- Remote - Containers 拡張機能
- NVIDIA Docker (GPUを使用する場合)

## セットアップ手順

### 1. Docker と VS Code の準備

```bash
# Dockerがインストールされているか確認
docker --version

# VS Codeがインストールされているか確認
code --version
```

### 2. VS Code 拡張機能のインストール

VS Codeで以下の拡張機能をインストール:
- Remote - Containers (ms-vscode-remote.remote-containers)

### 3. Dev Container を開く

1. VS Codeでプロジェクトのルートディレクトリを開く
2. `F1` キーを押して、コマンドパレットを開く
3. "Remote-Containers: Reopen in Container" を選択
4. Dev Containerのビルドが始まります（初回は時間がかかります）

### 4. コンテナ内での作業

コンテナが起動したら、VS Codeのターミナルでコマンドを実行できます：

```bash
# 依存関係のインストール確認
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import rdkit; print('RDKit: OK')"

# プロジェクトのインストール
pip3 install -e .

# 10個のサンプルでテストトレーニング
python3 scripts/test_training.py
```

## GPU サポート

### NVIDIA GPU を使用する場合

1. NVIDIA Dockerのインストール

```bash
# NVIDIA Container Toolkitのインストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. GPUが認識されているか確認

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

3. Dev ContainerでGPUを使用

`devcontainer.json`には既に`--gpus=all`が設定されているため、コンテナ起動時に自動的にGPUが利用可能になります。

## トラブルシューティング

### Docker メモリ不足

Docker Desktopの設定でメモリを増やしてください（推奨: 8GB以上）

### GPU が認識されない

```bash
# コンテナ内でGPUを確認
nvidia-smi

# PyTorchでGPUを確認
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### ビルドエラー

```bash
# キャッシュをクリアして再ビルド
docker system prune -a
# VS Codeでコマンドパレット > "Remote-Containers: Rebuild Container"
```

## 10個のMOLファイルでのテスト

テストトレーニングスクリプト (`scripts/test_training.py`) は、10個のMOLファイルを使用して以下を実行します：

1. **データセットの作成**: NIST MSPファイルとMOLファイルから最初の10サンプルを読み込み
2. **モデルの作成**: GCN (Graph Convolutional Network) モデルを構築
3. **トレーニング**: 5エポックの訓練を実行
4. **評価**: コサイン類似度などのメトリクスを計算
5. **保存**: 訓練済みモデルを `checkpoints/test/test_model.pt` に保存

実行方法:

```bash
# Dev Container内で
cd /workspace
python3 scripts/test_training.py
```

期待される出力:
```
============================================================
10個のMOLファイルを使用したテストトレーニング
============================================================
使用デバイス: cuda:0 (または cpu)
CUDA デバイス: NVIDIA RTX 5090

1. データセットの作成
------------------------------------------------------------
全データ数: 900 サンプル
テスト用サンプル数: 10
  訓練データ: 8
  検証データ: 1
  テストデータ: 1
  訓練バッチ数: 4
  検証バッチ数: 1

2. モデルの作成
------------------------------------------------------------
  ノード特徴量次元: 155
  エッジ特徴量次元: 15
  総パラメータ数: 1,234,567
  訓練可能パラメータ数: 1,234,567

3. 訓練設定
------------------------------------------------------------
  損失関数: SpectrumLoss (combined)
  オプティマイザ: Adam
  学習率: 0.001

4. トレーニング開始
------------------------------------------------------------
Epoch 1/5 [Train]: 100%|██████████| 4/4 [00:05<00:00, loss=0.8234, cos_sim=0.1234]
...
```

## ファイル構成

```
.devcontainer/
├── Dockerfile          # Docker イメージの定義
├── devcontainer.json   # VS Code Dev Container の設定
└── README.md          # このファイル
```

## カスタマイズ

### Pythonパッケージの追加

`Dockerfile` を編集して、必要なパッケージを追加できます：

```dockerfile
RUN pip3 install --no-cache-dir your-package-name
```

### VS Code 拡張機能の追加

`devcontainer.json` の `extensions` セクションに追加：

```json
"extensions": [
    "ms-python.python",
    "your-extension-id"
]
```

## 参考資料

- [Visual Studio Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
