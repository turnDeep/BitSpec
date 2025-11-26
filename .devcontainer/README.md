# Dev Container での NExtIMS v2.0 実行方法

このディレクトリには、NExtIMS v2.0 (Neural EI-MS Prediction with Knowledge Distillation) プロジェクトをDev Containerで実行するための設定ファイルが含まれています。

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

# GPU確認
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 統合トレーニングパイプライン
python3 scripts/train_pipeline.py --config config.yaml
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

Docker Desktopの設定でメモリを増やしてください（推奨: 16GB以上）

### GPU が認識されない

```bash
# コンテナ内でGPUを確認
nvidia-smi

# PyTorchでGPUを確認
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### ビルドエラー

```bash
# キャッシュをクリアして再ビルド
docker system prune -a
# VS Codeでコマンドパレット > "Remote-Containers: Rebuild Container"
```

## NExtIMS v2.0 トレーニング

### 統合パイプライン（推奨）

```bash
# Dev Container内で
cd /workspace
python3 scripts/train_pipeline.py --config config.yaml
```

このコマンドは以下を自動実行します：
1. **Phase 1**: PCQM4Mv2データセットでTeacher事前学習（BDE RegressionまたはBond Masking）
2. **Phase 2**: NIST EI-MSデータでTeacherをファインチューニング（MC Dropout使用）
3. **Phase 3**: TeacherからStudentへの知識蒸留（Uncertainty-Aware KD）

### 個別ステップの実行

```bash
# Phase 0: BDE事前計算（BDE Regression使用時のみ）
python3 scripts/precompute_bde.py --max-samples 500000

# Phase 1: Teacher事前学習
python3 scripts/train_teacher.py --config config_pretrain.yaml --phase pretrain

# Phase 2: Teacherファインチューニング
python3 scripts/train_teacher.py --config config.yaml --phase finetune \
    --pretrained checkpoints/teacher/best_pretrain_teacher.pt

# Phase 3: Student蒸留
python3 scripts/train_student.py --config config.yaml \
    --teacher checkpoints/teacher/best_finetune_teacher.pt
```

### 予測の実行

```bash
# 単一分子予測（Student：高速）
python3 scripts/predict.py \
    --config config.yaml \
    --checkpoint checkpoints/student/best_student.pt \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"

# 不確実性推定付き予測（Teacher）
python3 scripts/predict.py \
    --config config.yaml \
    --checkpoint checkpoints/teacher/best_finetune_teacher.pt \
    --model teacher \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
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

## プロジェクト特徴

### NExtIMS v2.0 の主な特徴

- **Teacher-Student Knowledge Distillation**: GNN+ECFP HybridのTeacherから軽量MoE Studentへの知識転移
- **BDE Regression Pretraining**: Bond Dissociation Energy（結合解離エネルギー）を学習タスクとして使用
- **Mixture of Experts (MoE)**: 4つの専門家ネットワーク（芳香族、脂肪族、複素環、一般）
- **Uncertainty-Aware Distillation**: MC Dropoutによる不確実性を考慮した知識蒸留
- **RTX 5070 Ti最適化**: 16GB VRAMに最適化、Mixed Precision Training対応
- **メモリ効率的データローディング**: 32GB RAMでNIST17全データ（30万化合物）対応

### 性能目標

| メトリック | NEIMS v1.0 | NExtIMS v2.0 (目標) | 改善率 |
|--------|------------|------------------|-------|
| Recall@10 | 91.8% | 95.5-96.0% | +3.7-4.2% |
| Recall@5 | ~85% | 90-91% | +5-6% |
| 推論速度 | 5ms | 8-12ms | 1.6-2.4x遅 |
| GPU要件 | 不要 | 不要（推論時） | 同等 |

## 参考資料

- [Visual Studio Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [NExtIMS v2.0 README](../README.md)
- [システム仕様書](../docs/NEIMS_v2_SYSTEM_SPECIFICATION.md)
- [BDE事前学習ガイド](../docs/BDE_PRETRAINING_IMPLEMENTATION_GUIDE.md)
