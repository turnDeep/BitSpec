# NExtIMS - NEIMS v2.0: Neural EI-MS Prediction with Knowledge Distillation

NEIMSを基礎とした次世代の電子衝撃イオン化マススペクトル（EI-MS）予測システム。
Teacher-Student Knowledge Distillationと Mixture of Experts (MoE)アーキテクチャを使用して、高精度かつ高速な質量スペクトル予測を実現します。

## 特徴

- **Teacher-Student Knowledge Distillation**: 重いTeacherモデル（GNN+ECFP Hybrid）から軽量Studentモデル（MoE-Residual MLP）への知識転移
- **BDE-Aware Learning**: Bond Dissociation Energy（結合解離エネルギー）を補助タスクとして学習し、フラグメンテーションパターンをより正確に予測
- **Mixture of Experts (MoE)**: 4つの専門家ネットワークによる効率的な予測（Studentモデル）
- **Memory Efficient**: HDF5を用いた遅延読み込みにより、32GB RAMで全NISTデータの学習が可能
- **Uncertainty-Aware**: MC Dropoutによる不確実性を考慮した知識蒸留
- **Adaptive Loss Weighting**: GradNormによるマルチタスク損失の自動バランシング

## システム要件

- **Python**: 3.10+
- **PyTorch**: 2.4.0+ (CPU or CUDA)
- **RAM**: 32GB以上推奨（メモリ効率モード使用時）
- **GPU**: NVIDIA GPU 推奨 (CUDA 12.x)

## インストール

### 方法1: Dev Container (推奨)

VS CodeのDev Container機能を使用することで、環境構築を自動化できます。

### 方法2: ローカルインストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# HDF5サポート（メモリ効率モード用）
pip install h5py

# ALFABET（BDE計算用、Phase 0で使用）
pip install alfabet tensorflow

# プロジェクトのインストール
pip install -e .
```

## プロジェクト構造

```
BitSpec/
├── config.yaml                    # メイン設定ファイル
├── requirements.txt               # 依存関係
├── README.md                     # このファイル
├── data/
│   ├── NIST17.msp                # NIST EI-MSスペクトルデータ（ユーザーが用意）
│   ├── mol_files/                # MOLファイルディレクトリ（ユーザーが用意）
│   └── processed/                # 前処理済みデータ・キャッシュ
├── checkpoints/                   # モデルの保存先
├── src/
│   ├── data/                     # データ読み込み・前処理
│   │   ├── nist_dataset.py       # NISTデータセットローダー
│   │   ├── bde_generator.py      # BDE生成・管理
│   │   └── ...
│   ├── models/                   # モデル定義
│   │   ├── teacher.py            # Teacher Model (GNN+ECFP)
│   │   ├── student.py            # Student Model (MoE)
│   │   └── ...
│   └── training/                 # 学習ループ実装
└── scripts/
    ├── precompute_bde.py         # Phase 0: BDE事前計算
    ├── train_teacher.py          # Phase 1: Teacher学習
    ├── train_student.py          # Phase 2: Student学習
    ├── train_pipeline.py         # 統合学習パイプライン
    ├── predict.py                # 推論スクリプト
    └── benchmark_memory.py       # メモリ使用量ベンチマーク
```

## 学習ワークフロー

本システムは、NIST17データセットを用いて2段階（+事前準備）のプロセスで学習を行います。

### データの準備

`data/` ディレクトリに以下のファイルを配置してください：
- `NIST17.msp`: NIST EI-MS ライブラリ（MSP形式）
- `mol_files/`: 対応するMOLファイルが含まれるディレクトリ

### Phase 0: BDE事前計算 (推奨)

学習を高速化するため、Bond Dissociation Energy (BDE) を事前に計算してキャッシュします。

```bash
python scripts/precompute_bde.py --max-samples 0
```
これらにより `data/processed/bde_cache/bde_cache.h5` が生成されます。

### Phase 1: Teacher Model Training

Teacherモデル（GNN+ECFP）をマルチタスク学習（スペクトル予測 + BDE回帰）させます。

```bash
# 設定ファイル (config.yaml) に基づいて学習を実行
python scripts/train_teacher.py --config config.yaml --phase finetune
```
※ 現状の構成では `config.yaml` の `teacher_multitask` 設定を使用する場合でも、スクリプト上は `finetune` フェーズとして実行する形になる場合があります。設定ファイルの記述に従ってください。

### Phase 2: Student Model Distillation

学習済みTeacherモデルからStudentモデル（MoE）へ知識を蒸留します。

```bash
python scripts/train_student.py --config config.yaml --teacher checkpoints/teacher/best_finetune_teacher.pt
```

## 設定 (config.yaml)

主な設定項目：

- **Memory Efficient Mode**: `data.memory_efficient_mode` を `true` にすると、HDF5を用いた遅延読み込みが有効になり、メモリ使用量を大幅に削減できます。
- **Model Architecture**: Teacher/Studentの次元数やレイヤー数を設定可能です。
- **Training Hyperparameters**: 学習率、バッチサイズ、Lossの重み付けなどを調整できます。

## 推論

学習済みモデルを使用してスペクトル予測を行います。

```bash
# 単一SMILESの予測
python scripts/predict.py --config config.yaml --checkpoint checkpoints/student/best_student.pt --smiles "CCO"

# ファイルからのバッチ予測
python scripts/predict.py --config config.yaml --checkpoint checkpoints/student/best_student.pt --batch smiles_list.txt
```

## ライセンス

MIT License
