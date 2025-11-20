# BitSpec - NEIMS v2.0: Neural EI-MS Prediction with Knowledge Distillation

次世代の電子衝撃イオン化マススペクトル（EI-MS）予測システム。Teacher-Student Knowledge Distillationと Mixture of Experts (MoE)アーキテクチャを使用して、高精度かつ高速な質量スペクトル予測を実現します。

## 特徴

- **Teacher-Student Knowledge Distillation**: 重いTeacherモデル（GNN+ECFP Hybrid）から軽量Studentモデル（MoE-Residual MLP）への知識転移
- **Mixture of Experts (MoE)**: 4つの専門家ネットワーク（芳香族、脂肪族、複素環、一般）による効率的な予測
- **Uncertainty-Aware Distillation**: MC Dropoutによる不確実性を考慮した知識蒸留
- **Adaptive Loss Weighting**: GradNormによる自動損失バランシング
- **MOL/MSP対応**: MOLファイルとNIST MSP形式の完全サポート
- **RTX 50シリーズ対応**: RTX 5070 Ti (16GB)に最適化
- **Mixed Precision Training**: FP16混合精度訓練による高速化とメモリ効率化

## 性能目標

| メトリック | NEIMS v1.0 | NEIMS v2.0 (目標) | 改善率 |
|--------|------------|------------------|-------|
| Recall@10 | 91.8% | 95.5-96.0% | +3.7-4.2% |
| Recall@5 | ~85% | 90-91% | +5-6% |
| 推論速度 | 5ms | 8-12ms | 1.6-2.4x遅 |
| GPU要件 | 不要 | 不要（推論時） | 同等 |
| モデルサイズ | ~50MB | ~150MB | 3倍 |

## システム要件

### 推論環境（最小要件）
- CPU: 4コア以上
- RAM: 8GB以上
- ストレージ: 500MB以上
- OS: Linux/macOS/Windows

### 学習環境（推奨構成）
- **CPU**: AMD Ryzen 7700 (8コア/16スレッド)
- **GPU**: NVIDIA RTX 5070 Ti (16GB VRAM)
- **RAM**: 32GB DDR4/DDR5
- **ストレージ**: 1TB SSD
- **OS**: Ubuntu 20.04+ / Windows 11 with WSL2
- **CUDA**: 12.8+
- **PyTorch**: 2.7.0+
- **Python**: 3.10+

## インストール

### 方法1: Dev Container (推奨)

```bash
# Visual Studio Codeで開く
# F1 → "Remote-Containers: Reopen in Container"
# 全ての依存関係が自動的にインストールされます
```

詳細は [DEV_CONTAINER_GUIDE.md](DEV_CONTAINER_GUIDE.md) を参照してください。

### 方法2: ローカルインストール

```bash
# リポジトリのクローン
git clone https://github.com/turnDeep/BitSpec.git
cd BitSpec

# PyTorch 2.7.0+ (CUDA 12.8対応)
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch>=2.7.0

# 依存関係のインストール
pip install -r requirements.txt

# パッケージのインストール
pip install -e .
```

## プロジェクト構造

```
BitSpec/
├── config.yaml                    # NEIMS v2.0 設定ファイル
├── config_pretrain.yaml           # Teacher事前学習設定
├── requirements.txt               # 依存関係
├── setup.py                      # パッケージ設定
├── README.md                     # このファイル
├── .devcontainer/                # Dev Container設定
├── data/
│   ├── NIST17.msp                # NIST EI-MSスペクトル（~300,000スペクトル）
│   ├── mol_files/                # 対応するMOLファイル
│   ├── pcqm4mv2/                 # PCQM4Mv2データセット（事前学習用、自動ダウンロード）
│   ├── massbank/                 # MassBank補助データ（オプション）
│   └── processed/                # 前処理済みデータ
├── checkpoints/
│   ├── teacher/                  # Teacherモデル（GNN+ECFP Hybrid）
│   │   ├── pretrained_teacher.pt       # 事前学習済みTeacher
│   │   └── finetuned_teacher.pt        # ファインチューニング済みTeacher
│   └── student/                  # Studentモデル（MoE-Residual MLP）
│       └── distilled_student.pt        # 知識蒸留済みStudent
├── docs/
│   └── NEIMS_v2_SYSTEM_SPECIFICATION.md  # 完全システム仕様書
├── src/
│   ├── data/                     # データ処理
│   │   ├── nist_dataset.py       # NISTデータセット（Teacher/Studentモード対応）
│   │   ├── pcqm4m_dataset.py     # PCQM4Mv2データセット（事前学習用）
│   │   ├── preprocessing.py      # データ前処理ユーティリティ
│   │   ├── augmentation.py       # データ拡張（LDS, Isotope, Conformer）
│   │   ├── mol_parser.py         # MOL/MSPパーサー（レガシー）
│   │   ├── features.py           # 分子特徴量抽出（レガシー）
│   │   └── dataset.py            # データセット（レガシー）
│   ├── models/                   # モデル定義
│   │   ├── teacher.py            # Teacher（GNN+ECFP Hybrid）
│   │   ├── student.py            # Student（MoE-Residual MLP）
│   │   ├── moe.py               # Mixture of Experts
│   │   └── modules.py           # 共通モジュール
│   ├── training/                 # トレーニング
│   │   ├── teacher_trainer.py    # Teacher学習ロジック
│   │   ├── student_trainer.py    # Student知識蒸留ロジック
│   │   ├── losses.py             # 損失関数（KD, Load Balancing, etc.）
│   │   └── schedulers.py         # Temperature/LRスケジューラ
│   ├── evaluation/               # 評価
│   │   ├── metrics.py            # 評価メトリクス（Recall@K, etc.）
│   │   └── visualize.py          # 結果可視化
│   └── utils/
│       ├── chemistry.py          # RDKitユーティリティ
│       └── rtx50_compat.py       # RTX 50互換性
└── scripts/
    ├── train_teacher.py          # Teacher学習（Phase 1-2）
    ├── train_student.py          # Student蒸留（Phase 3）
    ├── train_pipeline.py         # 統合パイプライン ★推奨★
    ├── evaluate.py               # 評価スクリプト
    └── predict.py                # 推論スクリプト
```

## クイックスタート

NEIMS v2.0は3段階の学習プロセスで最高性能を達成します：

### 学習フロー概要

```
Phase 1: Teacher事前学習 (PCQM4Mv2)
   ↓
Phase 2: Teacherファインチューニング (NIST EI-MS)
   ↓
Phase 3: Student知識蒸留 (Teacherから学習)
   ↓
最終モデル: 高精度・高速なStudent (Recall@10: 95.5%+, 推論: 10ms)
```

### 1. データの準備

プロジェクトルートに以下のデータを配置:

```
BitSpec/
├── data/
│   ├── NIST17.msp          # NIST EI-MSスペクトルデータ
│   ├── mol_files/          # 対応するMOLファイル
│   │   ├── ID200001.MOL
│   │   ├── ID200002.MOL
│   │   └── ...
│   └── pcqm4mv2/           # 自動ダウンロードされます
```

**PCQM4Mv2データセット**は初回実行時に自動ダウンロードされます（約3.74M分子、~20GB）。

### 2. 統合パイプラインの実行（推奨）

**完全なワークフロー（3段階）を1コマンドで実行:**

```bash
python scripts/train_pipeline.py --config config.yaml
```

このコマンドは以下を自動的に実行します:
1. **Phase 1**: PCQM4Mv2データセットのダウンロードとTeacher事前学習（Bond Masking）
2. **Phase 2**: NIST EI-MSデータでTeacherをファインチューニング（MC Dropout使用）
3. **Phase 3**: TeacherからStudentへの知識蒸留（Uncertainty-Aware KD）

**推定学習時間（RTX 5070 Ti 16GB）:**
- Phase 1 (Teacher事前学習): ~3-5日（50エポック）
- Phase 2 (Teacherファインチューニング): ~12-18時間（100エポック）
- Phase 3 (Student蒸留): ~8-12時間（150エポック）

**メモリ最適化オプション:**

```bash
# バッチサイズを小さくしてVRAM使用量を削減
python scripts/train_pipeline.py --config config.yaml --batch-size 16

# Mixed Precision（FP16）を使用してメモリ効率を向上
python scripts/train_pipeline.py --config config.yaml --use-amp
```

### 3. 個別ステップの実行（オプション）

統合パイプラインの代わりに、各段階を個別に実行することも可能です:

#### Phase 1: Teacher事前学習（PCQM4Mv2）

```bash
python scripts/train_teacher.py --config config_pretrain.yaml --phase pretrain
```

このステップでは、GNN+ECFP HybridのTeacherモデルがPCQM4Mv2データセット（3.74M分子）でBond Masking タスクを学習します。

#### Phase 2: Teacherファインチューニング（NIST EI-MS）

```bash
python scripts/train_teacher.py --config config.yaml --phase finetune \
    --pretrained checkpoints/teacher/best_pretrain_teacher.pt
```

事前学習済みTeacherをNIST EI-MSデータでファインチューニングし、MC Dropoutによる不確実性推定を有効化します。

**重要**:
- データセットローダー (`NISTDataset`) が自動的にNIST17.mspまたはmol_files/からデータをロード
- 初回実行時は自動的に前処理とキャッシングを実行（`data/processed/`に保存）
- Teacher/Studentモードに応じて異なる特徴量を生成

#### Phase 3: Student知識蒸留

```bash
python scripts/train_student.py --config config.yaml \
    --teacher checkpoints/teacher/best_finetune_teacher.pt
```

MoE-Residual StudentモデルがTeacherから知識を学習します（Uncertainty-Aware KD、GradNorm適応重み付け）。

**データローダーの仕組み**:
- Teacher用データローダー: グラフ + ECFP特徴量でソフトラベル生成（MC Dropout）
- Student用データローダー: ECFP + Count FP特徴量でトレーニング
- 同一データセットの異なる表現を同期的に処理

### 4. 推論

#### 単一分子の予測（Student：高速）

```bash
python scripts/predict.py \
    --config config.yaml \
    --checkpoint checkpoints/student/best_student.pt \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --output aspirin_prediction.msp
```

**出力例**:
```
2025-11-20 12:34:56 - INFO - Predicting spectrum for: CC(=O)OC1=CC=CC=C1C(=O)O

Top 10 peaks:
  1. m/z 180: 0.9876
  2. m/z 138: 0.7654
  3. m/z 120: 0.6543
  ...
```

#### 不確実性付き予測（Teacher：高精度）

```bash
python scripts/predict.py \
    --config config.yaml \
    --checkpoint checkpoints/teacher/best_finetune_teacher.pt \
    --model teacher \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --uncertainty
```

MC Dropoutで不確実性推定を行います（推論時間: ~100ms）。

#### バッチ予測

```bash
# smiles_list.txtに各行1つのSMILES文字列を記載
python scripts/predict.py \
    --config config.yaml \
    --checkpoint checkpoints/student/best_student.pt \
    --batch smiles_list.txt \
    --output predictions/
```

バッチ予測結果は `predictions/batch_predictions/` に保存されます。

## トレーニング戦略

### 3段階学習プロセス

NEIMS v2.0は、段階的な学習で最高性能を達成します:

#### Phase 1: Teacher事前学習（PCQM4Mv2）

```yaml
目的: ロバストな分子表現の学習
データセット: PCQM4Mv2（3.74M分子）
タスク: Bond Masking（自己教師あり学習）
期間: 50エポック（RTX 5070 Ti: ~3-5日）
最適化:
  - Optimizer: AdamW
  - Learning Rate: 1e-4
  - Scheduler: CosineAnnealingWarmRestarts
  - Gradient Clipping: 1.0
```

#### Phase 2: Teacherファインチューニング（NIST EI-MS）

```yaml
目的: スペクトル予測への特化
データセット: NIST17.msp + mol_files（~300K スペクトル）
タスク: MC Dropoutを用いたスペクトル予測
期間: 100エポック（RTX 5070 Ti: ~12-18時間）
最適化:
  - Batch Size: 32
  - Learning Rate: 1e-4
  - MC Dropout Samples: 30（不確実性推定用）
```

#### Phase 3: Student知識蒸留

```yaml
目的: 軽量で高速なモデルへの知識転移
データセット: NIST（Teacherのソフトラベル付き）
期間: 150エポック（RTX 5070 Ti: ~8-12時間）
最適化:
  - Batch Size: 32
  - Learning Rate: 5e-4
  - Scheduler: OneCycleLR
  - GradNorm: 15エポック後に有効化
  - Temperature: 4.0 → 1.0（Cosine Annealing）
```

### データ拡張

- **Label Distribution Smoothing（LDS）**: Gaussian smoothing（σ=1.5 m/z）
- **Isotope Substitution**: C12 → C13（5%の分子に適用）
- **Conformer Generation**: Teacher事前学習のみ（3-5コンフォーマー）

### データセットローダーの実装詳細

#### NISTDataset（`src/data/nist_dataset.py`）

NIST EI-MSデータセット用の統合ローダー。Teacher/Studentモードに応じて異なる特徴量を生成。

**特徴**:
- **MSPファイル解析**: NIST17.msp からスペクトルとメタデータを自動抽出
- **MOLファイル対応**: `data/mol_files/` から分子構造を読み込み
- **2つのモード**:
  - **Teacherモード**: PyG グラフ（48次元ノード、6次元エッジ）+ ECFP4（4096-bit）
  - **Studentモード**: ECFP4（4096-bit）+ Count FP（2048次元）
- **自動キャッシング**: 前処理済みデータを `data/processed/` に保存
- **Train/Val/Test分割**: 8:1:1 の自動分割（seed=42）
- **データ拡張**: LDS smoothing（σ=1.5）をオプションで適用

**使用例**:
```python
from src.data import NISTDataset, collate_fn_teacher

dataset = NISTDataset(
    data_config={'nist_msp_path': 'data/NIST17.msp',
                 'mol_files_dir': 'data/mol_files',
                 'max_mz': 500},
    mode='teacher',    # または 'student'
    split='train',     # または 'val', 'test'
    augment=True       # LDS smoothing有効化
)
```

#### PCQM4Mv2Dataset（`src/data/pcqm4m_dataset.py`）

PCQM4Mv2（3.8M分子）を用いたTeacher事前学習用データセット。

**特徴**:
- **自動ダウンロード**: OGB経由で初回実行時にダウンロード（~20GB）
- **ボンドマスキング**: Self-supervised学習タスク（15%のボンドをマスク）
- **PyG グラフ生成**: マスクされたボンド特徴量を含むグラフ構築
- **Train/Val分割**: 90:10 の自動分割
- **高速キャッシング**: 前処理済みグラフをキャッシュ

**ボンドマスキング**:
```python
mask_ratio = 0.15  # 15%のボンドをマスク
masked_graph, mask_targets = mol_to_graph_with_mask(mol, mask_ratio)
# Teacherは masked_graph からマスクされたボンドの特徴を予測
```

#### preprocessing.py（`src/data/preprocessing.py`）

データ前処理ユーティリティ関数集。

**主要関数**:
- `validate_smiles()`: SMILES検証
- `canonicalize_smiles()`: SMILES正規化
- `filter_by_molecular_weight()`: 分子量フィルタ（50-1000 Da）
- `normalize_spectrum()`: スペクトル正規化（max正規化またはL2正規化）
- `remove_noise_peaks()`: ノイズピーク除去（閾値: 0.001）
- `peaks_to_spectrum_array()`: ピークリスト → ビンドスペクトル変換
- `compute_molecular_descriptors()`: 分子記述子計算（MW, LogP, TPSA, etc.）

### リスク緩和策

| リスク | 確率 | 対策 |
|--------|------|------|
| Expert collapse | 高 | Load balance + Entropy + Bias調整 |
| 訓練不安定 | 中 | Warmup + Gradient clipping + Temperature annealing |
| 過学習 | 中 | Dropout + Weight decay + Data augmentation |
| GPU OOM | 中 | Gradient accumulation + Mixed precision |

詳細は `docs/NEIMS_v2_SYSTEM_SPECIFICATION.md` を参照してください。

## Pythonスクリプトでの使用

### Student モデル（高速推論）

```python
from scripts.predict import SpectrumPredictor

# Studentモデルで高速予測
predictor = SpectrumPredictor(
    config_path='config.yaml',
    checkpoint_path='checkpoints/student/best_student.pt',
    model_type='student',  # 'student' または 'teacher'
    device='cuda'
)

# SMILES文字列から予測（~10ms）
spectrum, _ = predictor.predict_from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O')

# Top peaks検出
peaks = predictor.find_top_peaks(spectrum, top_n=20, threshold=0.01)
print(f"Top 10 peaks: {peaks[:10]}")
# 出力: [(180, 0.9876), (138, 0.7654), ...]

# MSP形式でエクスポート
predictor.export_msp(
    smiles='CC(=O)OC1=CC=CC=C1C(=O)O',
    output_path='aspirin.msp',
    compound_name='Aspirin'
)

# バッチ予測
smiles_list = ['CCO', 'CC(C)O', 'c1ccccc1']
spectra = predictor.predict_batch(smiles_list, batch_size=32)
print(f"Predicted {len(spectra)} spectra")
```

### Teacher モデル（不確実性推定付き）

```python
# Teacherモデルで不確実性推定
teacher_predictor = SpectrumPredictor(
    config_path='config.yaml',
    checkpoint_path='checkpoints/teacher/best_finetune_teacher.pt',
    model_type='teacher',
    device='cuda'
)

# MC Dropoutで不確実性推定（~100ms）
spectrum, uncertainty = teacher_predictor.predict_from_smiles(
    'CC(=O)OC1=CC=CC=C1C(=O)O',
    return_uncertainty=True
)

print(f"Mean uncertainty: {uncertainty.mean():.4f}")
```

### データセットの直接使用

```python
from src.data import NISTDataset, collate_fn_student
from torch.utils.data import DataLoader

# NIST EI-MSデータセット（Studentモード）
dataset = NISTDataset(
    data_config={'nist_msp_path': 'data/NIST17.msp',
                 'mol_files_dir': 'data/mol_files',
                 'max_mz': 500},
    mode='student',  # 'teacher' または 'student'
    split='train',   # 'train', 'val', 'test'
    augment=True
)

# DataLoader作成
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn_student
)

# イテレーション
for batch in loader:
    # batch['input']: ECFP + Count FP [batch, 6144]
    # batch['spectrum']: Target spectrum [batch, 501]
    ...
```

## モデルアーキテクチャ

### 全体構成

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Teacher Model   │         │  Student Model   │         │
│  │  (GNN + ECFP)    │────────▶│  (MoE-Residual)  │         │
│  │                  │ KD      │                  │         │
│  │  - GINEConv x8   │         │  - 4 Experts     │         │
│  │  - Bond-Breaking │         │  - Residual MLP  │         │
│  │  - MC Dropout    │         │  - Gate Network  │         │
│  └──────────────────┘         └──────────────────┘         │
│         ▲                              ▲                    │
│         │                              │                    │
│         └──────────────┬───────────────┘                    │
│                        │                                    │
│                 ┌──────▼──────┐                            │
│                 │  NIST17.msp │                            │
│                 │  mol_files  │                            │
│                 │  PCQM4Mv2   │                            │
│                 └─────────────┘                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Molecule → ECFP4 + Count FP → Student Model         │
│                                          ↓                   │
│                                    Mass Spectrum             │
│                                    (8-12ms latency)          │
└─────────────────────────────────────────────────────────────┘
```

### Teacher Model（訓練のみ使用）

- **GNN Branch**: GINEConv x 8層（hidden_dim: 256）
  - Bond-Breaking Attention
  - DropEdge (p=0.2) + PairNorm
  - Global Pooling（Mean + Max + Attention）
- **ECFP Branch**: ECFP4（4096-dim）→ MLP（512-dim）
- **Fusion**: GNN (768-dim) + ECFP (512-dim) = 1280-dim
- **Prediction Head**: 1280 → 1024 → 512 → 501（m/z 0-500）
- **MC Dropout**: 30サンプルで不確実性推定

**パラメータ数**: ~15M | **推論速度**: ~100ms

### Student Model（本番使用）

- **Input**: ECFP4（4096-dim）+ Count FP（2048-dim）= 6144-dim
- **Gate Network**: 6144 → 512 → 128 → 4（Top-2 Routing）
- **Expert Networks（x4）**: 各エキスパートは6つのResidual Blockで構成
  - Expert 1: 芳香族化合物
  - Expert 2: 脂肪族化合物
  - Expert 3: 複素環化合物
  - Expert 4: 一般/混合
- **Fusion**: Expert出力の重み付き結合
- **Prediction Head**: 6144 → 2048 → 1024 → 501（m/z 0-500）

**パラメータ数**: ~50M | **推論速度**: ~10ms | **モデルサイズ**: ~200MB

## 損失関数

### Teacher Training Loss

```python
L_teacher = L_spectrum + λ_bond * L_bond_masking

L_spectrum = MSE(predicted_spectrum, target_spectrum)
L_bond_masking = CrossEntropy(predicted_masked_bonds, true_masked_bonds)
```

### Student Training Loss（完全版）

```python
L_student = (α * L_hard +           # Hard Label Loss (NIST Ground Truth)
             β * L_soft +            # Soft Label Loss (Teacher with Uncertainty)
             γ * L_feature +         # Feature-Level Distillation
             δ_load * L_load +       # Load Balancing Loss
             δ_entropy * L_entropy)  # Entropy Regularization
```

#### 主要損失の詳細

1. **L_hard**: NIST Ground Truthとの直接比較（MSE）
2. **L_soft**: Teacherのソフトラベルとの比較（Confidence-Weighted MSE、Temperature Annealing）
3. **L_feature**: Teacher-Student中間表現のアライメント
4. **L_load**: MoEエキスパートの負荷分散（Switch Transformer方式）
5. **L_entropy**: ゲートネットワークのエントロピー正則化

### GradNorm適応重み付け

- **Warmup期間（15エポック）**: 固定重み（α=0.3, β=0.5, γ=0.2）
- **GradNorm期間（15エポック以降）**: 勾配ノルムに基づく動的調整
- **Temperature Annealing**: T_init=4.0 → T_min=1.0（Cosineスケジュール）

## 評価メトリクス

### 主要メトリクス

- **Recall@K**: Top-Kピークの一致率（K=5, 10, 20）
  - **目標**: Recall@10 ≥ 95.5%（ベースラインNEIMS: 91.8%）
- **Spectral Similarity (Cosine)**: スペクトル全体の類似度
- **MAE/RMSE**: ピーク強度の予測誤差

### 効率性メトリクス

- **推論時間**: 平均ms/分子（目標: ≤ 10ms）
- **スループット**: 分子数/秒
- **メモリ使用量**: ピークGPU/RAM消費
- **モデルサイズ**: ディスク容量（MB）

### 専門メトリクス（NEIMS v2.0）

- **Expert Usage Distribution**: 各エキスパートの使用頻度
- **MC Dropout Uncertainty**: Teacher予測の不確実性
- **KD Transfer Efficiency**: Teacherから Studentへの知識転移効率

## ハードウェア最適化

### RTX 5070 Ti（16GB VRAM）向け最適化

本プロジェクトは RTX 5070 Ti に最適化されています:

- **CUDA 12.8+**: 最新CUDA Toolkitによる最適化
- **Mixed Precision Training**: FP16による高速化とメモリ効率化
- **Gradient Accumulation**: 実効バッチサイズを維持しながらメモリ削減
- **PyTorch 2.7.0+**: sm_120アーキテクチャの公式サポート

### メモリ使用量の目安（RTX 5070 Ti 16GB）

| Phase | バッチサイズ | VRAM使用量 | 推奨設定 |
|-------|------------|----------|---------|
| Teacher事前学習 | 128 | ~14GB | batch_size=128, use_amp=true |
| Teacherファインチューニング | 32 | ~12GB | batch_size=32, use_amp=true |
| Student蒸留 | 32 | ~10GB | batch_size=32, use_amp=true |
| 推論（Student） | 1 | ~2GB | CPU推論も可能 |

### CPU/RAM最適化（Ryzen 7700、32GB RAM）

- **データローダー**: num_workers=8（8コア/16スレッド活用）
- **事前処理**: マルチプロセス並列化
- **メモリ管理**: pin_memory=true（CUDA転送高速化）

詳細は `src/utils/rtx50_compat.py` および `config.yaml` を参照。

## データ形式

### NIST MSP形式

```
Name: Aspirin
InChIKey: BSYNRYMUTXBXSQ-UHFFFAOYSA-N
Formula: C9H8O4
MW: 180
ID: 200001
Num peaks: 15
41 100.0
55 50.0
69 25.0
...
180 999.0

```

### MOLファイル

標準のMOL V2000/V3000形式に対応。`data/mol_files/` ディレクトリにID付きで配置:

```
data/mol_files/
├── ID200001.MOL
├── ID200002.MOL
└── ...
```

MSPファイルのIDとMOLファイル名のIDが対応している必要があります。

## トラブルシューティング

### GPU が認識されない

```bash
# CUDAの確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# RTX 50対応の確認
python -c "from src.utils.rtx50_compat import setup_gpu_environment; setup_gpu_environment()"
```

### メモリ不足エラー

`config.yaml` でバッチサイズを調整:

```yaml
training:
  batch_size: 16  # 32から16に減らす
```

または `hidden_dim` を減らす:

```yaml
model:
  hidden_dim: 128  # 256から128に減らす
```

### データが見つからない

```bash
# データの配置を確認
ls -la data/NIST17.MSP
ls -la data/mol_files/ | head

# MOLファイルとMSPのIDマッピングを確認
python scripts/test_mol_nist_mapping.py
```

## コンソールスクリプト

パッケージインストール後、以下のコマンドが使用可能:

```bash
ms-train --config config.yaml         # トレーニング
ms-predict --checkpoint model.pt ...  # 予測
ms-evaluate --checkpoint model.pt ... # 評価
```

## 開発ツール

```bash
# コードフォーマット
black src/ scripts/

# 型チェック
mypy src/

# テスト実行
pytest
```

## 参考文献

### NEIMS v2.0 関連

1. **NEIMS v1.0**: Wei et al., "Rapid Prediction of Electron-Ionization Mass Spectrometry Using Neural Networks", *ACS Central Science*, 2019
2. **GLNNs**: Zhang et al., "Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation", *ICLR*, 2021
3. **Switch Transformers (MoE)**: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models", *JMLR*, 2022
4. **GradNorm**: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing", *ICML*, 2018
5. **MC Dropout**: Gal & Ghahramani, "Dropout as a Bayesian Approximation", *ICML*, 2016
6. **Uncertainty-Aware KD**: "Teaching with Uncertainty: Unleashing the Potential of Knowledge Distillation", *CVPR*, 2024
7. **FIORA**: "Local neighborhood-based prediction of compound mass spectra", *Nature Communications*, 2025
8. **MolCLR**: Wang et al., "Molecular Contrastive Learning of Representations via Graph Neural Networks", *Nature MI*, 2022

### データセット

- **NIST EI-MS**: National Institute of Standards and Technology Mass Spectral Library
- **PCQM4Mv2**: OGB Large-Scale Challenge Dataset（3.74M molecules）
- **MassBank**: Community Mass Spectrometry Database

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずIssueを開いて変更内容を議論してください。

## お問い合わせ

- **GitHub Issues**: https://github.com/turnDeep/BitSpec/issues
- **プロジェクトURL**: https://github.com/turnDeep/BitSpec

## 更新履歴

- **v2.0.1** (2025-11-20): 完全データセット統合とトレーニングパイプライン実装
  - **データセットローダー完全実装**:
    - `NISTDataset`: NIST EI-MS データローダー（Teacher/Studentモード対応）
    - `PCQM4Mv2Dataset`: PCQM4Mv2 事前学習データセット（ボンドマスキング）
    - `preprocessing.py`: データ前処理ユーティリティ（正規化、フィルタリング、統計計算）
  - **トレーニングスクリプト統合**:
    - `train_teacher.py`: Phase 1-2完全統合（PCQM4Mv2 → NIST EI-MS）
    - `train_student.py`: Phase 3知識蒸留完全統合（デュアルデータローダー）
    - `train_pipeline.py`: 3フェーズ統合パイプライン（自動チェックポイント管理）
  - **推論スクリプト更新**:
    - `predict.py`: Student/Teacher両対応、不確実性推定サポート
  - **完全動作可能**: エンドツーエンドトレーニング・推論パイプライン完成
  - MSPファイル解析、PyG グラフ構築、ECFP/Count FP 生成完全実装
  - データキャッシング、ハードウェア最適化（8 workers, prefetch_factor=4）

- **v2.0.0** (2025-11-20): NEIMS v2.0 完全リアーキテクチャ
  - **Teacher-Student Knowledge Distillation**: GNN+ECFP Teacher → MoE-Residual Student
  - **Mixture of Experts (MoE)**: 4エキスパート（芳香族、脂肪族、複素環、一般）
  - **Uncertainty-Aware Distillation**: MC Dropout + Confidence-Weighted KD
  - **Adaptive Loss Weighting**: GradNorm + Temperature Annealing
  - **3段階学習**: Teacher事前学習 → Teacherファインチューニング → Student蒸留
  - **ハードウェア最適化**: RTX 5070 Ti (16GB) + Ryzen 7700 + 32GB RAM
  - **目標性能**: Recall@10 ≥ 95.5%、推論時間 ≤ 10ms
  - 完全システム仕様書追加（`docs/NEIMS_v2_SYSTEM_SPECIFICATION.md`）
  - モデルアーキテクチャ、損失関数、トレーナー実装完了

- **v1.3.0** (2025-11): 統合パイプライン追加
  - `train_pipeline.py`: PCQM4Mv2ダウンロード→事前学習→ファインチューニング
  - PCQM4Mv2自動ダウンロード機能（OGBライブラリ経由）
  - 1コマンドで完全なワークフローを実行可能に

- **v1.2.0** (2025-11): PCQM4Mv2事前学習対応
  - PCQM4Mv2データセットでの事前学習機能追加
  - ファインチューニングスクリプト実装
  - 転移学習のための凍結戦略サポート

- **v1.1.0** (2025-11): 特徴量最適化
  - 原子特徴量を157次元→48次元に最適化
  - 結合特徴量を16次元→6次元に最適化
  - WeightedCosineLossに統一

- **v1.0.0** (2024): 初回リリース
  - GCNベースのマススペクトル予測モデル
  - RTX 50シリーズ対応
  - MOL/MSP完全サポート
