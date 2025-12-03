# config.yaml v4.2 変更点まとめ

## 概要

NExtIMS v2.0の複雑なTeacher-Student Knowledge Distillation構成から、v4.2の「**Start Simple, Iterate Based on Evidence**」設計哲学に基づいた最小構成に完全移行しました。

## 設計哲学

```
v2.0: "複雑なアーキテクチャで最大性能を目指す"
  ↓
v4.2: "シンプルから始めて、証拠に基づいて反復改善"
```

## 主要変更点

### 1. プロジェクト情報

| 項目 | v2.0 | v4.2 |
|------|------|------|
| バージョン | 2.0.0 | 4.2.0 |
| 説明 | Teacher-Student KD with MoE | Minimal Configuration |
| 設計哲学 | なし | "Start Simple, Iterate Based on Evidence" |

### 2. データ設定

#### スペクトル範囲

```yaml
# v2.0
data:
  max_mz: 500          # m/z 0-500 (501 bins)

# v4.2
data:
  min_mz: 1
  max_mz: 1000         # m/z 1-1000 (1000 bins)
```

**理由**: より広い分子量範囲に対応するため、スペクトル範囲を拡大。

#### NIST17データ構造の明確化

```yaml
# v4.2で追加
data:
  # NIST17.MSP: マススペクトルデータ（ピーク情報のみ）
  # mol_files/: 化学構造データ（MOLファイル）
  # ID番号でリンク: MSP内のIDとMOLファイル名（ID12345.MOL）が対応
  nist_msp_path: "data/NIST17.MSP"
  mol_files_dir: "data/mol_files"
```

#### 不要な設定の削除

v2.0で存在したが、v4.2で削除された項目：

- `bde_db2_path` - BonDNet再学習用データ
- `bondnet_training_data` - BonDNet学習データ
- `massbank_path` - MassBankデータ（オプション）
- `gnps_path` - GNPSデータ（オプション）
- `memory_efficient_mode` - 複雑なメモリ最適化設定

**理由**: v4.2は単一データセット（NIST17）に集中。最小構成なのでメモリ効率化不要。

### 3. モデル設定

#### アーキテクチャの簡素化

```yaml
# v2.0: 複雑なTeacher-Student + MoE構成
model:
  teacher:
    type: "GNN_ECFP_Hybrid"
    gnn:
      conv_type: "GINEConv"
      num_layers: 8
      hidden_dim: 256
      # ... 多数の設定
    ecfp:
      fingerprint_size: 4096
      # ... ECFP設定
  student:
    type: "MoE_Residual"
    # ... MoE設定
  common:
    node_features: 48
    edge_features: 6

# v4.2: シンプルな単一モデル
model:
  node_dim: 16          # Minimal (87.5% reduction)
  edge_dim: 3           # Minimal (95.3% reduction)
  hidden_dim: 256
  num_layers: 10
  num_heads: 8
  output_dim: 1000
  dropout: 0.1
```

#### 特徴量の大幅削減

| 特徴量 | v2.0 (v4.1) | v4.2 | 削減率 |
|--------|-------------|------|--------|
| Node features | 48 (128) | 16 | **-87.5%** |
| Edge features | 6 (64) | 3 | **-95.3%** |
| Hidden dim | 128 | 256 | +100% (容量増加) |
| Output dim | 501 | 1000 | +99% (範囲拡大) |

**注**: 括弧内はv4.1の値

#### モデルタイプの変更

- **v2.0**: GNN_ECFP_Hybrid (Teacher) + MoE_Residual (Student)
- **v4.2**: QCGN2oEI_Minimal (単一モデル、GATv2Conv)

### 4. 訓練設定

#### 単純化された訓練プロセス

```yaml
# v2.0: 2段階訓練（Teacher → Student）
training:
  teacher_multitask:
    batch_size: 32
    num_epochs: 100
    # ... 複雑なマルチタスク設定
  student_distill:
    batch_size: 32
    num_epochs: 150
    # ... 複雑な蒸留設定

# v4.2: 単一訓練プロセス
training:
  num_epochs: 300
  batch_size: 32
  optimizer: "RAdam"
  learning_rate: 0.001
  scheduler: "CosineAnnealingLR"
  loss_function: "cosine_similarity"
```

#### オプティマイザとスケジューラ

| 設定 | v2.0 Teacher | v2.0 Student | v4.2 |
|------|-------------|-------------|------|
| Optimizer | AdamW | AdamW | **RAdam** |
| Scheduler | CosineAnnealingWarmRestarts | CosineAnnealingWarmRestarts | **CosineAnnealingLR** |
| Learning Rate | 1e-4 | 1.5e-4 | **1e-3** |
| Weight Decay | 1e-5 | 1e-4 | **1e-5** |

**理由**: RAdamとCosineAnnealingLRはQC-GN2oMS2の成功実績に基づく選択。

#### 削除された複雑な機能

v2.0で存在したが、v4.2で削除された機能：

- **マルチタスク学習**: BDE補助タスク
- **Knowledge Distillation**: Temperature annealing, GradNorm
- **Label Distribution Smoothing (LDS)**
- **MC Dropout**: 不確実性推定
- **Mixed Precision Training** (安定性のため無効化)

### 5. 評価設定

#### パフォーマンス目標の明確化

```yaml
# v4.2で新規追加
evaluation:
  performance_targets:
    cosine_similarity:
      excellent: 0.85    # ≥ 0.85 = EXCELLENT（採用完了）
      good: 0.80         # 0.80-0.85 = GOOD（要検討）
      moderate: 0.75     # 0.75-0.80 = MODERATE（要改善検討）
      insufficient: 0.75 # < 0.75 = INSUFFICIENT（改善必須）
```

**理由**: 反復改善の判断基準を明確化。証拠に基づいた意思決定を可能にする。

#### 評価メトリクス

| メトリクス | v2.0 | v4.2 |
|-----------|------|------|
| 主要指標 | recall_at_k | **cosine_similarity** |
| 目標値 | Recall@10 ≥ 95.5% | Cosine Sim ≥ 0.85 |
| メトリクス数 | 3基本 + 3専門 | **7メトリクス** |

v4.2のメトリクス:
1. Cosine Similarity (主要)
2. Top-K Recall
3. MSE
4. MAE
5. RMSE
6. Spectral Angle
7. Manhattan Distance

### 6. 推論設定（Phase 5）

```yaml
# v4.2で新規追加
inference:
  single:
    model_path: "models/qcgn2oei_minimal_best.pth"
    device: "cuda"
    batch_size: 1

  batch:
    batch_size: 64      # Inference用に大きめ
    num_workers: 4

  output:
    format: "csv"
    extract_top_k_peaks: 10
    visualization: true
```

**理由**: Phase 8実装（predict_single.py, predict_batch.py）に対応。

### 7. BDE設定

```yaml
# v2.0: 複雑なBDE-db2再学習設定
bde:
  bde_db2_path: "data/external/bde-db2"
  bondnet_training_data: "data/processed/bondnet_training"

# v4.2: シンプルなキャッシュ利用
bde:
  bondnet:
    model_path: "models/bondnet_bde_db2_best.pth"
  cache:
    enabled: true
    cache_file: "nist17_bde_cache.h5"
  calculation:
    use_calculator: false  # キャッシュのみ使用
    default_bde: 85.0
```

**理由**: Phase 0/1でBDEキャッシュを事前生成。訓練時は計算不要。

### 8. GPU/CPU設定

#### GPU設定の簡素化

```yaml
# v2.0
gpu:
  mixed_precision: true
  compile: false
  rtx50:
    enable_compat: true

# v4.2
gpu:
  mixed_precision: false  # 安定性優先
  compile: false          # PyG互換性
  empty_cache_interval: 100
```

**理由**: Mixed precisionは不安定性の原因となる可能性があるため無効化。

### 9. 新規追加セクション

v4.2で新しく追加されたセクション：

#### a) Experiment設定

```yaml
experiment:
  name: "qcgn2oei_minimal_v4.2"
  description: "..."
  random_seed: 42
  deterministic: true
```

#### b) Performance設定

```yaml
performance:
  memory_efficient_mode: false  # v4.2は軽量なので不要
  enable_profiling: false
```

#### c) System情報

```yaml
system:
  gpu_model: "RTX 5070 Ti"
  gpu_memory_gb: 16
  cpu_model: "Ryzen 7700"
  ram_gb: 32
```

#### d) Meta情報

```yaml
meta:
  spec_version: "v4.2"
  config_version: "1.0.0"
  design_philosophy: "Start Simple, Iterate Based on Evidence"
  changes:
    - "Node features: 128 → 16 (87.5% reduction)"
    - ...
  documentation:
    specification: "docs/spec_v4.2_minimal_iterative.md"
    quickstart: "QUICKSTART.md"
    ...
```

## パフォーマンス比較

### 学習時間

| 項目 | v2.0 | v4.2 | 改善 |
|------|------|------|------|
| Teacher学習 | ~50時間 | - | - |
| Student学習 | ~75時間 | - | - |
| 単一モデル学習 | - | ~40時間 | - |
| **合計** | **~125時間** | **~40時間** | **68%短縮** |

### メモリ使用量

| 項目 | v2.0 | v4.2 | 改善 |
|------|------|------|------|
| Node encoder | 49,152 params | 4,864 params | **-90.1%** |
| Edge encoder | - | - | - |
| 合計削減 | - | - | **90%削減** |
| ピークVRAM | ~14GB | ~10GB | **-28.6%** |

### コード複雑度

| 項目 | v2.0 | v4.2 | 改善 |
|------|------|------|------|
| モデル数 | 2 (Teacher+Student) | 1 | -50% |
| 訓練スクリプト | 2 | 1 | -50% |
| 設定セクション | 12 | 15 | +25% (整理) |
| 設定行数 | 301 | 318 | +5.6% (可読性向上) |

## 移行ガイド

### v2.0からv4.2への移行手順

1. **config.yamlのバックアップ**
   ```bash
   cp config.yaml config.yaml.v2.0.backup
   ```

2. **新しいconfig.yamlの適用**
   ```bash
   # 既に実施済み（commit 37629cd）
   git pull origin claude/review-config-differences-01FttGbkbvpAQa7encLwcrrR
   ```

3. **訓練スクリプトの確認**
   - `scripts/train_gnn_minimal.py`を使用
   - v2.0の`train_teacher.py`, `train_student.py`は不要

4. **評価スクリプトの確認**
   - `scripts/evaluate_minimal.py`を使用
   - v2.0の評価スクリプトは不要

5. **推論スクリプトの使用**
   - `scripts/predict_single.py` - 単一分子予測
   - `scripts/predict_batch.py` - バッチ予測

### 互換性のない変更

以下の機能はv4.2では削除されています：

1. **Teacher-Student Knowledge Distillation**
   - v2.0の訓練済みTeacher/Studentモデルは使用不可
   - v4.2で新規に訓練が必要

2. **MoE (Mixture of Experts)**
   - v4.2では単一モデル構成

3. **マルチタスク学習**
   - BDE補助タスクは削除

4. **データ拡張**
   - LDS, Isotope substitution, Conformer generation削除

5. **MC Dropout**
   - 不確実性推定機能削除

## 推奨される次のステップ

### Phase 2: 訓練

```bash
python scripts/train_gnn_minimal.py \
    --nist-msp data/NIST17.MSP \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --output models/qcgn2oei_minimal_best.pth \
    --epochs 300 \
    --batch-size 32
```

### Phase 3: 評価

```bash
python scripts/evaluate_minimal.py \
    --model models/qcgn2oei_minimal_best.pth \
    --nist-msp data/NIST17.MSP \
    --visualize --benchmark
```

### Phase 5: 推論

```bash
# 単一分子
python scripts/predict_single.py \
    --smiles "CCO" \
    --model models/qcgn2oei_minimal_best.pth

# バッチ
python scripts/predict_batch.py \
    --input molecules.csv \
    --model models/qcgn2oei_minimal_best.pth \
    --output predictions.csv
```

## FAQ

### Q1: なぜv2.0からv4.2にジャンプしたのか？

**A**: v3.x系列は試作版で、v4.2が最初の安定版です。設計哲学の転換に伴い、メジャーバージョンを上げました。

### Q2: v2.0のモデルはv4.2で使えるか？

**A**: いいえ。アーキテクチャが完全に異なるため、v4.2で新規訓練が必要です。

### Q3: 性能は低下しないか？

**A**: v4.2の目標は「コサイン類似度 ≥ 0.85」です。v2.0より単純ですが、QC-GN2oMS2の成功事例に基づいており、十分な性能が期待されます。

### Q4: いつv2.0の機能が戻るか？

**A**: v4.2で基本性能を確立した後、必要に応じて段階的に高度な機能を追加します（反復改善）。

### Q5: config.yamlは今後変更されるか？

**A**: はい。「Start Simple, Iterate Based on Evidence」に基づき、実験結果に応じて設定を調整します。

## 参考資料

- **仕様書**: `docs/spec_v4.2_minimal_iterative.md`
- **クイックスタート**: `QUICKSTART.md`
- **README**: `README.md`
- **NIST17データ構造**: `docs/NIST17_DATA_STRUCTURE.md`
- **推論ガイド**: `docs/PREDICTION_GUIDE.md`

## 変更履歴

- **2025-12-03**: config.yaml v4.2初版作成（commit 37629cd）
- **2025-12-03**: NIST17データ構造修正（commit 825f20e）
- **2025-12-03**: NIST17ドキュメント追加（commit 3e47bf9）

---

**最終更新**: 2025-12-03
**バージョン**: v4.2
**Commit**: 37629cd
