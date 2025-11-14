# ファインチューニング最適化レポート

## 問題点の分析

### 1. 処理速度の変動（6.31it/s ～ 47.56it/s）
- **原因**：グラフサイズの変動によるCUDAメモリアロケーターの頻繁な再配置
- **原因**：DataLoaderのワーカーが各エポックで再起動されている
- **原因**：`persistent_workers`と`prefetch_factor`が未設定

### 2. Lossのサチュレーション（0.43-0.44で停滞）
- **原因**：学習率のWarmup phaseが無い
- **原因**：実効バッチサイズが小さい（64）
- **原因**：学習率スケジューリングが最適化されていない

## 実装した改善策（2025年最新ベストプラクティス）

### 1. DataLoader最適化（処理速度安定化・32GB RAM対応）

#### `src/data/dataset.py`の変更
```yaml
persistent_workers: true  # ワーカーをエポック間で再利用
prefetch_factor: 2        # 先読みバッチ数（32GB RAM制約下）
num_workers: 4            # ワーカー数（32GB RAM制約下）
```

**期待される効果**：
- エポック開始時のワーカー起動コスト削減
- データローディングとGPU計算のオーバーラップ向上
- 処理速度の安定化（変動幅の縮小）
- **32GBシステムメモリでの安全な動作**

**メモリ消費削減**：
- ワーカープリフェッチメモリ：4 workers × 2 batches × 64 samples = **512サンプル常駐**
- 以前の設定（8 × 4 × 128 = 4096）と比較して**87.5%削減**

**根拠**：
- PyTorch Lightning公式ドキュメント（2024）
- "8 PyTorch DataLoader Tactics to Max Out Your GPU" (2025)

### 2. 学習率スケジューリング最適化（Loss plateau突破）

#### OneCycleLR導入
```yaml
scheduler_type: "onecycle"
warmup_pct: 0.3           # 30%をwarmupに使用
div_factor: 25.0          # 初期学習率制御
final_div_factor: 10000.0 # 最終学習率制御
```

**期待される効果**：
- より高い学習率での安定した訓練
- 学習初期の勾配爆発防止
- Loss plateauからの脱出

**根拠**：
- "Super Convergence Cosine Annealing with Warm-Up" (IEEE 2024)
- "Why Warmup the Learning Rate? Underlying Mechanisms" (arXiv 2024)

### 3. ReduceLROnPlateau導入（適応的学習率調整）

```yaml
use_reduce_on_plateau: true
plateau_patience: 5
```

**期待される効果**：
- Validation lossが改善しない場合に自動的に学習率を下げる
- より細かい最適化が可能

### 4. Stochastic Weight Averaging (SWA)

```yaml
use_swa: true
swa_start_epoch: 30       # 50エポック中30エポック目から開始
swa_lr: 0.0001
```

**期待される効果**：
- 汎化性能の向上（テストlossの改善）
- より平坦な損失面での収束
- アンサンブル効果

**根拠**：
- PyTorch SWA公式実装
- 多数の論文で汎化性能向上を実証

### 5. Gradient Accumulation増加（32GB RAM対応）

```yaml
batch_size: 64                     # メモリ効率重視
gradient_accumulation_steps: 4     # 1 → 4
num_workers: 4                     # 8 → 4（32GB制約）
prefetch_factor: 2                 # 4 → 2（メモリ消費抑制）
# 実効バッチサイズ: 64 × 4 = 256
```

**期待される効果**：
- より安定した勾配推定
- Loss plateauの克服
- **32GBシステムメモリでも安全に動作**
- メモリ使用量を約75%削減（ワーカープリフェッチ: 4096サンプル → 512サンプル）

**根拠**：
- 大規模バッチは収束を安定化（多数の研究で実証）
- Gradient accumulationはメモリ効率的に大バッチ効果を実現

### 6. モデル正則化強化

```yaml
dropout: 0.15              # 0.1 → 0.15
backbone_lr: 0.0001        # 0.00005 → 0.0001（より積極的）
```

**期待される効果**：
- 過学習の抑制
- 汎化性能の向上
- バックボーンのより積極的な更新

### 7. 改善されたWarmup実装

`scripts/finetune.py`に以下を実装：
- エポックベースのlinear warmup
- OneCycleLRとの統合
- より滑らかな学習率遷移

## 技術的根拠（Web検索結果より）

### GNN最適化（2024-2025研究）
1. **DenseGNN**: Dense connectivity networksによる計算コスト削減
2. **Skip connections**: 訓練速度の向上（任意の深さで効果）
3. **Adaptive finetuning**: S2PGNN、GTOT-Tuningなどの最新手法

### DataLoader最適化
1. **persistent_workers**: DDP使用時のボトルネック解消
2. **pin_memory + non_blocking**: CPU→GPU転送の高速化
3. **prefetch_factor**: データローディングとGPU計算のオーバーラップ

### 学習率最適化
1. **WuC-Adam**: Warmup + Cosine annealingの組み合わせ（2024）
2. **OneCycleLR**: より高い学習率での訓練を可能に
3. **Warmup**: 深い層の訓練不安定性を防止

## 期待される改善効果

### 処理速度
- **現状**: 6.31～47.56 it/s（変動大）
- **改善後予測**: 20～30 it/s（安定）
- **エポック時間**: 約2-3分/エポック（安定化）

### Loss収束
- **現状**: Val Loss 0.43-0.44で停滞
- **改善後予測**: Val Loss 0.35-0.40まで改善可能
- **根拠**:
  - より大きい実効バッチサイズ（256）
  - OneCycleLRによる効果的な学習率スケジューリング
  - SWAによる汎化性能向上

### GPU利用率
- **現状**: 変動的
- **改善後予測**: より安定した高利用率
- **根拠**: persistent_workers + prefetch_factorによるデータローディング最適化

## 使用方法

```bash
# 最適化された設定でファインチューニングを実行
python scripts/finetune.py --config config_pretrain.yaml

# キャッシュを再構築する場合
python scripts/finetune.py --config config_pretrain.yaml --rebuild-cache
```

## メモリ使用量の見積もり（32GB RAM対応版）

### 現在の設定（最適化版）
```yaml
batch_size: 64
num_workers: 4
prefetch_factor: 2
gradient_accumulation_steps: 4
```

**推定メモリ消費**：
- 30万化合物キャッシュ：約2-4GB
- ワーカープリフェッチ：4 workers × 2 batches × 64 samples = **512サンプル** → 約3-5GB
- GPUメモリ（RTX 5070 Ti 16GB）：バッチサイズ64で約8-10GB
- **総システムメモリ消費**：約10-15GB（**32GBで安全**）

### もしメモリ不足が発生する場合

さらにメモリを節約する設定：

```yaml
finetuning:
  batch_size: 48                   # 64 → 48
  num_workers: 2                   # 4 → 2
  prefetch_factor: 2               # 維持
  gradient_accumulation_steps: 5   # 4 → 5（実効240）
  persistent_workers: false        # 必要に応じてfalse
```

## 代替設定（より保守的なアプローチ）

より保守的な学習戦略を希望する場合：

```yaml
finetuning:
  scheduler_type: "cosine_warmup"  # OneCycleLRの代わり
  use_swa: false                   # SWA無効化
  use_reduce_on_plateau: false     # ReduceLROnPlateau無効化
```

## モニタリング推奨事項

### 必須モニタリング項目

1. **システムメモリ使用率**（重要！）
   ```bash
   # 別ターミナルで実行（リアルタイム監視）
   watch -n 1 free -h

   # またはhtopでより詳細に
   htop
   ```
   - 使用率が80%を超えたら設定を見直し
   - スワップが発生していないか確認

2. **GPU使用率とメモリ**
   ```bash
   watch -n 1 nvidia-smi
   ```
   - GPUメモリ使用率を監視
   - GPU利用率（Utilization）を確認

3. **処理速度**: 各エポックのit/s値を監視
   - 安定して15-25 it/s程度なら良好
   - 大きな変動（10倍以上）は問題あり

4. **学習率**: ログで学習率の遷移を確認
   - OneCycleLRの場合、warmup→peak→annealingの遷移を確認

5. **Loss**: Train/Val lossの差を監視（過学習チェック）
   - Validation lossが改善しているか確認

## 参考文献

1. DenseGNN (Nature 2024): https://www.nature.com/articles/s41524-024-01444-x
2. PyTorch DataLoader Optimization (2025): Medium article
3. WuC-Adam (MBE 2024): https://www.aimspress.com/article/doi/10.3934/mbe.2024054
4. Learning Rate Warmup (arXiv 2024): https://arxiv.org/html/2406.09405v1
5. PyTorch Geometric Community Best Practices

## 変更ファイル一覧

1. `src/data/dataset.py` - DataLoader最適化
2. `scripts/finetune.py` - スケジューラー、SWA、Warmup実装
3. `config_pretrain.yaml` - 設定パラメータ更新

---

作成日: 2025-11-13
最終更新: 2025-11-13
