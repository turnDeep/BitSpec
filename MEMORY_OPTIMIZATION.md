# メモリ最適化ガイド

## 問題の背景

ファインチューニング中にメモリ消費量が100%に達し、Dockerコンテナがリロードする問題が発生していました。プレトレーニングでは問題がないにもかかわらず、ファインチューニングで発生する理由を調査し、解決策を実装しました。

## 原因分析

### 1. データローダーのプリフェッチ過剰（最大の原因）

**問題点:**
- `num_workers: 8` × `prefetch_factor: 4` × `batch_size: 128` = 4,096サンプルを常時CPUメモリに保持
- `persistent_workers: true` により、エポック間でもワーカープロセスが保持され、メモリが解放されない
- プレトレーニングは `persistent_workers: false` だったため、エポック毎にメモリが解放されていた

**メモリ使用量:**
- 各サンプルにはグラフ構造（ノード、エッジ、特徴量）とスペクトルデータが含まれる
- 4,096サンプル × 平均サイズ ≈ 数GB のCPUメモリを消費

### 2. スペクトル予測ヘッドの大きな中間層

**問題点:**
- `spectrum_predictor`は3層構造で、中間層が512次元
- `batch_size=128`の場合、中間活性化: `128 × 512 = 65,536要素`
- BatchNorm統計、Dropout mask、勾配も同サイズで保存
- プレトレーニングの`PretrainHead`は128次元で出力が1次元のため、はるかに小さい

### 3. Mixed Precisionのオーバーヘッド

**問題点:**
- FP16計算を行うが、`GradScaler`がFP32のマスターコピーを保持
- モデルパラメータの約50%の追加メモリが必要

### 4. SWAモデルの重複保持

**問題点:**
- Epoch 30以降、`AveragedModel`によりモデル全体のコピーをメモリに保持
- 実質的にモデルが2倍になる

### 5. Pin Memoryによるページロックメモリ

**問題点:**
- GPU転送を高速化するため、`pin_memory=True`を使用
- スワップ不可能なメモリ領域を使用するため、システムの利用可能メモリが減少

## 実装した解決策

### レベル1: データローダー最適化（即効性・高）

**変更箇所:** `config_pretrain.yaml`

```yaml
# 変更前
num_workers: 8
prefetch_factor: 4
persistent_workers: true
batch_size: 128
gradient_accumulation_steps: 2

# 変更後
num_workers: 4              # プリフェッチメモリを半減
prefetch_factor: 2          # さらに半減
persistent_workers: false   # エポック毎にメモリ解放
batch_size: 64              # バッチサイズ削減
gradient_accumulation_steps: 4  # 実効バッチサイズ256を維持
```

**効果:**
- プリフェッチサンプル数: 4,096 → 512（87.5%削減）
- 実効バッチサイズは256を維持（学習の安定性を保持）
- エポック毎にワーカープロセスがメモリを解放

### レベル2: SWA開始エポックの遅延（即効性・中）

**変更箇所:** `config_pretrain.yaml`

```yaml
# 変更前
swa_start_epoch: 30  # 50エポック中の30エポック目から

# 変更後
swa_start_epoch: 45  # 50エポック中の最後5エポックのみ
```

**効果:**
- SWAモデルの重複保持期間を削減（20エポック → 5エポック）
- 学習の大部分でメモリを節約
- 最後の5エポックでSWAの恩恵を得る

### レベル3: Gradient Checkpointing実装（即効性・中）

**変更箇所:**
1. `src/models/gcn_model.py`
2. `scripts/finetune.py`
3. `scripts/pretrain.py`
4. `config_pretrain.yaml`

**実装内容:**

```python
# gcn_model.pyに追加
from torch.utils.checkpoint import checkpoint

class GCNMassSpecPredictor(nn.Module):
    def __init__(self, ..., use_gradient_checkpointing: bool = False):
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def extract_graph_features(self, data: Data):
        # グラフ畳み込み（Gradient Checkpointing対応）
        if self.use_gradient_checkpointing and self.training:
            # メモリ節約：中間活性化を保存せず、逆伝播時に再計算
            for conv_layer in self.conv_layers:
                x = checkpoint(conv_layer, x, edge_index, use_reentrant=False)
        else:
            for conv_layer in self.conv_layers:
                x = conv_layer(x, edge_index)
```

```yaml
# config_pretrain.yamlに追加
model:
  use_gradient_checkpointing: true
```

**効果:**
- 中間活性化のメモリを削減（5層分の活性化を保存しない）
- 逆伝播時に再計算するため、計算時間は若干増加（約10-20%）
- メモリと速度のトレードオフとして優れた選択

## メモリ削減の見積もり

| 項目 | 変更前 | 変更後 | 削減率 |
|------|--------|--------|--------|
| **データローダープリフェッチ** | 4,096サンプル | 512サンプル | **87.5%** |
| **SWA重複期間** | 20エポック | 5エポック | **75%** |
| **中間活性化（5層）** | 全て保存 | 再計算 | **約80%** |

**総合的なメモリ削減:** 推定50-60%のメモリ使用量削減

## 性能への影響

### 学習速度
- エポック時間: 若干増加（約10-20%）
  - `persistent_workers: false`によるワーカー起動オーバーヘッド（エポック間のみ）
  - Gradient Checkpointingによる再計算オーバーヘッド
- 全体の学習時間: 2-2.5時間 → 2.5-3時間程度

### 学習品質
- **実効バッチサイズは256で維持**: 学習の安定性は変わらず
- **SWAは依然として有効**: 最後の5エポックで汎化性能向上
- **収束性**: 影響なし

## 使用方法

### ファインチューニング実行

```bash
# デフォルト設定（メモリ最適化済み）
python scripts/finetune.py --config config_pretrain.yaml

# キャッシュ再構築が必要な場合
python scripts/finetune.py --config config_pretrain.yaml --rebuild-cache
```

### さらにメモリを節約したい場合

`config_pretrain.yaml`を編集:

```yaml
finetuning:
  batch_size: 32              # 64 → 32（さらに削減）
  gradient_accumulation_steps: 8  # 4 → 8（実効バッチサイズ256維持）
  num_workers: 2              # 4 → 2（さらに削減）
  use_swa: false              # SWAを完全に無効化
```

### メモリに余裕がある場合（より高速化）

```yaml
finetuning:
  batch_size: 128
  gradient_accumulation_steps: 2
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: true

model:
  use_gradient_checkpointing: false  # 中間活性化を保存
```

## トラブルシューティング

### 依然としてOOMが発生する場合

1. **さらにバッチサイズを削減**
   ```yaml
   batch_size: 32
   gradient_accumulation_steps: 8
   ```

2. **ワーカー数を削減**
   ```yaml
   num_workers: 2
   prefetch_factor: 1
   ```

3. **SWAを無効化**
   ```yaml
   use_swa: false
   ```

4. **Mixed Precisionを確認**
   - AMPは既に有効（FP16使用でメモリ削減）
   - 無効化すると精度は上がるがメモリ消費増加

### パフォーマンスが遅すぎる場合

1. **persistent_workersを有効化**（メモリに余裕があれば）
   ```yaml
   persistent_workers: true
   ```

2. **Gradient Checkpointingを無効化**（メモリに余裕があれば）
   ```yaml
   use_gradient_checkpointing: false
   ```

3. **num_workersを増やす**
   ```yaml
   num_workers: 6
   ```

## 参考情報

- PyTorch Gradient Checkpointing: https://pytorch.org/docs/stable/checkpoint.html
- PyTorch DataLoader: https://pytorch.org/docs/stable/data.html
- Stochastic Weight Averaging: https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/

## 変更履歴

- 2025-11-16: 初版作成 - ファインチューニングメモリ最適化を実装
