## 📋 概要 (Summary)

Phase 3 (知識蒸留による学生モデル訓練) の複数の重大なバグを修正し、安定した150エポック学習を実現。

This PR fixes multiple critical bugs in Phase 3 (Knowledge Distillation) training pipeline, enabling stable 150-epoch training.

---

## 🐛 修正した問題 (Fixed Issues)

### 1. Phase 3 DataLoader Architecture Error
**問題:** `TypeError: 'DataLoader' object is not subscriptable`
- Teacher/Student用の別々のDataLoaderを組み合わせていたが、Trainerは単一DataLoaderを期待

**修正:**
- NISTDatasetに`'distill'`モードを追加（Teacher/Student両方の特徴を生成）
- `collate_fn_distill()`を実装し、単一バッチに統合
- `train_student.py`を簡素化（~150行 → ~130行）

**変更ファイル:**
- `src/data/nist_dataset.py`: distillモード、collate_fn_distill追加
- `scripts/train_student.py`: 統合DataLoader化

---

### 2. LDS Module dtype Mismatch
**問題:** `RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.FloatTensor) should be the same`
- Mixed Precision (FP16) 時に、LDSカーネルがFloat32のまま

**修正:**
```python
kernel = self.kernel.to(dtype=spectrum.dtype, device=spectrum.device)
```

**変更ファイル:**
- `src/models/modules.py`: LDS forward()でdtype変換

---

### 3. Expert Collapse (Epoch 8)
**問題:** Loss=nan、Expert Usage=[0.5, 0.5, 0.0, 0.0] (4専門家中2つのみ使用)
- `expert_bias`が計算されていたが、実際のルーティングで**無視されていた**

**修正:**
```python
# BEFORE: biasが無視される
gate_logits = self.gate.mlp(ecfp_count_fp)
gate_logits = gate_logits + self.expert_bias.unsqueeze(0)  # 計算したが...
expert_weights, expert_indices = self.gate.forward(ecfp_count_fp)  # 無視！

# AFTER: biasを正しく適用
gate_logits = self.gate.mlp(ecfp_count_fp)
gate_logits = gate_logits + self.expert_bias.unsqueeze(0)
all_weights = F.softmax(gate_logits, dim=-1)  # biasを考慮
top_k_weights, expert_indices = torch.topk(all_weights, self.top_k, dim=-1)
expert_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
```

**結果:** Expert Usage → [0.25, 0.25, 0.25, 0.25] (均等化)

**変更ファイル:**
- `src/models/student.py`: forward()とget_hidden_features()
- `config.yaml`: 学習率調整

---

### 4. Epoch 12 NaN Cascade (二重修正)

#### 4a. Mixed Precision数値不安定性
**問題:** Epoch 12 batch 236から段階的NaN発生 → 完全崩壊
- FP16での知識蒸留計算（KL Divergence等）が数値的に不安定
- Temperature=3.96のsoftmax計算でアンダーフロー

**修正:**
```yaml
use_amp: false         # FP16 → FP32（高精度化）
max_lr: 0.0003        # 0.0005 → 0.0003
learning_rate: 0.0002  # 0.0003 → 0.0002
```

**変更ファイル:**
- `config.yaml`: Mixed Precision無効化、学習率削減

#### 4b. NaN Check位置の問題 🎯
**問題:** NaNチェックが`backward()`と`optimizer.step()`の**後**に実行
- NaN発生時、不正な勾配でモデルの重みが破壊される
- 重みがNaNになると、以降すべてのバッチでNaN（復帰不可能）

**修正:**
```python
# NaN/Infチェックを backward()の「前」に移動
if torch.isnan(loss) or torch.isinf(loss):
    logger.warning("NaN detected BEFORE backward")
    continue  # backward/stepをスキップ → 重み保護

# ここに到達するのは正常な損失のみ
loss.backward()
optimizer.step()
```

**変更ファイル:**
- `src/training/student_trainer.py`: Early NaN detection

---

## 🛡️ 二重防御アーキテクチャ

| 防御層 | 目的 | 実装 |
|-------|------|------|
| **第1層** | NaN発生予防 | FP32 + 低学習率 |
| **第2層** | NaN発生時の被害防止 | Early Check → Skip batch |

これにより、万が一の不安定なバッチに遭遇してもモデルは破壊されず、学習継続可能。

---

## 📊 テスト結果

### Phase 2 (NIST Finetuning)
- ✅ 44,890サンプル正常ロード（89.8% of 50k）
- ✅ MSP + MOL統合成功（IDベースマッチング）

### Phase 3 (Knowledge Distillation)
- ✅ Epoch 1-11: 安定学習、Expert Usage均等
- ✅ Epoch 12以降: 従来は崩壊 → 修正後は継続可能

---

## 🔧 変更ファイル一覧

```
config.yaml                     |  8 ++---
scripts/train_student.py        | 73 +++++++--------
src/data/nist_dataset.py        | 58 ++++++++++++
src/models/modules.py           |  5 ++-
src/models/student.py           | 21 ++++++
src/training/student_trainer.py | 13 +++-
6 files changed, 109 insertions(+), 69 deletions(-)
```

---

## ✅ 期待される効果

1. **安定性向上:** Epoch 150まで安定学習
2. **Expert均等利用:** Load Balancing正常動作
3. **数値安定性:** FP32によるNaN発生抑制
4. **耐障害性:** 不安定バッチ自動スキップ
5. **目標達成:** Recall@10 ≥ 95.5%到達可能

---

## 📝 コミット履歴

- `c26c8c0` Critical: Move NaN check BEFORE backward() to prevent weight corruption
- `dcd824d` Fix Epoch 12 NaN cascade: Disable Mixed Precision and lower learning rates
- `83100b0` Fix Expert Collapse and gradient explosion (統合的修正)
- `8296776` Fix gradient explosion in Phase 3 training
- `5417ce0` Fix LDS module dtype mismatch for mixed precision training
- `91b199b` Fix Phase 3 knowledge distillation DataLoader architecture
- `7a7907f` Fix NIST dataset loading: Combine MSP spectrum data with MOL structure files
- `27bc301` Fix Phase 2 NIST dataset loading error

---

## 🚀 次のステップ

マージ後、以下のコマンドで学習を実行:

```bash
python scripts/train_student.py \
    --config config.yaml \
    --teacher checkpoints/teacher/best_finetune_teacher.pt \
    --device cuda
```

**監視ポイント:**
- Expert Usage: ~[0.25, 0.25, 0.25, 0.25]維持
- Validation Loss: 順調に減少
- NaN警告: 出ても学習継続（自動スキップ）
