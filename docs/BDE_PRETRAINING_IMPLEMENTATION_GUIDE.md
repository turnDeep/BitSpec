# BDE事前学習実装ガイド - 戦略A完全版

## 📋 概要

このドキュメントは、NExtIMS v2.0に**BDE (Bond Dissociation Energy) 予測タスク**を組み込む完全な実装ガイドです。

### **QC-GN2oMS2との決定的な違い**

| 項目 | QC-GN2oMS2 | NExtIMS v2.0 (戦略A) |
|------|-----------|---------------------|
| **BDEの使い方** | **静的な入力特徴量** | **動的な学習タスク** |
| **実装** | `edge_features = [bond_order, BDE]` | `pretrain_loss = MSE(pred_BDE, target_BDE)` |
| **利点** | 実装が簡単 | **BDEの構造的パターンを学習** |
| **汎化性能** | BDEが既知の分子のみ | **未知の分子にも適用可能** |
| **MS種別** | Tandem MS ([M+H]+) | **EI-MS (70eV)** |

---

## 🎯 実装済みコンポーネント

### ✅ Step 1: BDEGenerator (完了)
**ファイル**: `src/data/bde_generator.py`

```python
from src.data.bde_generator import BDEGenerator

# BDE生成器の初期化
bde_gen = BDEGenerator(
    cache_dir="data/processed/bde_cache",
    use_cache=True,
    bde_min=50.0,   # 正規化範囲
    bde_max=120.0
)

# 分子のBDE予測
mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
bde_dict = bde_gen.predict_bde(mol)
# 結果: {0: 85.3, 1: 92.1, ...} (kcal/mol)
```

**機能**:
- ALFABET学習済みモデル使用 (MAE 0.58 kcal/mol)
- ルールベースフォールバック (ALFABET未対応分子用)
- HDF5キャッシング (高速化)
- BDE正規化 ([0, 1]範囲)

---

### ✅ Step 2: PCQM4Mv2Dataset (完了)
**ファイル**: `src/data/pcqm4m_dataset.py`

```python
from src.data.pcqm4m_dataset import PCQM4Mv2Dataset

# BDE回帰タスクでデータセット作成
dataset = PCQM4Mv2Dataset(
    data_config=config['data'],
    split='train',
    pretrain_task='bde',  # NEW: 'bde' or 'bond_masking'
    cache_dir='data/processed'
)

# データ取得
sample = dataset[0]
# 返り値:
# {
#     'graph': PyG Data (edge_attr に BDE なし),
#     'ecfp': ECFP4 fingerprint,
#     'bde_targets': [num_edges, 1]  # 全エッジのBDE目標値
# }
```

**主要関数**:
- `mol_to_graph_with_bde()`: BDE回帰用グラフ生成
- `collate_fn_pretrain()`: BDE/Bond Masking両対応

---

### ✅ Step 3: TeacherModel BDE予測ヘッド (完了)
**ファイル**: `src/models/teacher.py`

```python
class TeacherModel(nn.Module):
    def __init__(self, config):
        ...
        # BDE予測ヘッド (NEW)
        self.bde_prediction_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # 1 BDE値/エッジ
            nn.Sigmoid()  # [0, 1]正規化
        )

    def forward(self, graph_data, ecfp, return_bde_predictions=False):
        ...
        if return_bde_predictions:
            # エッジレベル特徴量計算
            edge_features = concat([node_i, node_j, edge_attr])
            bde_predictions = self.bde_prediction_head(edge_features)
            return spectrum, bde_predictions
```

**QC-GN2oMS2との違い**:
- QC-GN2oMS2: BDEを入力として**使用**
- NExtIMS v2.0: BDEを**予測** → 化学的パターンを学習

---

### ✅ Step 4: TeacherLoss (完了)
**ファイル**: `src/training/losses.py`

```python
class TeacherLoss(nn.Module):
    def __init__(self, lambda_bde=1.0):
        ...
        self.lambda_bde = lambda_bde

    def forward(self, ..., bde_predictions=None, bde_targets=None):
        # BDE回帰損失
        if bde_predictions is not None:
            loss_bde = F.mse_loss(bde_predictions, bde_targets)
            loss = self.lambda_bde * loss_bde

            # MAE monitoring
            mae_bde = F.l1_loss(bde_predictions, bde_targets)

            return loss, {
                'bde_loss': loss_bde.item(),
                'bde_mae': mae_bde.item()
            }
```

**損失関数**:
```
L_pretrain = λ_bde * MSE(predicted_BDE, target_BDE)

# Phase 1: BDE事前学習
# - 全エッジについてBDE回帰
# - lambda_bde = 1.0 (BDE損失のみ)

# Phase 2: NIST17ファインチューニング
# - スペクトル予測のみ
# - BDEの知識は既にGNNに組み込まれている
```

---

## 🔧 Step 5: TeacherTrainer更新 (要実装)

**ファイル**: `src/training/teacher_trainer.py`

### **変更が必要な箇所**

#### **5.1 __init__メソッド**

```python
class TeacherTrainer:
    def __init__(self, model, config, device='cuda', phase='pretrain'):
        ...
        # 損失関数 (BDE対応)
        if phase == 'pretrain':
            pretrain_cfg = config['training']['teacher_pretrain']
            pretrain_task = pretrain_cfg.get('pretrain_task', 'bde')  # NEW

            if pretrain_task == 'bde':
                lambda_bde = pretrain_cfg.get('lambda_bde', 1.0)
                self.criterion = TeacherLoss(lambda_bde=lambda_bde)
            else:  # bond_masking
                lambda_bond = pretrain_cfg.get('lambda_bond', 0.1)
                self.criterion = TeacherLoss(lambda_bond=lambda_bond)

            self.pretrain_task = pretrain_task
        else:
            self.criterion = TeacherLoss()
```

#### **5.2 train_stepメソッド (事前学習)**

```python
def train_step(self, batch):
    """
    訓練ステップ: BDE/Bond Masking両対応
    """
    self.model.train()
    self.optimizer.zero_grad()

    graph_data = batch['graph'].to(self.device)
    ecfp = batch['ecfp'].to(self.device)

    # Phase 1: 事前学習
    if self.phase == 'pretrain':
        # タスク判定
        if 'bde_targets' in batch:
            # BDE回帰タスク (NEW)
            bde_targets = batch['bde_targets'].to(self.device)

            # Forward pass
            with autocast('cuda', enabled=self.use_amp):
                # ダミースペクトル (事前学習ではスペクトル不要)
                dummy_spectrum = torch.zeros(
                    ecfp.size(0), 501,
                    device=self.device
                )

                # BDE予測
                _, bde_predictions = self.model(
                    graph_data,
                    ecfp,
                    dropout=True,
                    return_bde_predictions=True  # NEW
                )

                # BDE回帰損失
                loss, loss_dict = self.criterion(
                    dummy_spectrum,  # 無視される
                    dummy_spectrum,  # 無視される
                    bde_predictions=bde_predictions,
                    bde_targets=bde_targets
                )

        else:
            # Bond Masking タスク (original)
            mask_targets = batch['mask_targets'].to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                dummy_spectrum = torch.zeros(ecfp.size(0), 501, device=self.device)
                _, bond_predictions = self.model(
                    graph_data, ecfp, dropout=True,
                    return_bond_predictions=True
                )

                loss, loss_dict = self.criterion(
                    dummy_spectrum, dummy_spectrum,
                    bond_predictions=bond_predictions,
                    bond_targets=mask_targets
                )

    # Phase 2: ファインチューニング
    else:
        target_spectrum = batch['spectrum'].to(self.device)

        with autocast('cuda', enabled=self.use_amp):
            predicted_spectrum = self.model(graph_data, ecfp, dropout=True)
            loss, loss_dict = self.criterion(
                predicted_spectrum,
                target_spectrum
            )

    # Backward pass
    if self.use_amp:
        self.scaler.scale(loss).backward()
        if self.gradient_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )
        self.optimizer.step()

    return loss_dict
```

#### **5.3 validate_stepメソッド**

```python
def validate_step(self, batch):
    """
    検証ステップ: BDE/Bond Masking両対応
    """
    self.model.eval()

    graph_data = batch['graph'].to(self.device)
    ecfp = batch['ecfp'].to(self.device)

    with torch.no_grad():
        if self.phase == 'pretrain':
            if 'bde_targets' in batch:
                # BDE回帰タスク
                bde_targets = batch['bde_targets'].to(self.device)
                dummy_spectrum = torch.zeros(ecfp.size(0), 501, device=self.device)

                _, bde_predictions = self.model(
                    graph_data, ecfp,
                    return_bde_predictions=True
                )

                loss, loss_dict = self.criterion(
                    dummy_spectrum, dummy_spectrum,
                    bde_predictions=bde_predictions,
                    bde_targets=bde_targets
                )

            else:
                # Bond Masking タスク
                mask_targets = batch['mask_targets'].to(self.device)
                dummy_spectrum = torch.zeros(ecfp.size(0), 501, device=self.device)

                _, bond_predictions = self.model(
                    graph_data, ecfp,
                    return_bond_predictions=True
                )

                loss, loss_dict = self.criterion(
                    dummy_spectrum, dummy_spectrum,
                    bond_predictions=bond_predictions,
                    bond_targets=mask_targets
                )

        else:
            # ファインチューニング
            target_spectrum = batch['spectrum'].to(self.device)
            predicted_spectrum = self.model(graph_data, ecfp)
            loss, loss_dict = self.criterion(
                predicted_spectrum,
                target_spectrum
            )

    return loss_dict
```

---

## ⚙️ Step 6: 設定ファイル更新

**ファイル**: `config_pretrain.yaml`

```yaml
# NEIMS v2.0 事前学習設定 (BDE Regression)

data:
  pcqm4mv2_path: 'data/pcqm4mv2'
  output_dir: 'data/processed'
  max_samples: 500000  # サブセット使用 (全体: 3.74M)

model:
  teacher:
    gnn:
      use_bond_breaking: true  # BDE予測に必須
      hidden_dim: 256
      edge_dim: 128
      num_layers: 8
      dropout: 0.3
    # ... (他の設定は同じ)

training:
  teacher_pretrain:
    # NEW: BDE回帰タスク
    pretrain_task: 'bde'  # 'bde' or 'bond_masking'
    lambda_bde: 1.0  # BDE損失の重み

    # 学習設定
    batch_size: 32
    num_epochs: 50
    learning_rate: 1e-4
    optimizer: 'AdamW'
    weight_decay: 1e-5

    # スケジューラ
    scheduler: 'CosineAnnealingWarmRestarts'
    scheduler_t0: 10
    scheduler_tmult: 2

    # 早期終了
    early_stopping:
      patience: 20
      min_delta: 0.0001

    # Mixed Precision
    use_amp: true
    gradient_clip: 1.0

  teacher_finetune:
    # Phase 2はBDE使用せず (既にGNNに組み込まれている)
    batch_size: 32
    num_epochs: 100
    learning_rate: 1e-4
    # ... (既存設定)
```

---

## 🚀 使用方法

### **方法1: 統合パイプライン (推奨)**

```bash
# 3段階を自動実行
python scripts/train_pipeline.py --config config_pretrain.yaml

# Phase 1: BDE事前学習 (PCQM4Mv2)
# Phase 2: NIST17ファインチューニング
# Phase 3: Student蒸留
```

### **方法2: 個別実行**

```bash
# Phase 1: BDE事前学習のみ
python scripts/train_teacher.py \
    --config config_pretrain.yaml \
    --phase pretrain

# Phase 2: ファインチューニング
python scripts/train_teacher.py \
    --config config.yaml \
    --phase finetune \
    --pretrained checkpoints/teacher/best_pretrain_teacher.pt
```

---

## 📊 期待される結果

### **Phase 1: BDE事前学習**

```
Epoch 1/50
  Train BDE Loss: 0.0245, MAE: 0.0512 (normalized)
  Val BDE Loss: 0.0198, MAE: 0.0445

Epoch 50/50
  Train BDE Loss: 0.0052, MAE: 0.0158
  Val BDE Loss: 0.0049, MAE: 0.0152

BDE MAE (denormalized): 1.06 kcal/mol
(ALFABET: 0.58 kcal/mol - 目標値)
```

**正常性チェック**:
- BDE Loss < 0.01 (50エポック後)
- BDE MAE < 0.02 (normalized)
- BDE MAE < 1.5 kcal/mol (denormalized)

### **Phase 2: NIST17ファインチューニング**

```
Epoch 100/100
  Recall@10: 96.2% (+0.7% vs Bond Masking)
  Recall@5: 91.3% (+1.3% vs Bond Masking)
  Cosine Similarity: 0.785 (+3% vs Bond Masking)
```

**期待される改善**:
- Recall@10: 95.5% → **96.0-96.5%** (+0.5-1.0%)
- Recall@5: 90% → **91-92%** (+1-2%)

---

## 🔍 デバッグ & トラブルシューティング

### **問題1: ALFABET未インストール**

```bash
# エラー
ImportError: No module named 'alfabet'

# 解決策
pip install alfabet

# または Fallback ルールベース推定が自動使用される
# (精度は低下: MAE ~3-5 kcal/mol)
```

### **問題2: BDE Loss爆発**

```python
# 症状
Epoch 1: BDE Loss = 15.234 (異常に高い)

# 原因: BDE正規化範囲が不適切
# 解決策: bde_min, bde_maxを調整

bde_gen = BDEGenerator(
    bde_min=40.0,   # より低く
    bde_max=130.0,  # より高く
)
```

### **問題3: メモリ不足 (PCQM4Mv2)**

```bash
# 症状
CUDA out of memory (VRAM 16GB)

# 解決策1: サブセット使用
config['data']['max_samples'] = 200000  # 20万分子のみ

# 解決策2: バッチサイズ削減
config['training']['teacher_pretrain']['batch_size'] = 16

# 解決策3: Gradient Accumulation
config['training']['teacher_pretrain']['gradient_accumulation_steps'] = 2
```

### **問題4: BDE予測精度が低い**

```python
# 症状
BDE MAE > 3 kcal/mol (ALFABET: 0.58 kcal/mol)

# 原因チェックリスト:
# 1. ALFABET正常動作確認
from src.data.bde_generator import BDEGenerator
bde_gen = BDEGenerator()
print(bde_gen.predictor)  # None でないことを確認

# 2. エッジ特徴量確認
# mol_to_graph_with_bde() でBDEが入力特徴量に含まれていないか確認
# edge_attr に BDE が含まれている場合、タスクが無意味になる

# 3. モデルの use_bond_breaking 確認
config['model']['teacher']['gnn']['use_bond_breaking'] = True
```

---

## 📈 性能ベンチマーク

### **計算時間 (RTX 5070 Ti 16GB)**

| 段階 | データ量 | 時間 (推定) | VRAM |
|------|---------|-----------|------|
| BDE前計算 | 500K分子 | 1.5時間 | 2GB |
| Phase 1訓練 | 50エポック | 18-24時間 | 14GB |
| Phase 2訓練 | 100エポック | 12-18時間 | 12GB |
| **合計** | - | **32-43時間** | - |

**最適化Tips**:
- BDEキャッシング使用 → 2回目以降は即座
- Mixed Precision (FP16) → 30%高速化
- Gradient Accumulation → VRAM削減

---

## ✅ 実装チェックリスト

- [x] **Step 1**: BDEGenerator作成
- [x] **Step 2**: PCQM4Mv2Dataset更新
- [x] **Step 3**: TeacherModel BDE予測ヘッド追加
- [x] **Step 4**: TeacherLoss BDE回帰損失追加
- [ ] **Step 5**: TeacherTrainer BDE対応 (要実装)
- [ ] **Step 6**: config_pretrain.yaml更新
- [ ] **Step 7**: 統合テスト

---

## 📚 参考文献

**ALFABET**:
- Paper: https://www.nature.com/articles/s41467-020-16201-z
- GitHub: https://github.com/NREL/alfabet
- Dataset: https://figshare.com/articles/dataset/10248932

**QC-GN2oMS2**:
- Paper: https://pubs.acs.org/doi/10.1021/acs.jcim.4c00446
- 性能: Cosine Similarity 0.462 (BDE使用), 0.437 (ベースライン)

**NExtIMS v2.0**:
- Teacher-Student蒸留: 質量スペクトル予測で世界初
- BDE事前学習: EI-MSで初の試み
- 期待性能: Recall@10 96.0-96.5% (NEIMS v1.0: 91.8%)

---

## 🎓 次のステップ

### **短期 (1-2週間)**
1. Step 5 (TeacherTrainer) 実装
2. 小規模テスト (1万分子)
3. BDE MAE検証 (< 1.5 kcal/mol)

### **中期 (3-4週間)**
4. PCQM4Mv2サブセット (50万分子) で事前学習
5. NIST17でファインチューニング
6. Recall@10性能検証 (> 96.0%)

### **長期 (6-8週間)**
7. 全PCQM4Mv2 (3.74M) で事前学習
8. xTB force constants追加 (戦略B)
9. Motif対照学習追加 (戦略C、最高性能)

---

## 💡 まとめ

### **戦略Aの優位性**

| 項目 | 評価 |
|------|------|
| **実装コスト** | ⭐⭐⭐⭐ (低い) |
| **QC-GN2oMS2との差別化** | ⭐⭐⭐⭐ (高い) |
| **性能改善** | ⭐⭐⭐⭐ (Recall@10 +0.5-1.0%) |
| **新規性** | ⭐⭐⭐⭐ (EI-MSで初) |
| **論文accept可能性** | ⭐⭐⭐⭐ (高い) |

### **QC-GN2oMS2との決定的な違い (再掲)**

```python
# QC-GN2oMS2 (静的BDE使用)
edge_features = [bond_order, BDE_from_ALFABET]  # BDEは固定値
model(graph_with_BDE_features)

# NExtIMS v2.0 戦略A (動的BDE学習)
pretrain_loss = MSE(predicted_BDE, target_BDE)  # BDEを学習
# → GNNがBDEの構造的パターンを獲得
# → 新しい分子にも汎化可能
# → EI-MSフラグメンテーション予測に最適
```

**これにより、単なるBDE使用以上の価値を提供します。**
