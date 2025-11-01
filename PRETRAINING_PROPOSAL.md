# PCQM4Mv2事前学習のBitSpecへの適用提案

## 背景

Massformerの成功例に基づき、EI-MSモデルであるBitSpecにもPCQM4Mv2事前学習を適用できる可能性が高い。

## 根拠

### 1. Massformerの実証
- **タンデムMS予測**でありながら、PCQM4Mv2（量子化学）で事前学習
- GitHubコード: `gf_pretrain_name: pcqm4mv2_graphormer_base`
- 分子表現の事前学習が異なるスペクトル予測タスクでも有効

### 2. 共通の学習課題
- EI-MSもMS/MSも「分子構造 → フラグメンテーション」
- 量子化学的性質（HOMO-LUMO gap）は結合の強度・フラグメンテーションしやすさに関連
- グラフニューラルネットワークによる分子表現学習が共通基盤

### 3. 研究文献の裏付け
- 「Pretraining graph transformers with atom-in-a-molecule quantum properties for improved ADMET modeling」
- 「atom-level pretraining with QM data improves performance and generalization」
- 転移学習で最大8倍の性能向上、少ないデータで学習可能

## 実装アプローチ

### フェーズ1: データ準備

```python
# PCQM4Mv2データセットの取得
from torch_geometric.datasets import PCQM4Mv2

dataset = PCQM4Mv2(root='data/pcqm4mv2')
# 3,746,619分子、HOMO-LUMO gap付き
```

### フェーズ2: 事前学習タスクの設計

#### オプションA: 量子化学的性質予測（推奨）
```python
class PretrainHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # HOMO-LUMO gap
        )

    def forward(self, graph_features):
        return self.predictor(graph_features)
```

#### オプションB: マルチタスク事前学習
```python
class MultiTaskPretrainHead(nn.Module):
    """複数の量子化学的性質を同時予測"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.homo_lumo_head = nn.Linear(hidden_dim, 1)
        self.dipole_head = nn.Linear(hidden_dim, 3)  # 双極子モーメント
        self.energy_head = nn.Linear(hidden_dim, 1)  # 全エネルギー
```

### フェーズ3: GCNバックボーンの事前学習

```python
# scripts/pretrain.py
from src.models.gcn_model import GCNMassSpecPredictor
from torch_geometric.datasets import PCQM4Mv2

# BitSpecのGCNバックボーンを使用
backbone = GCNMassSpecPredictor(
    node_features=48,  # BitSpecの現在の設定
    edge_features=6,
    hidden_dim=256,
    num_layers=5,
    spectrum_dim=1000
)

# 事前学習ヘッドを追加
pretrain_head = PretrainHead(hidden_dim=256)

# PCQM4Mv2で学習
for epoch in range(pretrain_epochs):
    for batch in pcqm4mv2_loader:
        # GCNでグラフ表現を抽出
        graph_features = backbone.extract_graph_features(batch)

        # HOMO-LUMO gap予測
        pred_gap = pretrain_head(graph_features)
        loss = F.mse_loss(pred_gap, batch.y)

        loss.backward()
        optimizer.step()

# バックボーンの重みを保存
torch.save(backbone.state_dict(), 'checkpoints/pretrained_backbone.pt')
```

### フェーズ4: EI-MSタスクへのファインチューニング

```python
# scripts/finetune.py

# 事前学習済みバックボーンをロード
backbone.load_state_dict(torch.load('checkpoints/pretrained_backbone.pt'))

# スペクトル予測ヘッドは新規初期化（または部分的に凍結）
# オプション1: バックボーンを凍結して予測ヘッドのみ学習
for param in backbone.conv_layers.parameters():
    param.requires_grad = False

# オプション2: 全体を低学習率でファインチューニング
optimizer = torch.optim.Adam([
    {'params': backbone.conv_layers.parameters(), 'lr': 1e-5},
    {'params': backbone.spectrum_predictor.parameters(), 'lr': 1e-3}
])

# NIST EI-MSデータでファインチューニング
for epoch in range(finetune_epochs):
    for batch in nist_loader:
        spectrum_pred = backbone(batch)
        loss = modified_cosine_loss(spectrum_pred, batch.spectrum)

        loss.backward()
        optimizer.step()
```

### フェーズ5: アーキテクチャの拡張（オプション）

```python
# src/models/gcn_model.py に追加

class GCNMassSpecPredictor(nn.Module):
    def extract_graph_features(self, data):
        """事前学習用：グラフレベル表現のみを抽出"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 入力埋め込み
        x = self.node_embedding(x)

        # グラフ畳み込み
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)

        # グラフプーリング
        graph_features = self.pool(x, batch)

        return graph_features  # スペクトル予測前の表現を返す

    def forward(self, data):
        """EI-MS予測用：完全なパイプライン"""
        graph_features = self.extract_graph_features(data)
        spectrum = self.spectrum_predictor(graph_features)
        return spectrum
```

## 実装ファイル構成

```
BitSpec/
├── data/
│   └── pcqm4mv2/           # 新規：事前学習データ
├── scripts/
│   ├── pretrain.py         # 新規：事前学習スクリプト
│   ├── finetune.py         # 新規：ファインチューニング
│   └── train.py            # 既存：通常訓練（比較用）
├── src/
│   ├── models/
│   │   └── gcn_model.py    # 拡張：extract_graph_features追加
│   └── data/
│       └── pcqm4mv2_loader.py  # 新規：PCQM4Mv2データローダー
└── config_pretrain.yaml    # 新規：事前学習設定
```

## 設定ファイル例

```yaml
# config_pretrain.yaml
pretraining:
  dataset: "PCQM4Mv2"
  data_path: "data/pcqm4mv2"
  task: "homo_lumo_gap"  # または multi_task

  batch_size: 256
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001

  # 学習する層の制御
  freeze_layers: []  # 事前学習では全層を学習

finetuning:
  pretrained_checkpoint: "checkpoints/pretrained_backbone.pt"

  # バックボーン凍結設定
  freeze_backbone: false
  freeze_layers: 3  # 下位3層を凍結（オプション）

  # 異なる学習率
  backbone_lr: 0.0001  # 事前学習済み層は低学習率
  head_lr: 0.001       # 新規層は高学習率

  batch_size: 32
  num_epochs: 200
```

## 期待される効果

### 1. データ効率の向上
- **少ないEI-MSデータで高性能**を達成
- NIST データが限られていても、300万分子の知識を活用

### 2. 汎化性能の向上
- 事前学習で学んだ分子の一般的な構造パターン
- 未知の分子に対するロバスト性

### 3. 学習の安定化
- より良い初期重み
- 収束が早く、局所最適解を回避

### 4. 解釈可能性
- 量子化学的性質を理解した上でのスペクトル予測
- アテンションメカニズムの可視化で、フラグメント重要度を解析

## ベンチマーク計画

### 比較実験
1. **ランダム初期化** (現在のBitSpec)
2. **PCQM4Mv2事前学習 + ファインチューニング**
3. **異なる事前学習タスク**（HOMO-LUMO vs マルチタスク）
4. **凍結vs全体ファインチューニング**

### 評価指標
- Cosine Similarity（主要指標）
- Top-K Accuracy
- 学習曲線の収束速度
- データ量を変えた性能（25%, 50%, 75%, 100%）

## タイムライン

| フェーズ | タスク | 推定時間 |
|---------|--------|---------|
| 1 | PCQM4Mv2データ準備 | 1-2日 |
| 2 | 事前学習スクリプト実装 | 2-3日 |
| 3 | 事前学習実行 | 3-5日（GPU依存） |
| 4 | ファインチューニング実装 | 1-2日 |
| 5 | ファインチューニング実行 | 2-3日 |
| 6 | ベンチマーク評価 | 2-3日 |
| **合計** | | **11-18日** |

## リスクと対策

### リスク1: メモリ不足
- **対策**: グラディエントチェックポイント、バッチサイズ削減、データサンプリング

### リスク2: 転移学習が効かない
- **対策**: 異なる凍結戦略、学習率調整、段階的解凍

### リスク3: 計算時間
- **対策**: Mixed Precision (FP16)、torch.compile、データ並列化

## 参考文献

1. **Massformer**: Young et al., "Tandem Mass Spectrum Prediction with Graph Transformers", Nature Machine Intelligence, 2024
   - PCQM4Mv2事前学習の実証例

2. **PCQM4Mv2**: "PCQM4Mv2 | Open Graph Benchmark"
   - 3.74M分子、量子化学的性質

3. **Transfer Learning**: "Transfer learning with graph neural networks for improved molecular property prediction", Nature Communications, 2024
   - 最大8倍の性能向上

4. **Atom-level Pretraining**: "Pretraining graph transformers with atom-in-a-molecule quantum properties for improved ADMET modeling", J. Cheminformatics, 2025
   - 量子化学的性質での事前学習の有効性

## 結論

**PCQM4Mv2事前学習はBitSpecのEI-MS予測性能を向上させる可能性が高い。**

Massformerの成功例、研究文献の裏付け、そして分子グラフ学習の共通基盤から判断して、実装する価値がある。特に、NIST EI-MSデータが限られている場合、300万分子の事前学習知識は大きなアドバンテージとなる。
