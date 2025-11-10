# エッジ統合機能の互換性について

## 変更内容

新しいモデルでは、結合（エッジ）の特徴量をノード特徴に統合する機能を追加しました。

### 追加された層

```python
self.edge_to_node_fusion = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout)
)
```

## 互換性の状況

### ✓ 出力次元は完全に互換性あり

- グラフ特徴量: `[batch_size, 256]` (変わらず)
- 予測スペクトル: `[batch_size, 1000]` (変わらず)
- PretrainHead、spectrum_predictorとの接続: **問題なし**

### ⚠️ チェックポイントの互換性

#### ケース1: 新規学習（事前学習から）
- **状態**: 問題なし
- **動作**: 全ての層（edge_to_node_fusionを含む）がゼロから学習される
- **推奨**: これが最も効果的

#### ケース2: 既存の事前学習済みチェックポイントをロード
- **状態**: 部分的な転移学習
- **動作**:
  - `edge_to_node_fusion`以外: 事前学習済み重みをロード ✓
  - `edge_to_node_fusion`: ランダム初期化 ⚠️
- **影響**:
  - edge_to_node_fusion層のみがランダム初期化
  - ファインチューニング時にこの層を学習する必要がある
  - 事前学習の効果は大部分保たれる（conv_layersなどは転移）
  - ただし、edge_to_node_fusionが未学習なため、最初は不安定な可能性

#### ケース3: edge_attr なしのデータで予測
- **状態**: 完全に後方互換性あり
- **動作**: edge_to_node_fusionがスキップされ、従来通りの処理
- **推奨**: edge_attrがないデータセットでも問題なく動作

## 推奨事項

### 最良の結果を得るには

1. **新しい事前学習を実行** (推奨)
   ```bash
   python scripts/pretrain.py --config config_pretrain.yaml
   ```
   - 全ての層（edge_to_node_fusionを含む）が最適化される
   - エッジ情報が完全に活用される

2. **既存のチェックポイントからファインチューニング** (次善)
   - edge_to_node_fusion以外は転移学習の恩恵を受ける
   - edge_to_node_fusionは学習率を高めに設定することを推奨
   - ファインチューニングのエポック数を増やす（50 → 70など）

### パラメータ数の変化

```python
# edge_to_node_fusion のパラメータ数
Linear(512, 256): 131,072 + 256 (bias) = 131,328
BatchNorm1d(256): 512
合計: 約 131,840 パラメータ
```

全体のパラメータ数に占める割合は小さい（< 1%）ため、ランダム初期化でも大きな問題にはならない可能性があります。

## 実装の詳細

### edge_attrの有無による動作の違い

```python
# edge_attrがある場合
if edge_attr is not None and edge_attr.size(0) > 0:
    edge_emb = self.edge_embedding(edge_attr)
    # ... エッジ情報をノードに統合
    x = self.edge_to_node_fusion(x)

# edge_attrがない場合
# → edge_to_node_fusionはスキップ、xはnode_embeddingの出力そのまま
```

これにより：
- edge_attrを持つデータセット: エッジ情報を活用
- edge_attrを持たないデータセット: 従来通り動作（後方互換性）

## まとめ

| 項目 | 状態 |
|------|------|
| 出力次元 | ✓ 変わらず |
| 予測時の動作 | ✓ edge_attrがあれば活用、なければスキップ |
| 新規学習 | ✓ 問題なし |
| 既存チェックポイント | ⚠️ edge_to_node_fusionのみ未学習 |
| 推奨アクション | 新しい事前学習を実行 |
