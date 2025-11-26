モデルをトレーニングします。

NExtIMS v2.0は3段階のトレーニングプロセスを使用します。統合パイプラインで全ステップを自動実行できます：

```bash
# 統合パイプライン（推奨）: 全3段階を自動実行
python scripts/train_pipeline.py --config config.yaml
```

## 個別ステップの実行

各段階を個別に実行することも可能です：

### Phase 0: BDE事前計算（BDE Regression使用時のみ）

```bash
# サブセット（50万分子、推奨）
python scripts/precompute_bde.py --max-samples 500000

# 全データ（3.74M分子）
python scripts/precompute_bde.py --max-samples 0
```

### Phase 1: Teacher事前学習（PCQM4Mv2）

```bash
# BDE Regression（推奨）
python scripts/train_teacher.py --config config_pretrain.yaml --phase pretrain

# Bond Masking（従来手法）
# config_pretrain.yaml の pretrain_task: 'bond_masking' に変更してから実行
python scripts/train_teacher.py --config config_pretrain.yaml --phase pretrain
```

### Phase 2: Teacherファインチューニング（NIST EI-MS）

```bash
python scripts/train_teacher.py \
    --config config.yaml \
    --phase finetune \
    --pretrained checkpoints/teacher/best_pretrain_teacher.pt
```

### Phase 3: Student知識蒸留

```bash
python scripts/train_student.py \
    --config config.yaml \
    --teacher checkpoints/teacher/best_finetune_teacher.pt
```

## トレーニング状態の確認

トレーニングが完了したら、以下を確認してください：

- チェックポイントが `checkpoints/` に保存されているか
- トレーニングログに異常がないか
- 評価メトリクス（Recall@10など）が目標値に達しているか

必要に応じてハイパーパラメータの調整を提案します。
