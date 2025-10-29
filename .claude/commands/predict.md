訓練済みモデルを使用して予測を実行します。

単一分子の予測を実行してください：
```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --output prediction.png
```

予測結果とピーク情報を分析して表示してください。
