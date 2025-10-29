NIST MSP形式のデータを前処理します。

以下のコマンドを使用してデータを前処理してください：
```bash
python scripts/preprocess_data.py \
    --input data/raw/nist_data.msp \
    --output data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

前処理が完了したら、生成されたデータの統計情報を表示してください。
