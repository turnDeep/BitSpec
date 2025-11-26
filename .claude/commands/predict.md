訓練済みモデルを使用して予測を実行します。

## 単一分子の予測（Student：高速）

```bash
python scripts/predict.py \
    --config config.yaml \
    --checkpoint checkpoints/student/best_student.pt \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

## 不確実性推定付き予測（Teacher：高精度）

```bash
python scripts/predict.py \
    --config config.yaml \
    --checkpoint checkpoints/teacher/best_finetune_teacher.pt \
    --model teacher \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

## バッチ予測

```bash
# smiles_list.txtに各行1つのSMILES文字列を記載
python scripts/predict.py \
    --config config.yaml \
    --checkpoint checkpoints/student/best_student.pt \
    --batch smiles_list.txt
```

## 予測結果の分析

予測結果とピーク情報を分析して表示してください。以下の情報を含めてください：

- Top 10ピークのm/z値と強度
- 予測の信頼度（Teacherの場合は不確実性も）
- 分子の基本情報（分子量、化学式など）
