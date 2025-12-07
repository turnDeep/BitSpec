# NExtIMS v4.2: クイックスタートガイド

このガイドでは、NExtIMS v4.2を使ってEI-MSスペクトル予測を行うための最短手順を説明します。

## 📋 前提条件

- **GPU**: NVIDIA RTX 5070 Ti (16GB) または同等
- **RAM**: 32GB以上
- **ストレージ**: 500GB以上の空き容量
- **OS**: Ubuntu 22.04+ または Windows 11 with WSL2

## 🚀 5分で始める

### ステップ1: 環境セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/turnDeep/NExtIMS.git
cd NExtIMS

# 依存関係のインストール
pip install -r requirements.txt

# 確認
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### ステップ2: データ準備

```bash
# NIST17データセットを配置（ライセンス取得が必要）
# https://www.nist.gov/srd/nist-standard-reference-database-1a

# データ構造:
# - NIST17.MSP: マススペクトルデータ（ピーク情報のみ）
# - mol_files/: 化学構造データ（MOLファイル）
# - ID番号でリンク: MSP内のIDとMOLファイル名（ID12345.MOL）が対応

# ファイルを配置
mkdir -p data
mkdir -p data/mol_files
cp /path/to/mainlib data/NIST17.MSP
cp -r /path/to/mol_files/* data/mol_files/

# 確認
ls -lh data/NIST17.MSP
ls data/mol_files/ | head -10
echo "Total MOL files: $(ls data/mol_files/*.MOL | wc -l)"
```

### ステップ3: BonDNet BDEモデル準備（Phase 0）

**初心者・すぐ始めたい方は「Option A」を推奨します**

#### Option A: 公式Pre-trained modelを使用（推奨）

```bash
# 何もする必要なし！
# BonDNet公式の学習済みモデル (bdncm/20200808) が
# 以降のスクリプトで自動ダウンロード・使用されます

# NIST17カバレッジ: ~95%
# 学習時間: 0時間（即座に開始可能）
# 対応元素: C, H, O, N, F (5元素)
```

#### Option B: BDE-db2で再学習（上級者向け）

**より高いカバレッジが必要な場合のみ**

```bash
# BDE-db2データセットのダウンロード（約10GB）
python scripts/download_bde_db2.py \
    --output data/external/bde-db2

# BonDNetの学習（48-72時間）
python scripts/train_bondnet_bde_db2.py \
    --data-path data/external/bde-db2 \
    --output models/bondnet_bde_db2_best.pth \
    --epochs 100 \
    --batch-size 256

# 検証
python scripts/train_bondnet_bde_db2.py \
    --data-path data/external/bde-db2 \
    --model models/bondnet_bde_db2_best.pth \
    --evaluate-only

# NIST17カバレッジ: ~99%+
# 学習時間: 48-72時間
# 対応元素: C, H, O, N, S, Cl, F, P, Br, I (10元素)
```

### ステップ4: GNN学習（Phase 2）

#### Option A使用時（公式Pre-trained BonDNet）

```bash
# データ準備とBDEキャッシュ生成（約2時間）
python scripts/train_gnn_minimal.py \
    --nist-msp data/NIST17.MSP \
    --output models/qcgn2oei_minimal_best.pth \
    --epochs 200 \
    --batch-size 32 \
    --create-cache
# bdncm/20200808 が自動使用される

# 学習開始（約40時間）
# ※ バックグラウンド実行推奨
nohup python scripts/train_gnn_minimal.py \
    --nist-msp data/NIST17.MSP \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --output models/qcgn2oei_minimal_best.pth \
    --epochs 200 \
    --batch-size 32 \
    > training.log 2>&1 &

# 進捗確認
tail -f training.log
```

#### Option B使用時（再学習済みBonDNet）

```bash
# データ準備とBDEキャッシュ生成（約2時間）
python scripts/train_gnn_minimal.py \
    --nist-msp data/NIST17.MSP \
    --bondnet-model models/bondnet_bde_db2_best.pth \
    --output models/qcgn2oei_minimal_best.pth \
    --epochs 200 \
    --batch-size 32 \
    --create-cache

# 学習開始（約40時間）
nohup python scripts/train_gnn_minimal.py \
    --nist-msp data/NIST17.MSP \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --bondnet-model models/bondnet_bde_db2_best.pth \
    --output models/qcgn2oei_minimal_best.pth \
    --epochs 200 \
    --batch-size 32 \
    > training.log 2>&1 &

# 進捗確認
tail -f training.log
```

### ステップ5: 評価（Phase 3）

```bash
# モデル評価
python scripts/evaluate_minimal.py \
    --model models/qcgn2oei_minimal_best.pth \
    --nist-msp data/NIST17.MSP \
    --visualize --benchmark \
    --output-dir results/evaluation

# 結果確認
cat results/evaluation/evaluation_report.json
```

### ステップ6: 推論（Phase 5）

```bash
# 単一分子予測
python scripts/predict_single.py "CCO" \
    --model models/qcgn2oei_minimal_best.pth \
    --visualize --output ethanol.png

# 結果表示
display ethanol.png  # または open ethanol.png (macOS)
```

## 🎯 よくある使用例

### 例1: カフェインのスペクトル予測

```bash
python scripts/predict_single.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
    --model models/qcgn2oei_minimal_best.pth \
    --visualize \
    --output caffeine_spectrum.png \
    --top-k 15
```

### 例2: 複数分子のバッチ予測

```bash
# CSVファイル作成
cat > molecules.csv << EOF
smiles,id,name
CCO,mol_001,ethanol
CC(C)O,mol_002,isopropanol
CC(=O)C,mol_003,acetone
c1ccccc1,mol_004,benzene
CC(=O)O,mol_005,acetic_acid
EOF

# バッチ予測
python scripts/predict_batch.py \
    --input molecules.csv \
    --output predictions.csv \
    --model models/qcgn2oei_minimal_best.pth \
    --batch-size 64 \
    --save-spectra spectra.npy

# 結果確認
head predictions.csv
```

### 例3: 学習済みモデルのダウンロード（将来）

```bash
# TODO: 学習済みモデルが公開されたら
# wget https://example.com/models/qcgn2oei_minimal_v4.2.pth
# mv qcgn2oei_minimal_v4.2.pth models/qcgn2oei_minimal_best.pth

# 予測実行
python scripts/predict_single.py "CCO" \
    --model models/qcgn2oei_minimal_best.pth \
    --visualize
```

## ⚡ パフォーマンス最適化

### GPU利用率の確認

```bash
# リアルタイムモニタリング
watch -n 1 nvidia-smi

# 学習中のGPU使用率を記録
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free \
    --format=csv -l 10 > gpu_usage.csv
```

### バッチサイズの調整

```bash
# メモリ不足の場合
python scripts/train_gnn_minimal.py \
    --batch-size 16  # 32 → 16に削減

# 推論時のバッチサイズ最適化
python scripts/predict_batch.py \
    --batch-size 128  # 推論は大きめ可能
```

### BDEキャッシュの活用

```bash
# 初回のみキャッシュ生成（約1時間）
python scripts/precompute_bde.py \
    --nist-msp data/NIST17.MSP \
    --bondnet-model models/bondnet_bde_db2_best.pth \
    --output data/processed/bde_cache/nist17_bde_cache.h5

# 以降は常にキャッシュを使用
python scripts/train_gnn_minimal.py \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    ...

python scripts/predict_batch.py \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    ...
```

## 🔍 トラブルシューティング

### CUDA Out of Memory

```bash
# 対処法1: バッチサイズ削減
--batch-size 16

# 対処法2: 混合精度無効化
# config.yamlで設定
gpu:
  mixed_precision: false

# 対処法3: CPU使用
--device cpu
```

### 学習が収束しない

```bash
# 学習率の調整
python scripts/train_gnn_minimal.py \
    --learning-rate 1e-5  # デフォルトは5e-5

# オプティマイザ変更
# train_gnn_minimal.pyを編集してAdamWに変更
```

### データが見つからない

```bash
# パスの確認
ls -la data/NIST17.MSP
ls -la data/external/bde-db2/

# 絶対パスで指定
python scripts/train_gnn_minimal.py \
    --nist-msp /absolute/path/to/data/NIST17.MSP
```

## 📊 結果の確認

### 評価レポート

```bash
# JSON形式で詳細確認
python -m json.tool results/evaluation/evaluation_report.json

# 重要メトリクスの抽出
jq '.metrics.cosine_similarity' results/evaluation/evaluation_report.json
jq '.metrics.top10_recall' results/evaluation/evaluation_report.json
```

### 可視化

```bash
# 評価時の可視化プロット
ls results/evaluation/prediction_sample_*.png

# 一括表示（ImageMagickが必要）
montage results/evaluation/prediction_sample_*.png \
    -tile 4x3 -geometry +5+5 \
    evaluation_summary.png
```

## 🧪 テストの実行

```bash
# 全テストの実行
python tests/test_evaluation_metrics.py
python tests/test_prediction.py
python tests/test_models.py
python tests/test_data_modules.py

# または一括実行（pytestが必要）
pytest tests/ -v
```

## 📚 追加リソース

- **完全ドキュメント**: [README.md](README.md)
- **技術仕様**: [docs/spec_v4.2_minimal_iterative.md](docs/spec_v4.2_minimal_iterative.md)
- **予測ガイド**: [docs/PREDICTION_GUIDE.md](docs/PREDICTION_GUIDE.md)
- **Issue報告**: https://github.com/turnDeep/NExtIMS/issues

## ⏱️ 予想所要時間

### Option A使用時（公式Pre-trained BonDNet）

| タスク | 時間 | 備考 |
|--------|------|------|
| 環境セットアップ | 30分 | 初回のみ |
| データ準備 | 15分 | NIST17入手含む |
| Phase 0（BDE環境） | **0時間** | Pre-trained使用 |
| Phase 1（データ準備） | 2時間 | BDEキャッシュ生成 |
| Phase 2（GNN学習） | 40時間 | Early stopping想定 |
| Phase 3（評価） | 2時間 | 可視化含む |
| Phase 5（推論） | 数秒-数分 | バッチサイズ依存 |
| **合計** | **約2日** | すぐに始められる！ |

### Option B使用時（再学習BonDNet）

| タスク | 時間 | 備考 |
|--------|------|------|
| 環境セットアップ | 30分 | 初回のみ |
| データ準備 | 15分 | NIST17入手含む |
| Phase 0（BDE環境） | **48-72時間** | BonDNet再学習 |
| Phase 1（データ準備） | 2時間 | BDEキャッシュ生成 |
| Phase 2（GNN学習） | 40時間 | Early stopping想定 |
| Phase 3（評価） | 2時間 | 可視化含む |
| Phase 5（推論） | 数秒-数分 | バッチサイズ依存 |
| **合計** | **約5-6日** | より高カバレッジ |

## 💡 ヒント

1. **並列実行**: Phase 0とPhase 1は独立しているため、Phase 0実行中にデータ準備可能
2. **チェックポイント**: 学習は定期的にチェックポイント保存（10 epochごと）
3. **早期停止**: Validation lossが20 epoch改善しない場合、自動停止
4. **ログ保存**: `nohup`や`screen`でバックグラウンド実行を推奨

## 🎓 学習のコツ

- **初回実行**: まず小規模データ（`--max-samples 1000`）でパイプライン確認
- **デバッグ**: `--epochs 5 --batch-size 8`で動作確認
- **本番実行**: パラメータをフルに戻して実行

---

**クイックスタートで問題があれば**: [Issue](https://github.com/turnDeep/NExtIMS/issues)を作成してください。

**最終更新**: 2025-12-03
**バージョン**: NExtIMS v4.2
