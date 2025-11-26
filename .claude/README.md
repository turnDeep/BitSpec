# NExtIMS v2.0 Claude Code ガイド

このディレクトリにはNExtIMS v2.0プロジェクト用のClaude Code設定が含まれています。

## プロジェクト概要

NExtIMS v2.0（Neural EI-MS Prediction with Knowledge Distillation）は、電子衝撃イオン化マススペクトル（EI-MS）を予測する次世代深層学習システムです。Teacher-Student Knowledge DistillationとMixture of Experts (MoE)アーキテクチャを使用して、高精度かつ高速な質量スペクトル予測を実現します。

### 主な特徴
- **Teacher-Student Knowledge Distillation**: GNN+ECFP HybridのTeacherから軽量MoE StudentへのSchemas転移
- **BDE Regression Pretraining**: Bond Dissociation Energy（結合解離エネルギー）を学習タスクとして使用
- **Mixture of Experts (MoE)**: 4つの専門家ネットワーク（芳香族、脂肪族、複素環、一般）
- **Uncertainty-Aware Distillation**: MC Dropoutによる不確実性を考慮した知識蒸留
- **RTX 5070 Ti最適化**: 16GB VRAMに最適化、Mixed Precision Training対応

## ディレクトリ構造

```
.claude/
├── claude.json           # プロジェクト設定
├── commands/             # カスタムコマンド
│   ├── setup.md         # 環境セットアップ
│   ├── train.md         # モデルトレーニング
│   └── predict.md       # 予測実行
└── README.md            # このファイル
```

## 利用可能なコマンド

### /setup
開発環境のセットアップを行います。依存関係のインストール、パッケージのインストール、必要なディレクトリの作成を実行します。

### /train
モデルのトレーニングを実行します。3段階のトレーニングプロセス（Teacher事前学習→Teacherファインチューニング→Student蒸留）を実行します。

### /predict
訓練済みモデルを使用して予測を実行します。SMILES文字列から質量スペクトルを生成します。

## Claude Codeの使い方

1. プロジェクトディレクトリでClaude Codeを起動：
   ```bash
   claude-code
   ```

2. スラッシュコマンドを実行：
   ```
   /setup
   /train
   /predict
   ```

3. 質問や開発タスクを直接入力：
   ```
   "BDE事前学習の実装方法を教えてください"
   "Student蒸留の損失関数を最適化してください"
   "メモリ効率的なデータローディングを実装してください"
   ```

## カスタマイズ

新しいコマンドを追加するには、`.claude/commands/`ディレクトリに新しいMarkdownファイルを作成してください。ファイル名がコマンド名になります。

例：`.claude/commands/evaluate.md`を作成すると、`/evaluate`コマンドが使用可能になります。

## プロジェクト固有のガイドライン

- **PyTorch 2.7+とCUDA 12.8+を使用**
- **RTX 5070 Ti（16GB VRAM）に最適化されたコードを維持**
- **Mixed Precision Training (FP16)を活用**
- **メモリ効率的データローディング**: 32GB RAMでNIST17全データ（30万化合物）対応
- **データは`data/`ディレクトリに配置**:
  - `data/NIST17.msp`: NIST EI-MSスペクトル
  - `data/mol_files/`: 対応するMOLファイル
  - `data/pcqm4mv2/`: PCQM4Mv2データセット（事前学習用）
  - `data/processed/`: 前処理済みデータとキャッシュ
- **モデルチェックポイントは`checkpoints/`ディレクトリに保存**:
  - `checkpoints/teacher/`: Teacherモデル
  - `checkpoints/student/`: Studentモデル

## 3段階トレーニングプロセス

NExtIMS v2.0は以下の3段階で学習します：

1. **Phase 0（オプション）**: BDE事前計算
   - PCQM4Mv2全分子のBDE値を一括計算（ALFABET使用）
   - HDF5データベース化（史上最大規模、9,350万BDE値）

2. **Phase 1**: Teacher事前学習（PCQM4Mv2）
   - BDE Regression（推奨）またはBond Masking
   - サブセット（50万分子）: ~16時間
   - 全データ（3.74M分子）: ~3.5日

3. **Phase 2**: Teacherファインチューニング（NIST EI-MS）
   - MC Dropoutによる不確実性推定
   - ~12-18時間

4. **Phase 3**: Student知識蒸留
   - Uncertainty-Aware KD
   - GradNorm適応重み付け
   - ~8-12時間

## トラブルシューティング

### コマンドが見つからない場合
`.claude/commands/`ディレクトリにコマンドファイルが存在することを確認してください。

### 設定が反映されない場合
`claude.json`の構文が正しいことを確認してください（有効なJSON形式であること）。

### GPU/CUDAの問題
```bash
# CUDAの確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# RTX 5070 Ti対応の確認
python -c "from src.utils.rtx50_compat import setup_gpu_environment; setup_gpu_environment()"
```

### メモリ不足
```bash
# メモリ使用量推定ツール
python scripts/benchmark_memory.py --mode estimate --ram_gb 32

# config.yamlでバッチサイズを調整
# training.batch_size: 16  # 32から16に減らす
```

## ドキュメント

- **README.md**: プロジェクト全体の説明
- **docs/NEIMS_v2_SYSTEM_SPECIFICATION.md**: 完全システム仕様書
- **docs/BDE_PRETRAINING_IMPLEMENTATION_GUIDE.md**: BDE事前学習実装ガイド
- **docs/PCQM4Mv2_TRAINING_TIME_ESTIMATE.md**: 訓練時間見積もり
- **FIX_TORCH_SCATTER_CUDA.md**: GPU互換性修正記録

## サポート

問題や質問がある場合は、GitHubのIssueで報告してください：
https://github.com/turnDeep/NExtIMS/issues
