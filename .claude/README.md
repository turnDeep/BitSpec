# NExtIMS Claude Code ガイド

このディレクトリにはNExtIMS (Next Extended NEIMS) プロジェクト用のClaude Code設定が含まれています。

## プロジェクト概要

**NExtIMS** は次世代の電子衝撃イオン化マススペクトル（EI-MS）予測システムです。Teacher-Student Knowledge DistillationとMixture of Experts (MoE)アーキテクチャを使用して、化学構造から高精度な質量スペクトルを予測します。

## ディレクトリ構造

```
.claude/
├── claude.json           # プロジェクト設定
├── commands/             # カスタムコマンド
│   ├── setup.md         # 環境セットアップ
│   ├── preprocess.md    # データ前処理
│   ├── train.md         # モデルトレーニング
│   ├── predict.md       # 予測実行
│   └── test.md          # テスト実行
└── README.md            # このファイル
```

## 利用可能なコマンド

### /setup
開発環境のセットアップを行います。依存関係のインストール、パッケージのインストール、必要なディレクトリの作成を実行します。

### /preprocess
NIST MSP形式のデータを前処理します。トレーニング、検証、テストデータに分割します。

### /train
モデルのトレーニングを実行します。config.yamlの設定を使用してGCNモデルをトレーニングします。

### /predict
訓練済みモデルを使用して予測を実行します。SMILES文字列から質量スペクトルを生成します。

### /test
プロジェクトのテストスクリプトを実行します。データローディングとトレーニングのテストを行います。

## Claude Codeの使い方

1. プロジェクトディレクトリでClaude Codeを起動：
   ```bash
   claude-code
   ```

2. スラッシュコマンドを実行：
   ```
   /setup
   /preprocess
   /train
   ```

3. 質問や開発タスクを直接入力：
   ```
   "モデルアーキテクチャを説明してください"
   "新しいデータセットローダーを追加してください"
   "トレーニングのハイパーパラメータを最適化してください"
   ```

## カスタマイズ

新しいコマンドを追加するには、`.claude/commands/`ディレクトリに新しいMarkdownファイルを作成してください。ファイル名がコマンド名になります。

例：`.claude/commands/evaluate.md`を作成すると、`/evaluate`コマンドが使用可能になります。

## プロジェクト固有のガイドライン

- PyTorch 2.7+とCUDA 12.8+を使用
- RTX 50シリーズGPUに最適化されたコードを維持
- Mixed Precision Trainingを活用
- データは`data/processed/`ディレクトリに配置
- モデルチェックポイントは`checkpoints/`ディレクトリに保存

## トラブルシューティング

### コマンドが見つからない場合
`.claude/commands/`ディレクトリにコマンドファイルが存在することを確認してください。

### 設定が反映されない場合
`claude.json`の構文が正しいことを確認してください（有効なJSON形式であること）。

## サポート

問題や質問がある場合は、GitHubのIssueで報告してください：
https://github.com/yourusername/mass-spectrum-prediction/issues
