# NIST17 データ構造とロード方法

## 概要

NIST17データセットは2つの独立したデータソースで構成されています：

1. **NIST17.MSP**: マススペクトルデータ（ピーク情報のみ）
2. **mol_files/**: 化学構造データ（MOLファイル形式）

これらは **ID番号** でリンクされています。

## データ構造

```
data/
├── NIST17.MSP          # マススペクトルデータ
└── mol_files/          # 化学構造データ
    ├── ID12345.MOL     # ID=12345の化合物構造
    ├── ID12346.MOL     # ID=12346の化合物構造
    └── ...
```

### NIST17.MSP フォーマット

```
Name: Ethanol
MW: 46.07
Formula: C2H6O
ID: 12345
Num peaks: 4
46 100.0
31 80.0
45 60.0
27 40.0
```

**重要**: MSPファイルには化学構造情報（SMILES）が含まれていません。

### mol_files/ ディレクトリ

- 各MOLファイルは `ID{番号}.MOL` の形式で命名
- 例: MSPの `ID: 12345` は `ID12345.MOL` に対応
- MOLファイルには3D座標や結合情報が含まれる

## データロード実装

### parse_msp_file() 関数

```python
from src.data.nist_dataset import parse_msp_file

# MOLファイルディレクトリを指定してロード
entries = parse_msp_file(
    msp_path="data/NIST17.MSP",
    mol_files_dir="data/mol_files"  # 化学構造をMOLファイルからロード
)

# 結果: 各entryに以下が含まれる
# - name: 化合物名
# - mw: 分子量
# - formula: 分子式
# - id: ID番号
# - peaks: ピークリスト [(mz, intensity), ...]
# - smiles: 化学構造（MOLファイルから生成）
```

### 処理フロー

1. **MSPファイルパース**:
   - Name, MW, Formula, ID, Peaks を抽出

2. **MOLファイルロード**（mol_files_dir 指定時）:
   - 各entryの `ID` を使用
   - `mol_files/ID{id}.MOL` を読み込み
   - RDKitでMOL → SMILES変換
   - entryに `smiles` フィールドを追加

3. **エラーハンドリング**:
   - MOLファイルが見つからない場合: ログ警告、スキップ
   - MOLファイル読み込みエラー: ログ出力、スキップ
   - SMILESが既にMSPに存在: MOLファイルロードをスキップ

## 使用例

### トレーニングスクリプト

```python
# scripts/train_gnn_minimal.py
entries = parse_msp_file(
    msp_path="data/NIST17.MSP",
    mol_files_dir="data/mol_files"
)

for entry in entries:
    smiles = entry['smiles']  # MOLファイルから生成
    spectrum = peaks_to_spectrum(entry['peaks'])
    # ... トレーニング処理
```

### 評価スクリプト

```python
# scripts/evaluate_minimal.py
test_data = parse_msp_file(
    msp_path="data/NIST17.MSP",
    mol_files_dir="data/mol_files"
)

for entry in test_data:
    predicted = model.predict(entry['smiles'])
    actual = peaks_to_spectrum(entry['peaks'])
    # ... 評価処理
```

## セットアップ手順

### 1. ディレクトリ作成

```bash
mkdir -p data/mol_files
```

### 2. NIST17データ配置

```bash
# MSPファイルをコピー
cp /path/to/nist17/mainlib data/NIST17.MSP

# MOLファイルをコピー
cp -r /path/to/nist17/mol_files/* data/mol_files/
```

### 3. 確認

```bash
# MSPファイル確認
ls -lh data/NIST17.MSP

# MOLファイル数確認
ls data/mol_files/*.MOL | wc -l

# サンプル表示
ls data/mol_files/ | head -10
```

期待される出力:
```
ID12345.MOL
ID12346.MOL
ID12347.MOL
...
```

## トラブルシューティング

### エラー: "MOL files directory not found"

**原因**: `data/mol_files/` ディレクトリが存在しない

**解決策**:
```bash
mkdir -p data/mol_files
cp -r /path/to/mol_files/* data/mol_files/
```

### エラー: "Loaded 0 chemical structures from MOL files"

**原因**: MOLファイルの命名が正しくない

**確認**:
```bash
# ID番号を確認
grep "^ID:" data/NIST17.MSP | head -5

# MOLファイル名を確認
ls data/mol_files/ | head -5

# 形式: ID12345.MOL（IDの後に番号、.MOL拡張子）
```

### エラー: "Invalid entries: 100%"

**原因**: SMILES生成失敗またはMOLファイル破損

**解決策**:
1. MOLファイルの整合性確認
2. RDKitインストール確認: `pip install rdkit`
3. ログを確認してエラー詳細を調査

## パフォーマンス

### ロード時間

- **MSPパースのみ**: 5-10秒（30万エントリ）
- **MOLファイルロード込み**: 10-20分（30万エントリ、初回のみ）
- **キャッシュ使用**: 数秒（2回目以降）

### メモリ使用量

- **MSPデータ**: ~500MB
- **MOLロード後**: ~2GB（SMILES文字列含む）
- **グラフ生成後**: ~15-20GB（全データ、メモリ効率モード無効時）

### 推奨設定

32GB RAM環境では **memory_efficient_mode** を有効化:

```yaml
# config.yaml
data:
  memory_efficient_mode:
    enabled: true
    use_lazy_loading: true
```

## 技術詳細

### ID形式

- **MSP内**: `ID: 12345`（整数）
- **MOLファイル名**: `ID12345.MOL`（ID + 番号 + .MOL）
- **パース時**: 文字列として扱い、ファイル名構築に使用

### SMILESフォールバック

優先順位:
1. MSP内のSMILESフィールド（存在する場合）
2. MOLファイルから生成（mol_files_dir指定時）
3. スキップ（どちらも利用不可の場合）

### 後方互換性

`mol_files_dir` パラメータはオプション:

```python
# MOLファイルなし（MSP内にSMILESが必要）
entries = parse_msp_file("data/NIST17.MSP")

# MOLファイルあり（SMILESをMOLから生成）
entries = parse_msp_file("data/NIST17.MSP", mol_files_dir="data/mol_files")
```

## 参考資料

- **NIST17仕様**: [NIST Standard Reference Database 1A](https://www.nist.gov/srd/nist-standard-reference-database-1a)
- **MSPフォーマット**: NIST公式ドキュメント参照
- **MOLフォーマット**: [MDL Molfile specification](http://c4.cabrillo.edu/404/ctfile.pdf)

## 更新履歴

- **2025-12-03**: 初版作成、MOLファイルサポート追加
- **Commit**: 825f20e

---

**最終更新**: 2025-12-03
**バージョン**: NExtIMS v4.2
