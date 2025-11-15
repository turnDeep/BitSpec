# SQLite データベース設計

## 概要

BitSpecのプレトレーニングとファインチューニングデータをSQLiteで管理するための設計ドキュメント。

## 1. プレトレーニング用データベース: `pretrain_pcqm4mv2.db`

### 目的
PCQM4Mv2データセット（約370万分子）の量子化学的性質を高速アクセス可能な形式で管理。

### 現在の問題点
- **HTTP依存**: PyTorch GeometricのPCQM4Mv2クラスがOGBサーバーからHTTP経由でダウンロード
- **HTTP 500エラー**: エポック2以降でpersistent_workersがHTTP接続タイムアウトを引き起こす
- **遅い起動**: 毎回ネットワーク経由でメタデータを取得
- **メモリキャッシュ**: オンデマンドキャッシングは辞書形式でメモリ効率が悪い

### スキーマ設計

```sql
-- 分子メタデータ
CREATE TABLE molecules (
    idx INTEGER PRIMARY KEY,           -- データセット内のインデックス
    smiles TEXT,                        -- SMILES表記
    num_atoms INTEGER,                  -- 原子数
    num_bonds INTEGER,                  -- 結合数
    split TEXT,                         -- 'train', 'val', 'test'
    homo_lumo_gap REAL,                 -- HOMO-LUMO gap (eV)
    INDEX idx_split (split),
    INDEX idx_num_atoms (num_atoms)
);

-- グラフ構造データ（圧縮BLOB）
CREATE TABLE graph_structures (
    idx INTEGER PRIMARY KEY,
    node_features BLOB,                 -- ノード特徴量 (N x 48) - numpy array圧縮
    edge_index BLOB,                    -- エッジインデックス (2 x E) - numpy array圧縮
    edge_attr BLOB,                     -- エッジ特徴量 (E x 6) - numpy array圧縮
    FOREIGN KEY (idx) REFERENCES molecules(idx)
);

-- データセット統計（メタデータ）
CREATE TABLE dataset_stats (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- インデックス最適化
CREATE INDEX idx_homo_lumo ON molecules(homo_lumo_gap);
```

### データサイズ見積もり（370万分子）

| テーブル | 行数 | 1行あたり | 合計サイズ |
|---------|------|----------|-----------|
| molecules | 3.7M | 100バイト | 370 MB |
| graph_structures | 3.7M | 500バイト（圧縮） | 1.85 GB |
| インデックス | - | - | 300 MB |
| **合計** | - | - | **約2.5 GB** |

### パフォーマンス改善見積もり

| 操作 | 現在（HTTP） | SQLite | 改善率 |
|------|------------|--------|--------|
| データセット初期化 | 10-30秒 | 0.1秒 | **100-300倍** |
| 1サンプル取得 | 5-10ms | 0.5ms | **10-20倍** |
| バッチ取得（512） | 2-5秒 | 0.2秒 | **10-25倍** |
| エポック2以降 | HTTP 500エラー | 安定動作 | **信頼性向上** |

---

## 2. ファインチューニング用データベース: `finetune_nist.db`

### 目的
NISTマススペクトルデータ（30万化合物）をファイルI/Oから解放し、高速アクセスを実現。

### 現在の問題点
- **線形探索**: `get_compound_by_id()` が O(n) の線形探索 (mol_parser.py:184)
- **個別ファイルI/O**: 30万個のMOLファイルを順次読み込み (dataset.py:84)
- **巨大pickle**: 数GBのキャッシュファイル (dataset.py:64)
- **全データメモリ展開**: compounds配列全体をメモリに保持 (mol_parser.py:76)

### スキーマ設計

```sql
-- 化合物メタデータ
CREATE TABLE compounds (
    id TEXT PRIMARY KEY,                -- ID200001, ID200002, ...
    name TEXT,                          -- 化合物名
    inchikey TEXT,                      -- InChIKey
    formula TEXT,                       -- 分子式
    mol_weight REAL,                    -- 分子量
    exact_mass REAL,                    -- 精密質量
    cas_no TEXT,                        -- CAS番号
    ri REAL,                            -- Retention Index
    INDEX idx_inchikey (inchikey),
    INDEX idx_formula (formula),
    INDEX idx_mol_weight (mol_weight)
);

-- スペクトルデータ
CREATE TABLE spectra (
    compound_id TEXT PRIMARY KEY,
    num_peaks INTEGER,                  -- ピーク数
    mz_values BLOB,                     -- m/z値配列（圧縮）
    intensities BLOB,                   -- 強度配列（圧縮）
    normalized_spectrum BLOB,           -- 正規化済みスペクトル（1000次元）
    FOREIGN KEY (compound_id) REFERENCES compounds(id)
);

-- MOL構造データ
CREATE TABLE mol_structures (
    compound_id TEXT PRIMARY KEY,
    mol_block TEXT,                     -- MOLファイルの内容
    num_atoms INTEGER,
    num_bonds INTEGER,
    smiles TEXT,                        -- RDKitで生成したSMILES
    has_3d_coords BOOLEAN,              -- 3D座標を持つか
    FOREIGN KEY (compound_id) REFERENCES compounds(id)
);

-- グラフ特徴量（事前計算）
CREATE TABLE graph_features (
    compound_id TEXT PRIMARY KEY,
    node_features BLOB,                 -- ノード特徴量 (N x 48) - numpy array圧縮
    edge_index BLOB,                    -- エッジインデックス (2 x E)
    edge_attr BLOB,                     -- エッジ特徴量 (E x 6)
    FOREIGN KEY (compound_id) REFERENCES compounds(id)
);

-- データセット分割（再現性のため）
CREATE TABLE dataset_splits (
    compound_id TEXT PRIMARY KEY,
    split TEXT,                         -- 'train', 'val', 'test'
    seed INTEGER,                       -- 乱数シード
    FOREIGN KEY (compound_id) REFERENCES compounds(id),
    INDEX idx_split (split)
);
```

### データサイズ見積もり（30万化合物）

| テーブル | 行数 | 1行あたり | 合計サイズ |
|---------|------|----------|-----------|
| compounds | 300K | 150バイト | 45 MB |
| spectra | 300K | 800バイト | 240 MB |
| mol_structures | 300K | 2.5KB | 750 MB |
| graph_features | 300K | 1KB（圧縮） | 300 MB |
| dataset_splits | 300K | 50バイト | 15 MB |
| インデックス | - | - | 100 MB |
| **合計** | - | - | **約1.45 GB** |

### パフォーマンス改善見積もり

| 操作 | 現在（ファイルI/O） | SQLite | 改善率 |
|------|------------------|--------|--------|
| MSPファイル解析 | 全体をメモリ読込 | インデックス検索 | **メモリ効率10倍** |
| ID検索 | 150ms（線形探索） | 0.1ms | **1,500倍** |
| 100化合物取得 | 11ms（100ファイルI/O） | 0.7ms（1クエリ） | **15倍** |
| pickleロード | 5-10秒（数GB） | 不要 | **起動時間削減** |
| データセット初期化 | 30-60秒 | 1-2秒 | **30倍** |

---

## 3. データ移行戦略

### Phase 1: ファインチューニングDB構築（優先度：高）

**理由**:
- ローカルファイルベースなので移行が簡単
- 即座にパフォーマンス改善が体感できる
- 30万化合物で実証済み

**ステップ**:
1. `scripts/build_finetune_db.py` を作成
   - NIST17.MSPをパース → `compounds` + `spectra` テーブル
   - MOLファイルを読込 → `mol_structures` テーブル
   - RDKitで特徴量抽出 → `graph_features` テーブル

2. `src/data/sqlite_dataset.py` を作成
   - `MassSpecDataset` のSQLite版
   - pickleキャッシュの代わりにSQLiteクエリ

3. `scripts/finetune.py` を更新
   - `MassSpecDataset` → `SQLiteMassSpecDataset` に切り替え

### Phase 2: プレトレーニングDB構築（優先度：中）

**理由**:
- PCQM4Mv2はPyTorch Geometric経由で自動ダウンロード
- 370万分子の大規模データ
- HTTP 500エラー問題を根本解決

**ステップ**:
1. `scripts/build_pretrain_db.py` を作成
   - PCQM4Mv2データセットを一度ダウンロード
   - SQLiteに変換して永続化

2. `src/data/sqlite_pcqm4mv2_loader.py` を作成
   - `PCQM4Mv2Wrapper` のSQLite版
   - HTTP依存を完全除去

3. `scripts/pretrain.py` を更新
   - `PCQM4Mv2DataLoader` → `SQLitePCQM4Mv2DataLoader` に切り替え

### Phase 3: 最適化とベンチマーク（優先度：低）

1. SQLiteクエリの最適化
   ```sql
   PRAGMA journal_mode = WAL;      -- 並行読み取り性能向上
   PRAGMA synchronous = NORMAL;     -- 書き込み性能向上
   PRAGMA cache_size = 10000;       -- キャッシュサイズ増加
   PRAGMA temp_store = MEMORY;      -- 一時データをメモリに
   ```

2. BLOB圧縮の最適化
   - numpy配列のzlib圧縮
   - 圧縮率 vs 展開速度のトレードオフ調整

3. バッチ読み込みの最適化
   - `SELECT ... WHERE id IN (?, ?, ...)`
   - prefetchingとの連携

---

## 4. 実装上の注意点

### BLOB データの扱い

```python
import numpy as np
import sqlite3
import zlib

# 保存
def compress_array(arr: np.ndarray) -> bytes:
    return zlib.compress(arr.tobytes())

# 読み込み
def decompress_array(blob: bytes, dtype, shape) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), dtype=dtype).reshape(shape)
```

### バッチクエリ

```python
def get_batch(db_conn, compound_ids: List[str]) -> List[Dict]:
    placeholders = ','.join(['?'] * len(compound_ids))
    query = f"""
        SELECT c.*, s.normalized_spectrum, g.node_features, g.edge_index, g.edge_attr
        FROM compounds c
        JOIN spectra s ON c.id = s.compound_id
        JOIN graph_features g ON c.id = g.compound_id
        WHERE c.id IN ({placeholders})
    """
    cursor = db_conn.execute(query, compound_ids)
    return cursor.fetchall()
```

### トランザクション管理

```python
# 読み取り専用（デフォルト）
conn = sqlite3.connect('finetune_nist.db', check_same_thread=False)

# DataLoaderの複数ワーカーで共有
# SQLiteはread-onlyなら安全に並行アクセス可能
```

---

## 5. 期待される効果

### プレトレーニング
- **HTTP 500エラー解消**: ネットワーク依存を完全除去
- **起動時間**: 10-30秒 → 0.1秒（**100-300倍**）
- **安定性**: エポック間のHTTPタイムアウトなし

### ファインチューニング
- **初期化時間**: 30-60秒 → 1-2秒（**30倍**）
- **ID検索**: 150ms → 0.1ms（**1,500倍**）
- **メモリ使用量**: 数GB → 数十MB（**1/10以下**）

### 開発体験
- **デバッグ効率**: SQLブラウザでデータ確認可能
- **再現性**: データセット分割をDBに保存
- **拡張性**: 新しいメタデータを追加しやすい

---

## 6. 互換性維持

既存のコードを壊さないための設計:

```python
# 既存のAPI
dataset = MassSpecDataset(
    msp_file="data/NIST17.MSP",
    mol_files_dir="data/mol_files",
    cache_file="data/processed/cache.pkl"
)

# 新しいAPI（同じインターフェース）
dataset = SQLiteMassSpecDataset(
    db_path="data/finetune_nist.db"
)

# または、自動切り替え
dataset = MassSpecDataset(
    msp_file="data/NIST17.MSP",
    mol_files_dir="data/mol_files",
    use_sqlite=True,  # フラグで切り替え
    db_path="data/finetune_nist.db"
)
```

---

## 7. ロールバック計画

SQLite移行後に問題が発生した場合:

1. **設定フラグで即座に戻せる**
   ```yaml
   data:
     use_sqlite: false  # ファイルベースに戻す
   ```

2. **両方のデータソースを並行維持**
   - SQLiteデータベース
   - 元のMSP/MOLファイル

3. **段階的移行**
   - 開発環境で十分テスト
   - 本番環境で並行運用
   - 問題なければ完全移行

---

## まとめ

SQLite化により:
- **パフォーマンス**: 10-1,500倍の高速化
- **信頼性**: HTTP 500エラー解消
- **メモリ効率**: 1/10以下に削減
- **開発体験**: デバッグ・拡張が容易に

30万化合物でも370万分子でも、SQLiteは十分なスケーラビリティを持っています。
