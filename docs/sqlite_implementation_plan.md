# SQLite実装計画

## プロジェクト概要

BitSpecのプレトレーニングとファインチューニングデータをSQLiteで管理することで、パフォーマンス、信頼性、メモリ効率を大幅に改善する。

---

## Phase 1: ファインチューニングDB構築（優先度：最高）

### 目標
30万化合物のNISTデータをSQLite化し、ファイルI/Oと線形探索のボトルネックを解消。

### 成果物
1. `scripts/build_finetune_db.py` - データベース構築スクリプト
2. `src/data/sqlite_dataset.py` - SQLiteベースのデータセットクラス
3. `finetune_nist.db` - SQLiteデータベース（約1.45GB）

### 実装ステップ

#### Step 1.1: データベース構築スクリプト作成

**ファイル**: `scripts/build_finetune_db.py`

**機能**:
```python
def build_finetune_database(
    msp_file: str,
    mol_files_dir: str,
    db_path: str,
    rebuild: bool = False
):
    """
    NISTデータをSQLiteに変換

    処理フロー:
    1. データベース初期化（テーブル作成）
    2. MSPファイルのパース
       - compounds テーブルに挿入
       - spectra テーブルに挿入
    3. MOLファイルの読み込み
       - mol_structures テーブルに挿入
    4. グラフ特徴量の事前計算
       - RDKitで分子グラフ化
       - graph_features テーブルに挿入
    5. インデックス作成
    6. VACUUM（データベース最適化）
    """
```

**実装詳細**:
- バッチインサート（10,000件ずつ）でパフォーマンス向上
- トランザクション管理で整合性保証
- 進捗バーで処理状況を可視化（tqdm）
- エラーハンドリング（不正なMOLファイルをスキップ）

#### Step 1.2: SQLiteデータセットクラス作成

**ファイル**: `src/data/sqlite_dataset.py`

**クラス**: `SQLiteMassSpecDataset(Dataset)`

```python
class SQLiteMassSpecDataset(Dataset):
    """
    SQLiteベースのマススペクトルデータセット

    既存のMassSpecDatasetと同じインターフェースを提供
    """

    def __init__(
        self,
        db_path: str,
        split: str = None,  # 'train', 'val', 'test' または None（全データ）
        transform=None
    ):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.split = split
        self.transform = transform

        # インデックスリストを取得（高速化のため）
        if split:
            query = "SELECT compound_id FROM dataset_splits WHERE split = ?"
            self.compound_ids = [row[0] for row in self.conn.execute(query, (split,))]
        else:
            query = "SELECT id FROM compounds"
            self.compound_ids = [row[0] for row in self.conn.execute(query)]

    def __len__(self) -> int:
        return len(self.compound_ids)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor, Dict]:
        compound_id = self.compound_ids[idx]

        # 単一クエリで全データ取得（JOIN使用）
        query = """
            SELECT
                c.*,
                s.normalized_spectrum,
                g.node_features,
                g.edge_index,
                g.edge_attr
            FROM compounds c
            JOIN spectra s ON c.id = s.compound_id
            JOIN graph_features g ON c.id = g.compound_id
            WHERE c.id = ?
        """

        row = self.conn.execute(query, (compound_id,)).fetchone()

        # BLOBデータをnumpy配列に変換
        spectrum = decompress_array(row['normalized_spectrum'], ...)
        node_features = decompress_array(row['node_features'], ...)
        edge_index = decompress_array(row['edge_index'], ...)
        edge_attr = decompress_array(row['edge_attr'], ...)

        # PyTorch Geometricのグラフデータ作成
        graph_data = Data(
            x=torch.from_numpy(node_features),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            y=torch.from_numpy(spectrum)
        )

        # メタデータ
        metadata = {
            'name': row['name'],
            'formula': row['formula'],
            'mol_weight': row['mol_weight'],
            'cas_no': row['cas_no'],
            'id': compound_id
        }

        if self.transform:
            graph_data = self.transform(graph_data)

        return graph_data, torch.tensor(spectrum), metadata
```

**最適化ポイント**:
- 接続プーリング（check_same_thread=False）
- JOINで単一クエリに集約
- BLOBデータの遅延展開（必要な時のみ）

#### Step 1.3: データローダー統合

**ファイル**: `src/data/dataset.py` の拡張

```python
class MassSpecDataset(Dataset):
    """
    既存のデータセットにSQLiteサポートを追加
    """

    def __init__(
        self,
        msp_file: str = None,
        mol_files_dir: str = None,
        db_path: str = None,  # 新規追加
        use_sqlite: bool = False,  # 新規追加
        **kwargs
    ):
        if use_sqlite or db_path:
            # SQLiteモード
            self.backend = SQLiteMassSpecDataset(db_path, **kwargs)
        else:
            # 従来のファイルモード
            self.backend = self._init_file_backend(msp_file, mol_files_dir, **kwargs)

    def __getitem__(self, idx):
        return self.backend[idx]

    def __len__(self):
        return len(self.backend)
```

#### Step 1.4: 設定ファイル更新

**ファイル**: `config_pretrain.yaml`

```yaml
data:
  # 従来の設定（互換性維持）
  nist_msp_path: "data/NIST17.MSP"
  mol_files_dir: "data/mol_files"

  # SQLite設定（新規）
  use_sqlite: true  # SQLiteモードを有効化
  sqlite_db_path: "data/finetune_nist.db"

  # データセット分割の再現性
  random_seed: 42
```

#### Step 1.5: ファインチューニングスクリプト更新

**ファイル**: `scripts/finetune.py`

```python
# 64行目付近を変更
if self.config['data'].get('use_sqlite', False):
    # SQLiteモード
    logger.info(f"Using SQLite database: {self.config['data']['sqlite_db_path']}")
    dataset = SQLiteMassSpecDataset(
        db_path=self.config['data']['sqlite_db_path']
    )
else:
    # 従来のファイルモード
    logger.info("Using file-based dataset...")
    dataset = MassSpecDataset(
        msp_file=self.config['data']['nist_msp_path'],
        mol_files_dir=self.config['data']['mol_files_dir'],
        cache_file=cache_file
    )
```

### テスト計画

1. **ユニットテスト**: `tests/test_sqlite_dataset.py`
   - データベース構築
   - データ取得
   - バッチ処理

2. **統合テスト**:
   - 小規模データ（100化合物）で完全実行
   - ファイルモードとSQLiteモードの出力比較
   - パフォーマンスベンチマーク

3. **本番テスト**:
   - 30万化合物で構築
   - ファインチューニング1エポック実行
   - メモリ使用量・速度の測定

### 期待される成果

| 指標 | 現在 | SQLite化後 |
|------|------|-----------|
| データセット初期化 | 30-60秒 | 1-2秒 |
| ID検索速度 | 150ms | 0.1ms |
| メモリ使用量 | 数GB | 数十MB |
| pickleロード時間 | 5-10秒 | 不要 |

---

## Phase 2: プレトレーニングDB構築（優先度：高）

### 目標
PCQM4Mv2データセット（370万分子）をSQLite化し、HTTP依存とHTTP 500エラーを解消。

### 成果物
1. `scripts/build_pretrain_db.py` - PCQM4Mv2→SQLite変換スクリプト
2. `src/data/sqlite_pcqm4mv2_loader.py` - SQLiteベースのローダー
3. `pretrain_pcqm4mv2.db` - SQLiteデータベース（約2.5GB）

### 実装ステップ

#### Step 2.1: データベース構築スクリプト作成

**ファイル**: `scripts/build_pretrain_db.py`

```python
def build_pretrain_database(
    pcqm4mv2_root: str,
    db_path: str,
    splits: List[str] = ['train', 'val'],
    rebuild: bool = False
):
    """
    PCQM4Mv2をSQLiteに変換

    処理フロー:
    1. PyTorch Geometricで一度ダウンロード
    2. SQLiteデータベース初期化
    3. 各splitごとに処理
       - molecules テーブルに挿入
       - graph_structures テーブルに挿入
    4. インデックス作成
    5. 統計情報の保存（dataset_stats）
    """

    # PCQM4Mv2のダウンロード（初回のみ）
    for split in splits:
        dataset = PCQM4Mv2(root=pcqm4mv2_root, split=split)

        # バッチ処理で高速化
        for idx in tqdm(range(len(dataset)), desc=f"Processing {split}"):
            data = dataset[idx]

            # メタデータ抽出
            smiles = data.smiles if hasattr(data, 'smiles') else None
            homo_lumo_gap = data.y.item() if hasattr(data, 'y') else None

            # BLOBデータ圧縮
            node_features_blob = compress_array(data.x.numpy())
            edge_index_blob = compress_array(data.edge_index.numpy())
            edge_attr_blob = compress_array(data.edge_attr.numpy()) if hasattr(data, 'edge_attr') else None

            # データベースに挿入
            insert_molecule(idx, smiles, split, homo_lumo_gap, ...)
            insert_graph_structure(idx, node_features_blob, edge_index_blob, edge_attr_blob)
```

**実行時間見積もり**:
- 370万分子の処理: 約1-2時間（初回のみ）
- 以降はSQLiteから読み込み: 0.1秒

#### Step 2.2: SQLiteローダークラス作成

**ファイル**: `src/data/sqlite_pcqm4mv2_loader.py`

```python
class SQLitePCQM4Mv2Wrapper(Dataset):
    """
    SQLiteベースのPCQM4Mv2データセット

    HTTP依存を完全除去
    """

    def __init__(
        self,
        db_path: str = 'data/pretrain_pcqm4mv2.db',
        split: str = 'train',
        transform=None,
        node_feature_dim: int = 48,
        edge_feature_dim: int = 6
    ):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.split = split
        self.transform = transform
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim

        # インデックスリスト取得
        query = "SELECT idx FROM molecules WHERE split = ? ORDER BY idx"
        self.indices = [row[0] for row in self.conn.execute(query, (split,))]

        print(f"Loaded {len(self.indices)} molecules from SQLite (split: {split})")

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        mol_idx = self.indices[idx]

        # JOINで一括取得
        query = """
            SELECT
                m.homo_lumo_gap,
                g.node_features,
                g.edge_index,
                g.edge_attr
            FROM molecules m
            JOIN graph_structures g ON m.idx = g.idx
            WHERE m.idx = ?
        """

        row = self.conn.execute(query, (mol_idx,)).fetchone()

        # BLOBデータを展開
        node_features = decompress_array(row['node_features'], ...)
        edge_index = decompress_array(row['edge_index'], ...)
        edge_attr = decompress_array(row['edge_attr'], ...) if row['edge_attr'] else None

        # 特徴量の次元調整（既存のロジックを再利用）
        node_features = self._adapt_node_features(node_features)
        edge_attr = self._adapt_edge_features(edge_attr)

        # Data object作成
        data = Data(
            x=torch.from_numpy(node_features),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr) if edge_attr is not None else None
        )

        target = torch.tensor([row['homo_lumo_gap']], dtype=torch.float32)

        if self.transform:
            data = self.transform(data)

        return data, target
```

#### Step 2.3: プレトレーニングスクリプト更新

**ファイル**: `scripts/pretrain.py`

```python
# 59行目付近を変更
if self.config['pretraining'].get('use_sqlite', False):
    # SQLiteモード（HTTP 500エラー解消）
    logger.info("Using SQLite-based PCQM4Mv2 dataset")
    from src.data.sqlite_pcqm4mv2_loader import SQLitePCQM4Mv2DataLoader

    self.train_loader, self.val_loader, self.test_loader = SQLitePCQM4Mv2DataLoader.create_dataloaders(
        db_path=self.config['pretraining']['sqlite_db_path'],
        batch_size=self.config['pretraining']['batch_size'],
        num_workers=self.config['pretraining'].get('num_workers', 4),
        ...
    )
else:
    # 従来のHTTPベース
    logger.info("Using HTTP-based PCQM4Mv2 dataset")
    self.train_loader, self.val_loader, self.test_loader = PCQM4Mv2DataLoader.create_dataloaders(
        root=self.config['pretraining']['data_path'],
        ...
    )
```

### テスト計画

1. **小規模テスト**: 10,000分子で動作確認
2. **HTTP vs SQLite比較**: エポック3まで実行し、HTTP 500エラーが出ないか確認
3. **パフォーマンス測定**: バッチ読み込み速度、GPU使用率

### 期待される成果

| 指標 | 現在（HTTP） | SQLite化後 |
|------|------------|-----------|
| 初期化時間 | 10-30秒 | 0.1秒 |
| エポック2以降 | HTTP 500エラー | 安定動作 |
| バッチ取得速度 | 2-5秒 | 0.2秒 |

---

## Phase 3: 最適化とベンチマーク（優先度：中）

### SQLite最適化

**ファイル**: `src/data/sqlite_utils.py`

```python
def optimize_database(db_path: str):
    """
    SQLiteデータベースを最適化
    """
    conn = sqlite3.connect(db_path)

    # WALモード（並行読み取り性能向上）
    conn.execute("PRAGMA journal_mode = WAL")

    # 書き込み同期レベル調整
    conn.execute("PRAGMA synchronous = NORMAL")

    # キャッシュサイズ増加（10,000ページ = 約40MB）
    conn.execute("PRAGMA cache_size = 10000")

    # 一時データをメモリに
    conn.execute("PRAGMA temp_store = MEMORY")

    # データベースを最適化
    conn.execute("VACUUM")

    conn.close()
```

### ベンチマークスクリプト

**ファイル**: `scripts/benchmark_sqlite.py`

```python
def benchmark_comparison():
    """
    ファイルベース vs SQLiteのベンチマーク
    """

    # 1. データセット初期化時間
    # 2. ランダムアクセス速度（1000サンプル）
    # 3. シーケンシャルアクセス速度（10000サンプル）
    # 4. バッチ取得速度（バッチサイズ128）
    # 5. メモリ使用量

    # 結果をMarkdownテーブルで出力
```

---

## 実装スケジュール

### Week 1: ファインチューニングDB（Phase 1）

- **Day 1-2**: データベース構築スクリプト作成
- **Day 3-4**: SQLiteデータセットクラス実装
- **Day 5**: 統合とテスト
- **Day 6-7**: ベンチマークと調整

### Week 2: プレトレーニングDB（Phase 2）

- **Day 1-2**: PCQM4Mv2→SQLite変換スクリプト
- **Day 3**: SQLiteローダー実装
- **Day 4-5**: 統合とテスト
- **Day 6-7**: HTTP 500エラー解消の確認

### Week 3: 最適化とドキュメント（Phase 3）

- **Day 1-2**: SQLite最適化
- **Day 3-4**: ベンチマーク実施
- **Day 5-7**: ドキュメント整備、コードレビュー

---

## マイルストーン

### Milestone 1: ファインチューニングDB完成
- [ ] `build_finetune_db.py` 動作確認
- [ ] `SQLiteMassSpecDataset` テスト通過
- [ ] ファインチューニング1エポック成功
- [ ] パフォーマンス改善を確認

### Milestone 2: プレトレーニングDB完成
- [ ] `build_pretrain_db.py` 動作確認
- [ ] `SQLitePCQM4Mv2Wrapper` テスト通過
- [ ] プレトレーニング3エポック成功（HTTP 500エラーなし）
- [ ] パフォーマンス改善を確認

### Milestone 3: 本番運用開始
- [ ] ベンチマーク結果をドキュメント化
- [ ] 設定ファイルでSQLiteをデフォルト化
- [ ] READMEにセットアップ手順を追記
- [ ] 旧ファイルベースのコードを非推奨化

---

## リスク管理

### リスク1: データベースサイズが大きすぎる

**緩和策**:
- BLOB圧縮率を調整（zlib, lz4, zstd）
- 不要なデータを削除（例: SMILESは必要時のみ）
- 複数DBに分割（train.db, val.db, test.db）

### リスク2: SQLiteのパフォーマンスが期待値に届かない

**緩和策**:
- インデックスの最適化
- クエリのプロファイリング（EXPLAIN QUERY PLAN）
- バッチサイズの調整
- 必要に応じてDuckDBやParquetも検討

### リスク3: 既存コードとの互換性問題

**緩和策**:
- インターフェースを完全に揃える
- フラグで簡単に切り替え可能に
- テストで両方のモードを検証

---

## 成功基準

### 定量的指標
- データセット初期化: **10倍以上高速化**
- ID検索: **100倍以上高速化**
- メモリ使用量: **1/5以下に削減**
- HTTP 500エラー: **ゼロ**

### 定性的指標
- コードの可読性・保守性向上
- 開発体験の改善（SQLブラウザでデバッグ可能）
- 再現性の向上（データセット分割をDB管理）

---

## 次のステップ

1. **Phase 1の開始**: `scripts/build_finetune_db.py` の実装
2. **進捗報告**: 各マイルストーン達成時に報告
3. **フィードバック**: ベンチマーク結果を共有し、調整

実装を始めますか？
