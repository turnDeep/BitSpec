# 循環インポートエラーの修正方法

## エラー内容

```
ImportError: cannot import name 'MassSpecDataset' from partially initialized module 'src.data.dataset'
(most likely due to a circular import)
```

## 原因

このエラーは、Pythonのバイトコードキャッシュ(`.pyc`ファイル)に古いバージョンのコードが残っている可能性があります。
以前のバージョンで循環インポートが存在していた場合、キャッシュがクリアされないと問題が継続します。

## 修正手順

### 方法1: Pythonスクリプトでキャッシュをクリーン(推奨)

```bash
python clean_cache.py
```

### 方法2: シェルスクリプトでキャッシュをクリーン

```bash
bash clean_cache.sh
```

### 方法3: 手動でクリーン

```bash
# .pycファイルを削除
find . -type f -name "*.pyc" -delete

# __pycache__ディレクトリを削除
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# .pyoファイルを削除
find . -type f -name "*.pyo" -delete
```

### 方法4: Dockerコンテナを使用している場合

```bash
# コンテナを停止
docker-compose down

# キャッシュをクリーン
python clean_cache.py
# または
bash clean_cache.sh

# コンテナを再ビルド
docker-compose build --no-cache

# コンテナを起動
docker-compose up
```

## インポートの検証

修正後、以下のスクリプトでインポートが正しく動作するか確認できます:

```bash
python verify_imports.py
```

このスクリプトは以下をチェックします:
- 循環インポートの有無
- 主要なモジュールのインポート可否
- 期待されるクラスや関数の存在

## 再発防止

1. **キャッシュを定期的にクリーン**
   - 大きな変更を加えた後は `clean_cache.py` を実行

2. **Dockerを使用している場合**
   - コードを大幅に変更した場合は `--no-cache` オプションでリビルド

3. **開発環境の同期**
   - Git の最新変更をプル: `git pull origin main`
   - 仮想環境を再作成することも検討

## 現在のコード状態

現在のリポジトリのコードには循環インポートは存在しません:

- `src/data/dataset.py`: 自己インポートなし ✓
- `src/data/__init__.py`: 正しくインポートを定義 ✓
- `src/data/features.py`: 問題なし ✓
- `src/data/mol_parser.py`: 問題なし ✓

## トラブルシューティング

### それでもエラーが出る場合

1. Pythonプロセスを完全に終了
2. IDE/エディタを再起動
3. 仮想環境を再作成:
   ```bash
   deactivate  # 仮想環境を無効化
   rm -rf venv  # 仮想環境を削除
   python -m venv venv  # 仮想環境を再作成
   source venv/bin/activate  # 仮想環境を有効化
   pip install -r requirements.txt  # 依存関係を再インストール
   ```

4. Dockerボリュームをクリーン(Dockerを使用している場合):
   ```bash
   docker-compose down -v
   docker-compose up --build
   ```

## 問い合わせ

問題が解決しない場合は、以下の情報と共にissueを作成してください:
- エラーの完全なトレースバック
- `python verify_imports.py` の出力
- 使用しているPythonのバージョン (`python --version`)
- 使用している環境(Docker、venv、conda等)
