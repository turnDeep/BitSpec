#!/bin/bash
# Pythonキャッシュとバイトコードをクリーンアップするスクリプト

echo "Cleaning Python cache files..."

# .pycファイルを削除
find . -type f -name "*.pyc" -delete
echo "Deleted .pyc files"

# __pycache__ディレクトリを削除
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "Deleted __pycache__ directories"

# .pyo ファイルを削除
find . -type f -name "*.pyo" -delete
echo "Deleted .pyo files"

# pytest cache を削除
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
echo "Deleted .pytest_cache directories"

echo "Cache cleanup complete!"
echo "Please restart your Python environment and try again."
