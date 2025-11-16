#!/usr/bin/env python3
"""
Pythonキャッシュとバイトコードをクリーンアップするスクリプト
循環インポートエラーが発生した場合に実行してください
"""

import os
import shutil
from pathlib import Path

def clean_python_cache(root_dir="."):
    """Pythonのキャッシュファイルを削除"""
    root_path = Path(root_dir).resolve()

    print(f"Cleaning Python cache in: {root_path}")
    print("-" * 60)

    # カウンター
    pyc_count = 0
    pycache_count = 0
    pyo_count = 0
    pytest_count = 0

    # .pyc ファイルを削除
    for pyc_file in root_path.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            pyc_count += 1
        except Exception as e:
            print(f"Error deleting {pyc_file}: {e}")

    # .pyo ファイルを削除
    for pyo_file in root_path.rglob("*.pyo"):
        try:
            pyo_file.unlink()
            pyo_count += 1
        except Exception as e:
            print(f"Error deleting {pyo_file}: {e}")

    # __pycache__ ディレクトリを削除
    for pycache_dir in root_path.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            pycache_count += 1
        except Exception as e:
            print(f"Error deleting {pycache_dir}: {e}")

    # .pytest_cache ディレクトリを削除
    for pytest_dir in root_path.rglob(".pytest_cache"):
        try:
            shutil.rmtree(pytest_dir)
            pytest_count += 1
        except Exception as e:
            print(f"Error deleting {pytest_dir}: {e}")

    print(f"✓ Deleted {pyc_count} .pyc files")
    print(f"✓ Deleted {pyo_count} .pyo files")
    print(f"✓ Deleted {pycache_count} __pycache__ directories")
    print(f"✓ Deleted {pytest_count} .pytest_cache directories")
    print("-" * 60)
    print("Cache cleanup complete!")
    print("\nPlease restart your Python environment and try again.")

if __name__ == "__main__":
    clean_python_cache()
