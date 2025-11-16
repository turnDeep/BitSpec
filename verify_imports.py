#!/usr/bin/env python3
"""
インポートの整合性を確認するスクリプト
循環インポートがないか検証します
"""

import sys
import importlib
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """主要なモジュールのインポートをテスト"""
    print("Testing imports...")
    print("=" * 60)

    tests = [
        ("src.data.mol_parser", ["MOLParser", "NISTMSPParser"]),
        ("src.data.features", ["MolecularFeaturizer", "SubstructureFeaturizer"]),
        ("src.data.dataset", ["MassSpecDataset", "NISTDataLoader"]),
    ]

    success_count = 0
    fail_count = 0

    for module_name, expected_attrs in tests:
        try:
            print(f"\n{'─' * 60}")
            print(f"Testing: {module_name}")

            # モジュールをインポート
            module = importlib.import_module(module_name)
            print(f"  ✓ Module imported successfully")

            # 期待される属性をチェック
            for attr in expected_attrs:
                if hasattr(module, attr):
                    print(f"  ✓ Found: {attr}")
                else:
                    print(f"  ✗ Missing: {attr}")
                    fail_count += 1

            success_count += 1

        except ImportError as e:
            print(f"  ✗ Import failed: {e}")
            fail_count += 1
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Results: {success_count} succeeded, {fail_count} failed")

    if fail_count == 0:
        print("✓ All imports are working correctly!")
        return True
    else:
        print("✗ Some imports failed. Please check the errors above.")
        return False

def check_circular_imports():
    """循環インポートの可能性をチェック"""
    print("\n" + "=" * 60)
    print("Checking for potential circular imports...")
    print("=" * 60)

    src_data_path = project_root / "src" / "data"

    if not src_data_path.exists():
        print(f"✗ Directory not found: {src_data_path}")
        return False

    # 各ファイルのインポート文を確認
    for py_file in src_data_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        print(f"\nChecking: {py_file.name}")

        with open(py_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            line = line.strip()
            # 自己インポートをチェック
            if py_file.stem in line and (line.startswith("from") or line.startswith("import")):
                if f"from src.data.{py_file.stem} import" in line:
                    print(f"  ✗ Line {i}: Self-import detected!")
                    print(f"     {line}")
                    return False

            # パッケージからの絶対インポートをチェック(推奨されない)
            if "from src.data.dataset import" in line and py_file.stem == "dataset":
                print(f"  ✗ Line {i}: Self-import detected!")
                print(f"     {line}")
                return False

    print("  ✓ No circular imports detected")
    return True

if __name__ == "__main__":
    print("BitSpec Import Verification")
    print("=" * 60)

    # 循環インポートチェック
    circular_check = check_circular_imports()

    # インポートテスト
    import_check = test_imports()

    print("\n" + "=" * 60)
    if circular_check and import_check:
        print("✓ All checks passed!")
        sys.exit(0)
    else:
        print("✗ Some checks failed. Please review the output above.")
        sys.exit(1)
