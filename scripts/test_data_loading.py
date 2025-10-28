#!/usr/bin/env python3
"""
データ読み込みのテストスクリプト（PyTorch不要）
10個のMOLファイルが正しく読み込めるかを検証
"""

import sys
from pathlib import Path
import logging

# パスの追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# 直接インポート（__init__.pyを回避）
import importlib.util

# mol_parserを直接読み込み
spec = importlib.util.spec_from_file_location(
    "mol_parser",
    Path(__file__).parent.parent / "src" / "data" / "mol_parser.py"
)
mol_parser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mol_parser_module)

MOLParser = mol_parser_module.MOLParser
NISTMSPParser = mol_parser_module.NISTMSPParser

# RDKitのみで特徴量抽出の簡易版
from rdkit import Chem

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_loading():
    """データ読み込みのテスト"""

    logger.info("=" * 60)
    logger.info("データ読み込みテスト開始")
    logger.info("=" * 60)

    # 1. MSPファイルの解析
    logger.info("\n1. MSPファイルの解析")
    logger.info("-" * 60)

    msp_file = "data/NIST17.MSP"
    if not Path(msp_file).exists():
        logger.error(f"MSPファイルが見つかりません: {msp_file}")
        return False

    msp_parser = NISTMSPParser()
    compounds = msp_parser.parse_file(msp_file)

    logger.info(f"解析された化合物数: {len(compounds)}")

    # 最初の10個のID
    test_ids = [f"200{str(i).zfill(3)}" for i in range(1, 11)]
    logger.info(f"テスト対象ID: {test_ids}")

    # 2. MOLファイルの読み込み
    logger.info("\n2. MOLファイルの読み込みテスト")
    logger.info("-" * 60)

    mol_files_dir = Path("data/mol_files")
    if not mol_files_dir.exists():
        logger.error(f"MOLファイルディレクトリが見つかりません: {mol_files_dir}")
        return False

    mol_parser = MOLParser()
    successful_loads = 0
    failed_loads = []

    for test_id in test_ids:
        mol_file = mol_files_dir / f"ID{test_id}.MOL"

        if mol_file.exists():
            try:
                mol = mol_parser.parse_file(str(mol_file))

                # 分子情報を取得
                formula = mol_parser.get_molecular_formula()
                mol_weight = mol_parser.get_molecular_weight()
                num_atoms = mol.GetNumAtoms()
                num_bonds = mol.GetNumBonds()

                logger.info(f"✓ ID{test_id}:")
                logger.info(f"    分子式: {formula}")
                logger.info(f"    分子量: {mol_weight:.2f}")
                logger.info(f"    原子数: {num_atoms}")
                logger.info(f"    結合数: {num_bonds}")

                successful_loads += 1

            except Exception as e:
                logger.error(f"✗ ID{test_id} の読み込みエラー: {e}")
                failed_loads.append(test_id)
        else:
            logger.warning(f"⚠ ID{test_id} のMOLファイルが見つかりません")
            failed_loads.append(test_id)

    # 3. 分子構造の簡易解析
    logger.info("\n3. 分子構造の簡易解析")
    logger.info("-" * 60)

    # 最初の成功したMOLファイルで分子構造を確認
    for test_id in test_ids:
        mol_file = mol_files_dir / f"ID{test_id}.MOL"

        if mol_file.exists():
            try:
                mol = mol_parser.parse_file(str(mol_file))

                # 基本的な分子情報
                if mol.GetNumAtoms() > 0:
                    logger.info(f"ID{test_id} の分子構造:")
                    logger.info(f"  原子タイプ: {set([atom.GetSymbol() for atom in mol.GetAtoms()])}")
                    logger.info(f"  環構造: {'あり' if mol.GetRingInfo().NumRings() > 0 else 'なし'}")
                    logger.info(f"  環の数: {mol.GetRingInfo().NumRings()}")

                break  # 1つ成功したら終了

            except Exception as e:
                logger.error(f"分子構造解析エラー (ID{test_id}): {e}")
                continue

    # 4. スペクトルデータのテスト
    logger.info("\n4. スペクトルデータテスト")
    logger.info("-" * 60)

    for test_id in test_ids[:3]:  # 最初の3つだけ
        # MSPからスペクトルを取得
        compound = None
        for comp in compounds:
            if comp.get('ID') == test_id:
                compound = comp
                break

        if compound and 'Spectrum' in compound:
            spectrum_data = compound['Spectrum']

            logger.info(f"ID{test_id} のスペクトル:")
            logger.info(f"  名前: {compound.get('Name', 'N/A')}")
            logger.info(f"  ピーク数: {len(spectrum_data)}")

            # スペクトルを正規化
            normalized = msp_parser.normalize_spectrum(
                spectrum_data,
                max_mz=1000,
                mz_bin_size=1.0
            )

            logger.info(f"  正規化後の次元: {len(normalized)}")
            logger.info(f"  非ゼロ要素数: {(normalized > 0).sum()}")
            logger.info(f"  最大強度: {normalized.max():.4f}")

    # 結果サマリー
    logger.info("\n" + "=" * 60)
    logger.info("テスト結果サマリー")
    logger.info("=" * 60)
    logger.info(f"成功: {successful_loads} / {len(test_ids)}")
    logger.info(f"失敗: {len(failed_loads)} / {len(test_ids)}")

    if failed_loads:
        logger.warning(f"失敗したID: {failed_loads}")

    if successful_loads >= 5:  # 最低5個成功
        logger.info("\n✓ データ読み込みテスト合格")
        logger.info("Dev Containerでの完全なトレーニングに進めます")
        return True
    else:
        logger.error("\n✗ データ読み込みテスト不合格")
        logger.error("十分なデータが読み込めませんでした")
        return False


if __name__ == "__main__":
    try:
        success = test_data_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
