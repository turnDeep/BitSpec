#!/usr/bin/env python3
"""
PCQM4Mv2 Dataset Download Script

Downloads the PCQM4Mv2 dataset from OGB (Open Graph Benchmark).
The dataset contains ~3.8M molecules for pretraining.

Usage:
    python scripts/download_pcqm4mv2.py [--data-dir DATA_DIR]

The dataset will be downloaded to the specified directory (default: data/pcqm4mv2).
Note: The download may take several hours and requires ~8GB of disk space.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_pcqm4mv2(data_dir: str = "data/pcqm4mv2"):
    """
    Download PCQM4Mv2 dataset using OGB library.

    Args:
        data_dir: Directory to save the dataset

    Returns:
        True if successful, False otherwise
    """
    try:
        from ogb.lsc import PCQM4Mv2Dataset
        from ogb.utils import smiles2graph as original_smiles2graph
        import ogb.utils.mol

        logger.info("=" * 80)
        logger.info("PCQM4Mv2 Dataset Download")
        logger.info("=" * 80)
        logger.info(f"Download directory: {data_dir}")
        logger.info("Note: This may take several hours on first run (~8GB dataset)")
        logger.info("=" * 80)

        # Create data directory
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        # Patch smiles2graph to handle invalid SMILES gracefully
        invalid_smiles_count = 0

        def patched_smiles2graph(smiles_string):
            """Patched version that handles invalid SMILES gracefully."""
            nonlocal invalid_smiles_count
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:
                invalid_smiles_count += 1
                if invalid_smiles_count <= 10:  # Log first 10 invalid SMILES
                    logger.warning(f"Invalid SMILES (#{invalid_smiles_count}): {smiles_string[:50]}...")
                elif invalid_smiles_count == 11:
                    logger.warning("(Suppressing further invalid SMILES warnings...)")

                # Return a minimal valid graph structure
                return {
                    'edge_index': [[], []],
                    'edge_feat': [],
                    'node_feat': [[0] * 9],  # Single dummy atom with 9 features
                    'num_nodes': 1
                }

            # Call original function
            return original_smiles2graph(smiles_string)

        # Apply the patch
        ogb.utils.mol.smiles2graph = patched_smiles2graph

        # Download dataset
        logger.info("Downloading PCQM4Mv2 dataset...")
        logger.info("Note: Invalid SMILES will be replaced with dummy graphs")
        dataset = PCQM4Mv2Dataset(root=str(data_path), only_smiles=False)

        # Restore original function
        ogb.utils.mol.smiles2graph = original_smiles2graph

        if invalid_smiles_count > 0:
            logger.warning(f"⚠️  Found {invalid_smiles_count:,} invalid SMILES (replaced with dummy graphs)")

        logger.info("=" * 80)
        logger.info(f"✅ Download complete!")
        logger.info(f"Total molecules: {len(dataset):,}")
        logger.info(f"Dataset location: {data_path.absolute()}")
        logger.info("=" * 80)

        # Verify SMILES file
        smiles_file = data_path / "pcqm4m-v2" / "raw" / "data.csv.gz"
        if smiles_file.exists():
            logger.info(f"✅ SMILES file verified: {smiles_file}")

            # Load and verify
            import pandas as pd
            df = pd.read_csv(smiles_file, compression='gzip')
            logger.info(f"Total SMILES entries: {len(df):,}")
            logger.info(f"Sample SMILES: {df['smiles'].iloc[0]}")
        else:
            logger.warning(f"⚠️  SMILES file not found: {smiles_file}")

        logger.info("=" * 80)
        logger.info("Dataset ready for training!")
        logger.info("You can now run: python scripts/train_pipeline.py --config config.yaml")
        logger.info("=" * 80)

        return True

    except ImportError as e:
        logger.error("=" * 80)
        logger.error("❌ OGB library not found!")
        logger.error("Please install OGB first:")
        logger.error("  pip install ogb")
        logger.error("=" * 80)
        return False

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ Download failed: {e}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Download PCQM4Mv2 dataset for pretraining"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/pcqm4mv2",
        help="Directory to save the dataset (default: data/pcqm4mv2)"
    )

    args = parser.parse_args()

    # Download dataset
    success = download_pcqm4mv2(args.data_dir)

    if not success:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
