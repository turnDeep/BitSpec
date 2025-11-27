#!/usr/bin/env python3
"""
Phase 0: BDE Precomputation Script

Pre-computes Bond Dissociation Energy (BDE) values for PCQM4Mv2 dataset
using ALFABET, storing results in HDF5 cache for Phase 1 training.

This script creates a large-scale BDE database (up to 93.5M BDE values)
that will be used during Phase 1 Teacher pretraining.

Usage:
    # Subset (500K molecules, ~20-30 min on CPU)
    python scripts/precompute_bde.py --max-samples 500000

    # Full dataset (3.74M molecules, ~2-3 hours on CPU)
    python scripts/precompute_bde.py --max-samples 0

Requirements:
    - TensorFlow 2.10+ (ALFABET dependency)
    - ALFABET >= 0.4.1
    - Note: RTX 5070 Ti GPU support is limited in TensorFlow 2.x
          CPU execution is recommended and performant enough
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import h5py
import numpy as np
from tqdm import tqdm
from rdkit import Chem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        logger.info(f"TensorFlow version: {tf_version}")

        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                logger.info(f"Found GPU: {gpu}")
                # Get GPU details
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    compute_capability = gpu_details.get('compute_capability', 'Unknown')
                    logger.info(f"  Compute Capability: {compute_capability}")

                    # Check if Blackwell (RTX 50 series, sm_120)
                    if compute_capability == (12, 0):
                        logger.warning(
                            "Detected Blackwell GPU (RTX 50 series, sm_120). "
                            "TensorFlow 2.x has limited support for sm_120. "
                            "Expect JIT compilation delays or errors. "
                            "CPU execution is recommended."
                        )
                except Exception as e:
                    logger.warning(f"Could not get GPU details: {e}")

            # Set memory growth to avoid OOM
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Set memory growth for {gpu}")
                except RuntimeError as e:
                    logger.warning(f"Could not set memory growth: {e}")
        else:
            logger.info("No GPU found. Using CPU (recommended for ALFABET).")

        # Optimize CPU performance
        num_threads = os.cpu_count() or 8
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        logger.info(f"TensorFlow CPU threads: {num_threads}")

    except ImportError:
        logger.error("TensorFlow is not installed. Please run: pip install tensorflow>=2.10.0")
        sys.exit(1)

    try:
        from alfabet import model as alfabet_model
        logger.info("ALFABET loaded successfully")
        return alfabet_model
    except ImportError:
        logger.error("ALFABET is not installed. Please run: pip install alfabet>=0.4.1")
        sys.exit(1)


def load_pcqm4mv2_dataset(data_dir: str, max_samples: int = 0):
    """Load PCQM4Mv2 dataset.

    Args:
        data_dir: Directory containing PCQM4Mv2 data
        max_samples: Maximum number of samples (0 = all)

    Returns:
        List of (smiles, mol) tuples
    """
    try:
        from ogb.lsc import PCQM4Mv2Dataset
    except ImportError:
        logger.error("OGB is not installed. Please run: pip install ogb>=1.3.6")
        sys.exit(1)

    logger.info(f"Loading PCQM4Mv2 dataset from {data_dir}")
    dataset = PCQM4Mv2Dataset(root=data_dir, only_smiles=True)

    total_samples = len(dataset)
    if max_samples > 0:
        num_samples = min(max_samples, total_samples)
        logger.info(f"Using subset: {num_samples:,} / {total_samples:,} molecules")
    else:
        num_samples = total_samples
        logger.info(f"Using full dataset: {total_samples:,} molecules")

    # Extract SMILES and convert to RDKit molecules
    molecules = []
    invalid_count = 0

    for idx in tqdm(range(num_samples), desc="Loading molecules"):
        # PCQM4Mv2Dataset with only_smiles=True returns data in various formats
        data = dataset[idx]

        # Extract SMILES string from different data formats
        if isinstance(data, dict):
            # Dictionary format: {'smiles': 'CC...'}
            smiles = data.get('smiles', data.get('SMILES', ''))
        elif isinstance(data, tuple):
            # Tuple format: (smiles,) or (graph, smiles)
            # PCQM4Mv2 with only_smiles=True typically returns (smiles,)
            smiles = data[0] if len(data) > 0 else ''
        elif isinstance(data, str):
            # Direct string format
            smiles = data
        else:
            # Unknown format
            logger.warning(f"Unexpected data type at index {idx}: {type(data)}")
            smiles = str(data) if data else ''

        if not smiles or not isinstance(smiles, str):
            invalid_count += 1
            continue

        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            molecules.append((smiles, mol))
        else:
            invalid_count += 1

    logger.info(f"Loaded {len(molecules):,} valid molecules ({invalid_count} invalid)")
    return molecules


def predict_bde_batch(alfabet_model, smiles_list: List[str], batch_size: int = 32) -> Dict[str, Dict[int, float]]:
    """Predict BDE values in batches.

    Args:
        alfabet_model: ALFABET model instance
        smiles_list: List of SMILES strings
        batch_size: Batch size for prediction

    Returns:
        Dictionary mapping SMILES to BDE values {bond_idx: bde_value}
    """
    results = {}

    # Process in batches
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]

        try:
            # ALFABET expects list of SMILES
            predictions = alfabet_model.predict(batch)

            # Store results
            for smiles, bde_dict in zip(batch, predictions):
                if bde_dict:
                    results[smiles] = bde_dict

        except Exception as e:
            logger.warning(f"Batch prediction failed: {e}. Trying individual predictions.")

            # Fallback: predict individually
            for smiles in batch:
                try:
                    pred = alfabet_model.predict([smiles])
                    if pred and len(pred) > 0:
                        results[smiles] = pred[0]
                except Exception as e2:
                    logger.warning(f"Failed to predict BDE for {smiles}: {e2}")

    return results


def save_bde_cache(bde_data: Dict[str, Dict[int, float]], output_path: str):
    """Save BDE cache to HDF5 file.

    Args:
        bde_data: Dictionary mapping SMILES to BDE values
        output_path: Output HDF5 file path
    """
    logger.info(f"Saving BDE cache to {output_path}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Count total BDE values
    total_bde_values = sum(len(bde_dict) for bde_dict in bde_data.values())

    with h5py.File(output_path, 'w') as f:
        # Store metadata
        f.attrs['num_molecules'] = len(bde_data)
        f.attrs['num_bde_values'] = total_bde_values
        f.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Store BDE data
        for smiles, bde_dict in tqdm(bde_data.items(), desc="Writing HDF5"):
            # Create group for each molecule
            grp = f.create_group(smiles)

            # Store BDE values for each bond
            for bond_idx, bde_value in bde_dict.items():
                grp.create_dataset(str(bond_idx), data=bde_value, dtype='float32')

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    logger.info(f"BDE cache saved successfully:")
    logger.info(f"  - Molecules: {len(bde_data):,}")
    logger.info(f"  - Total BDE values: {total_bde_values:,}")
    logger.info(f"  - File size: {file_size_mb:.2f} MB")


def compute_bde_statistics(bde_data: Dict[str, Dict[int, float]]):
    """Compute and log BDE statistics.

    Args:
        bde_data: Dictionary mapping SMILES to BDE values
    """
    all_bde_values = []
    bonds_per_molecule = []

    for bde_dict in bde_data.values():
        bde_values = list(bde_dict.values())
        all_bde_values.extend(bde_values)
        bonds_per_molecule.append(len(bde_values))

    if all_bde_values:
        logger.info("BDE Statistics:")
        logger.info(f"  - Mean BDE: {np.mean(all_bde_values):.2f} kcal/mol")
        logger.info(f"  - Std BDE: {np.std(all_bde_values):.2f} kcal/mol")
        logger.info(f"  - Min BDE: {np.min(all_bde_values):.2f} kcal/mol")
        logger.info(f"  - Max BDE: {np.max(all_bde_values):.2f} kcal/mol")
        logger.info(f"  - Avg bonds/molecule: {np.mean(bonds_per_molecule):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: Pre-compute BDE values for PCQM4Mv2 dataset"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/pcqm4mv2',
        help='PCQM4Mv2 dataset directory (default: data/pcqm4mv2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/bde_cache/bde_cache.h5',
        help='Output HDF5 file path (default: data/processed/bde_cache/bde_cache.h5)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=500000,
        help='Maximum number of molecules to process (0 = all, default: 500000)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for BDE prediction (default: 32)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing cache file'
    )

    args = parser.parse_args()

    # Check if output file already exists
    if os.path.exists(args.output) and not args.force:
        logger.error(f"Output file already exists: {args.output}")
        logger.error("Use --force to overwrite")
        sys.exit(1)

    logger.info("="*80)
    logger.info("Phase 0: BDE Pre-computation for PCQM4Mv2")
    logger.info("="*80)

    # Check dependencies and load ALFABET
    alfabet_model = check_dependencies()

    # Load PCQM4Mv2 dataset
    molecules = load_pcqm4mv2_dataset(args.data_dir, args.max_samples)

    # Estimate time
    estimated_time_min = len(molecules) / 25000 * 60  # ~25K molecules/hour
    logger.info(f"Estimated time: {estimated_time_min:.1f} minutes ({estimated_time_min/60:.1f} hours)")

    # Predict BDE values
    logger.info("Computing BDE values with ALFABET...")
    start_time = time.time()

    bde_data = {}
    failed_count = 0

    for smiles, mol in tqdm(molecules, desc="Computing BDEs"):
        try:
            # Predict BDE for this molecule
            predictions = alfabet_model.predict([smiles])

            if predictions and len(predictions) > 0:
                bde_dict = predictions[0]
                if bde_dict:
                    bde_data[smiles] = bde_dict
                else:
                    failed_count += 1
            else:
                failed_count += 1

        except Exception as e:
            logger.warning(f"Failed to predict BDE for {smiles}: {e}")
            failed_count += 1

    elapsed_time = time.time() - start_time
    logger.info(f"BDE computation completed in {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
    logger.info(f"Success: {len(bde_data):,} / {len(molecules):,} molecules ({failed_count} failed)")

    # Compute statistics
    compute_bde_statistics(bde_data)

    # Save to HDF5
    save_bde_cache(bde_data, args.output)

    logger.info("="*80)
    logger.info("Phase 0 completed successfully!")
    logger.info(f"BDE cache: {args.output}")
    logger.info("You can now run Phase 1 training with:")
    logger.info("  python scripts/train_teacher.py --config config_pretrain.yaml --phase pretrain")
    logger.info("="*80)


if __name__ == '__main__':
    main()
