#!/usr/bin/env python3
"""
NExtIMS v4.2: BonDNet Retraining on BDE-db2

Retrains BonDNet on the BDE-db2 dataset (531,244 BDEs from 65,540 molecules)
to improve BDE prediction coverage and accuracy for NIST17 EI-MS dataset.

Why retrain BonDNet?
    - Original BonDNet: Trained on 64,000 molecules
    - BDE-db2: 531,244 BDEs (8x larger dataset)
    - NIST17 coverage: 95% → 99%+ (halogen-containing compounds)
    - Expected MAE: ~0.51 kcal/mol (similar or better)

Workflow:
    1. Download BDE-db2 dataset:
       python scripts/download_bde_db2.py --output data/external/bde-db2

    2. Convert to BonDNet format:
       python scripts/convert_bde_db2_to_bondnet.py \\
           --input data/external/bde-db2/bde-db2.csv \\
           --output data/processed/bondnet_training/

    3. Retrain BonDNet (this script):
       python scripts/train_bondnet_bde_db2.py \\
           --data-dir data/processed/bondnet_training/ \\
           --output models/bondnet_bde_db2.pth

    4. Pre-compute BDE with custom model:
       python scripts/precompute_bde.py \\
           --nist-msp data/NIST17.MSP \\
           --model models/bondnet_bde_db2.pth \\
           --output data/processed/bde_cache/nist17_bde_cache.h5

Hardware Requirements:
    - GPU: RTX 5070 Ti 16GB (or similar)
    - Training time: ~2-3 days (with optimizations)
    - Disk: ~10 GB (dataset + model checkpoints)

Requirements:
    - BonDNet: pip install git+https://github.com/mjwen/bondnet.git
    - DGL: pip install dgl
    - PyTorch >= 1.10.0
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
import json
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_bondnet_installation():
    """Check if BonDNet is installed and get installation path"""
    try:
        import bondnet
        bondnet_path = Path(bondnet.__file__).parent.parent
        logger.info(f"BonDNet found: {bondnet_path}")
        return bondnet_path
    except ImportError:
        logger.error("BonDNet is not installed!")
        logger.error("Install with:")
        logger.error("  pip install git+https://github.com/mjwen/bondnet.git")
        logger.error("Or clone and install:")
        logger.error("  git clone https://github.com/mjwen/bondnet.git")
        logger.error("  cd bondnet && pip install -e .")
        sys.exit(1)


def check_training_data(data_dir: Path) -> bool:
    """
    Check if training data exists and is valid

    Required files:
        - molecules.sdf
        - molecule_attributes.yaml
        - reactions.yaml
    """
    required_files = [
        'molecules.sdf',
        'molecule_attributes.yaml',
        'reactions.yaml'
    ]

    missing_files = []
    for filename in required_files:
        filepath = data_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
        else:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ {filename} ({file_size_mb:.1f} MB)")

    if missing_files:
        logger.error("Missing required files:")
        for filename in missing_files:
            logger.error(f"  ✗ {filename}")
        logger.error("\nPlease run data preparation first:")
        logger.error("  python scripts/download_bde_db2.py")
        logger.error("  python scripts/convert_bde_db2_to_bondnet.py")
        return False

    return True


def create_training_config(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = 'cuda'
) -> Path:
    """
    Create BonDNet training configuration file

    Args:
        data_dir: Directory containing training data
        output_dir: Directory for model checkpoints
        epochs: Number of training epochs
        batch_size: Batch size (64 for RTX 5070 Ti 16GB)
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'

    Returns:
        config_path: Path to generated config file
    """
    config = {
        # Data paths
        'dataset_location': str(data_dir),
        'molecules_file': str(data_dir / 'molecules.sdf'),
        'molecule_attributes_file': str(data_dir / 'molecule_attributes.yaml'),
        'reactions_file': str(data_dir / 'reactions.yaml'),

        # Model architecture
        'embedding_size': 128,
        'num_gnn_layers': 4,
        'gnn_hidden_size': 128,
        'num_fc_layers': 3,
        'fc_hidden_size': 128,
        'conv_fn': 'GatedGCNConv',  # BonDNet default

        # Training parameters
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': 1e-4,
        'scheduler': 'reduce_on_plateau',
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,

        # GPU optimization
        'device': device,
        'num_workers': 4,
        'pin_memory': True if device == 'cuda' else False,

        # Checkpointing
        'checkpoint_dir': str(output_dir / 'checkpoints'),
        'save_interval': 10,  # Save every 10 epochs
        'early_stopping_patience': 30,

        # Logging
        'log_dir': str(output_dir / 'logs'),
        'log_interval': 100,  # Log every 100 batches

        # Model output
        'output_path': str(output_dir / 'bondnet_bde_db2.pth')
    }

    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Created training config: {config_path}")
    return config_path


def run_bondnet_training(
    data_dir: Path,
    config_path: Path,
    bondnet_path: Path,
    device: str = 'cuda'
):
    """
    Run BonDNet training using its native training script

    Args:
        data_dir: Training data directory
        config_path: Configuration file path
        bondnet_path: BonDNet installation path
        device: 'cuda' or 'cpu'
    """
    # Find BonDNet training script
    training_script = bondnet_path / 'bondnet' / 'scripts' / 'train.py'

    if not training_script.exists():
        logger.warning(f"BonDNet training script not found at {training_script}")
        logger.warning("Using alternative training approach...")
        run_bondnet_training_alternative(data_dir, config_path, device)
        return

    logger.info(f"Running BonDNet training script: {training_script}")

    # Prepare command
    cmd = [
        sys.executable,
        str(training_script),
        '--config', str(config_path)
    ]

    logger.info(f"Command: {' '.join(cmd)}")

    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def run_bondnet_training_alternative(
    data_dir: Path,
    config_path: Path,
    device: str = 'cuda'
):
    """
    Alternative training approach using BonDNet API directly

    This is a fallback if the training script is not available.
    """
    logger.info("Using alternative training approach (BonDNet API)")

    try:
        import torchdata
    except ImportError:
        logger.error("Required dependency 'torchdata' is missing!")
        logger.error("Please install it with:")
        logger.error("  pip install torchdata")
        sys.exit(1)

    try:
        from bondnet.data.dataset import ReactionNetworkDataset
        try:
            from bondnet.model.training_utils import train_model
        except ImportError:
            # Fallback if train_model is not in training_utils (e.g. newer bondnet versions)
            # We might need to implement training loop or import it from somewhere else
            # For now, we report this as a specific error.
            logger.error("Could not import 'train_model' from 'bondnet.model.training_utils'.")
            logger.error("Your BonDNet installation might be different from expected.")
            raise

        from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
        import torch
        import yaml

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load dataset
        logger.info("Loading dataset...")
        dataset = ReactionNetworkDataset(
            molecules_file=config['molecules_file'],
            molecule_attributes_file=config['molecule_attributes_file'],
            reactions_file=config['reactions_file']
        )

        logger.info(f"Dataset loaded: {len(dataset)} reactions")

        # Create model
        logger.info("Creating model...")
        model = GatedGCNReactionNetwork(
            in_feats=dataset.feature_size,
            embedding_size=config['embedding_size'],
            gated_num_layers=config['num_gnn_layers'],
            gated_hidden_size=config['gnn_hidden_size'],
            fc_num_layers=config['num_fc_layers'],
            fc_hidden_size=config['fc_hidden_size']
        )

        # Move to device
        device_obj = torch.device(device)
        model = model.to(device_obj)

        # Train
        logger.info("Starting training...")
        train_model(
            model=model,
            dataset=dataset,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            device=device_obj,
            checkpoint_dir=config['checkpoint_dir'],
            log_dir=config['log_dir']
        )

        # Save final model
        logger.info(f"Saving final model to {config['output_path']}")
        torch.save(model.state_dict(), config['output_path'])

        logger.info("Training complete!")

    except Exception as e:
        logger.error(f"Alternative training failed: {e}")
        logger.error("\nPlease install BonDNet from source:")
        logger.error("  git clone https://github.com/mjwen/bondnet.git")
        logger.error("  cd bondnet && pip install -e .")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Retrain BonDNet on BDE-db2 dataset"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing BonDNet training data (SDF + YAML files)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/bondnet_bde_db2.pth',
        help='Output model path (default: models/bondnet_bde_db2.pth)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs (default: 200)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64 for RTX 5070 Ti 16GB)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda or cpu (default: cuda)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint (optional)'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("NExtIMS v4.2: BonDNet Retraining on BDE-db2")
    logger.info("="*80)

    # Check BonDNet installation
    bondnet_path = check_bondnet_installation()

    # Check training data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info(f"Data directory: {data_dir}")
    logger.info("Checking training data files...")

    if not check_training_data(data_dir):
        sys.exit(1)

    logger.info("✓ All training data files present")

    # Create output directory
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model will be saved to: {output_path}")

    # Training parameters
    logger.info("")
    logger.info("Training parameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Device: {args.device}")

    # Estimate training time
    if args.device == 'cuda':
        estimated_time_hours = 48  # ~2 days for RTX 5070 Ti
        logger.info(f"  Estimated time: ~{estimated_time_hours} hours (~{estimated_time_hours/24:.1f} days)")
    else:
        logger.warning("  CPU training will be very slow (not recommended)")

    logger.info("")

    # Create training config
    config_path = create_training_config(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )

    # Run training
    logger.info("")
    logger.info("Starting BonDNet training...")
    logger.info("="*80)

    run_bondnet_training(
        data_dir=data_dir,
        config_path=config_path,
        bondnet_path=bondnet_path,
        device=args.device
    )

    logger.info("")
    logger.info("="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Model saved to: {output_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Evaluate model performance:")
    logger.info(f"     (check {output_dir / 'logs'} for training metrics)")
    logger.info("")
    logger.info("  2. Pre-compute BDE for NIST17 with custom model:")
    logger.info("     python scripts/precompute_bde.py \\")
    logger.info("       --nist-msp data/NIST17.MSP \\")
    logger.info(f"       --model {output_path} \\")
    logger.info("       --output data/processed/bde_cache/nist17_bde_cache_custom.h5")
    logger.info("="*80)


if __name__ == '__main__':
    main()
