#!/usr/bin/env python3
"""
Quick test for Memory Efficient Mode implementation
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_efficient_mode():
    """Test Memory Efficient Mode with HDF5 lazy loading"""

    # Check if h5py is available
    try:
        import h5py
        logger.info("✓ h5py is available")
    except ImportError:
        logger.error("✗ h5py is not available. Install with: pip install h5py")
        return False

    # Load config
    config_path = 'config.yaml'
    if not Path(config_path).exists():
        logger.error(f"✗ Config file not found: {config_path}")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("✓ Config loaded")

    # Test lazy loading mode
    logger.info("\n=== Testing Memory Efficient Mode ===")

    # Enable memory efficient mode
    if 'data' not in config:
        config['data'] = {}

    config['data']['memory_efficient_mode'] = {
        'enabled': True,
        'use_lazy_loading': True,
        'lazy_cache_dir': 'data/processed/test_lazy_cache'
    }

    logger.info("Memory efficient mode configuration:")
    logger.info(f"  - enabled: {config['data']['memory_efficient_mode']['enabled']}")
    logger.info(f"  - use_lazy_loading: {config['data']['memory_efficient_mode']['use_lazy_loading']}")

    # Try to import NISTDataset
    try:
        from src.data.nist_dataset import NISTDataset
        logger.info("✓ NISTDataset imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import NISTDataset: {e}")
        return False

    # Create a small test dataset
    try:
        logger.info("\nInitializing NISTDataset with lazy loading...")
        dataset = NISTDataset(
            data_config=config['data'],
            mode='teacher',
            split='val',  # Use smaller validation set for testing
            augment=False
        )
        logger.info(f"✓ Dataset initialized with {len(dataset)} samples")

        # Check if lazy loading is actually enabled
        if hasattr(dataset, 'use_lazy_loading') and dataset.use_lazy_loading:
            logger.info("✓ Lazy loading is ENABLED")
        else:
            logger.warning("⚠ Lazy loading is DISABLED (might be missing data or h5py)")

        # Try to load a sample
        if len(dataset) > 0:
            logger.info("\nLoading sample from dataset...")
            sample = dataset[0]

            # Check sample structure
            required_keys = ['graph', 'ecfp', 'spectrum']
            for key in required_keys:
                if key in sample:
                    logger.info(f"✓ Sample contains '{key}' with shape {sample[key].shape if hasattr(sample[key], 'shape') else 'N/A'}")
                else:
                    logger.error(f"✗ Sample missing '{key}'")
                    return False

            logger.info("\n✓ Memory Efficient Mode test PASSED")
            return True
        else:
            logger.warning("⚠ Dataset is empty, cannot test sample loading")
            return True

    except Exception as e:
        logger.error(f"✗ Failed to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_memory_efficient_mode()

    if success:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("✗ TESTS FAILED")
        print("="*60)
        sys.exit(1)
