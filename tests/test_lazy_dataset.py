#!/usr/bin/env python3
# tests/test_lazy_dataset.py
"""
Unit tests for LazyMassSpecDataset

Tests the memory-efficient lazy loading implementation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.lazy_dataset import LazyMassSpecDataset, estimate_memory_usage


class TestLazyDataset(unittest.TestCase):
    """Test LazyMassSpecDataset functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test files"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_memory_estimation(self):
        """Test memory usage estimation function"""
        # Test traditional mode
        trad = estimate_memory_usage(100000, mode='traditional')
        self.assertIn('dataset_mb', trad)
        self.assertIn('total_mb', trad)
        self.assertGreater(trad['total_mb'], 0)

        # Test lazy mode
        lazy = estimate_memory_usage(100000, mode='lazy')
        self.assertIn('dataset_mb', lazy)
        self.assertIn('hdf5_disk_mb', lazy)
        self.assertIn('total_mb', lazy)

        # Lazy should use less memory
        self.assertLess(lazy['dataset_mb'], trad['dataset_mb'])
        self.assertLess(lazy['total_mb'], trad['total_mb'])

    def test_memory_estimation_300k(self):
        """Test memory estimation for full NIST17 (300k compounds)"""
        lazy = estimate_memory_usage(300000, mode='lazy')

        # Should fit in 32GB RAM
        total_gb = lazy['total_mb'] / 1024
        self.assertLess(total_gb, 32,
                       f"Total memory {total_gb:.1f}GB should be less than 32GB")

        # Dataset should be around 150MB
        self.assertLess(lazy['dataset_mb'], 200,
                       f"Dataset memory {lazy['dataset_mb']:.1f}MB should be around 150MB")

    def test_memory_reduction_ratio(self):
        """Test that lazy loading provides significant memory reduction"""
        dataset_size = 300000

        trad = estimate_memory_usage(dataset_size, mode='traditional')
        lazy = estimate_memory_usage(dataset_size, mode='lazy')

        # Dataset memory reduction should be 50x or more
        dataset_reduction = trad['dataset_mb'] / lazy['dataset_mb']
        self.assertGreater(dataset_reduction, 50,
                          f"Dataset reduction {dataset_reduction:.1f}x should be >50x")

        # Total memory reduction should be 2x or more
        total_reduction = trad['total_mb'] / lazy['total_mb']
        self.assertGreater(total_reduction, 2,
                          f"Total reduction {total_reduction:.1f}x should be >2x")

    def test_lazy_dataset_mock(self):
        """Test LazyMassSpecDataset with mock data"""
        # Note: This test requires actual NIST data to run fully
        # Here we just test the import and class structure

        # Check that LazyMassSpecDataset exists and is a class
        self.assertTrue(hasattr(LazyMassSpecDataset, '__init__'))
        self.assertTrue(hasattr(LazyMassSpecDataset, '__getitem__'))
        self.assertTrue(hasattr(LazyMassSpecDataset, '__len__'))

        # Check that required methods exist
        self.assertTrue(hasattr(LazyMassSpecDataset, '_build_cache'))
        self.assertTrue(hasattr(LazyMassSpecDataset, '_load_cache'))
        self.assertTrue(hasattr(LazyMassSpecDataset, '_estimate_memory_usage'))


class TestMemoryBenchmarkIntegration(unittest.TestCase):
    """Integration tests for memory benchmark tool"""

    def test_benchmark_script_exists(self):
        """Test that benchmark script exists and is executable"""
        benchmark_script = Path(__file__).parent.parent / "scripts" / "benchmark_memory.py"
        self.assertTrue(benchmark_script.exists(),
                       "benchmark_memory.py should exist")

    def test_lazy_dataset_module_exists(self):
        """Test that lazy_dataset module exists"""
        lazy_dataset_module = Path(__file__).parent.parent / "src" / "data" / "lazy_dataset.py"
        self.assertTrue(lazy_dataset_module.exists(),
                       "lazy_dataset.py should exist")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
