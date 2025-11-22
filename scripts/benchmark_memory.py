#!/usr/bin/env python3
# scripts/benchmark_memory.py
"""
Memory Benchmark Tool for NIST17 Full Dataset Training

Estimates memory usage for different configurations and validates
compatibility with 32GB RAM systems.

Usage:
    # Estimate memory for full NIST17
    python scripts/benchmark_memory.py --mode estimate --ram_gb 32

    # Benchmark actual dataset loading
    python scripts/benchmark_memory.py --mode benchmark --dataset_size 100000

    # Compare lazy vs traditional
    python scripts/benchmark_memory.py --mode compare --max_samples 300000
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.lazy_dataset import estimate_memory_usage


def estimate_configuration(ram_gb: int = 32, dataset_size: int = 300000):
    """
    Estimate memory usage for NIST17 full dataset training

    Args:
        ram_gb: Available system RAM (GB)
        dataset_size: Number of compounds to train on
    """
    print("=" * 80)
    print(f"NEIMS v2.0 Memory Estimation Tool")
    print("=" * 80)
    print(f"System RAM: {ram_gb} GB")
    print(f"Dataset Size: {dataset_size:,} compounds\n")

    # Traditional approach
    print("🔴 Traditional In-Memory Dataset")
    print("-" * 80)
    trad = estimate_memory_usage(dataset_size, mode='traditional')

    print(f"  Dataset in RAM:     {trad['dataset_mb']:>10,.1f} MB  ({trad['dataset_mb']/1024:.1f} GB)")
    print(f"  Model (Teacher):    {trad['model_mb']:>10,.1f} MB  ({trad['model_mb']/1024:.1f} GB)")
    print(f"  Training Overhead:  {trad['training_mb']:>10,.1f} MB  ({trad['training_mb']/1024:.1f} GB)")
    print(f"  {'─'*40}")
    print(f"  Total RAM Required: {trad['total_mb']:>10,.1f} MB  ({trad['total_mb']/1024:.1f} GB)")

    if trad['total_mb'] / 1024 <= ram_gb:
        print(f"  Status: ✅ Fits in {ram_gb}GB RAM")
    else:
        print(f"  Status: ❌ EXCEEDS {ram_gb}GB RAM (overflow risk)")

    # Lazy loading approach
    print("\n🟢 Lazy Loading Dataset (HDF5 + On-the-Fly)")
    print("-" * 80)
    lazy = estimate_memory_usage(dataset_size, mode='lazy')

    print(f"  Dataset in RAM:     {lazy['dataset_mb']:>10,.1f} MB  ({lazy['dataset_mb']/1024:.2f} GB)")
    print(f"  HDF5 Cache (Disk):  {lazy['hdf5_disk_mb']:>10,.1f} MB  ({lazy['hdf5_disk_mb']/1024:.1f} GB)")
    print(f"  Model (Teacher):    {lazy['model_mb']:>10,.1f} MB  ({lazy['model_mb']/1024:.1f} GB)")
    print(f"  Training Overhead:  {lazy['training_mb']:>10,.1f} MB  ({lazy['training_mb']/1024:.1f} GB)")
    print(f"  {'─'*40}")
    print(f"  Total RAM Required: {lazy['total_mb']:>10,.1f} MB  ({lazy['total_mb']/1024:.1f} GB)")

    if lazy['total_mb'] / 1024 <= ram_gb:
        print(f"  Status: ✅ RECOMMENDED (fits in {ram_gb}GB RAM)")
    else:
        print(f"  Status: ⚠️  May be tight on {ram_gb}GB RAM")

    # Comparison
    print("\n📊 Memory Reduction Analysis")
    print("-" * 80)
    dataset_reduction = trad['dataset_mb'] / lazy['dataset_mb']
    total_reduction = trad['total_mb'] / lazy['total_mb']
    disk_reduction = trad['dataset_mb'] / lazy['hdf5_disk_mb']

    print(f"  Dataset Memory Reduction:   {dataset_reduction:>6.1f}x  (RAM saving)")
    print(f"  Total Memory Reduction:     {total_reduction:>6.1f}x  (overall)")
    print(f"  Disk Storage Efficiency:    {disk_reduction:>6.1f}x  (compression)")

    print(f"\n  RAM Freed: {(trad['total_mb'] - lazy['total_mb'])/1024:>6.1f} GB")
    print(f"  Disk Used: {lazy['hdf5_disk_mb']/1024:>6.1f} GB (compressed HDF5)")

    # Recommendations
    print("\n💡 Recommendations")
    print("-" * 80)

    if lazy['total_mb'] / 1024 <= ram_gb * 0.75:
        print(f"  ✅ Excellent: Using {(lazy['total_mb']/1024):.1f}GB / {ram_gb}GB ({(lazy['total_mb']/1024/ram_gb*100):.1f}%)")
        print(f"     Safe margin for OS and other processes")
        print(f"     Recommended settings:")
        print(f"       - use_lazy_loading: true")
        print(f"       - precompute_graphs: false")
        print(f"       - batch_size: 32")
        print(f"       - gradient_accumulation: 2")

    elif lazy['total_mb'] / 1024 <= ram_gb * 0.9:
        print(f"  ⚠️  Good but tight: Using {(lazy['total_mb']/1024):.1f}GB / {ram_gb}GB ({(lazy['total_mb']/1024/ram_gb*100):.1f}%)")
        print(f"     Recommended settings:")
        print(f"       - use_lazy_loading: true")
        print(f"       - precompute_graphs: false")
        print(f"       - batch_size: 16-24 (reduce if OOM)")
        print(f"       - gradient_accumulation: 4")
        print(f"       - empty_cache_frequency: 50")

    else:
        print(f"  ❌ Insufficient RAM: {(lazy['total_mb']/1024):.1f}GB / {ram_gb}GB")
        print(f"     Options:")
        print(f"       1. Upgrade RAM to {int((lazy['total_mb']/1024) * 1.25)}GB+")
        print(f"       2. Reduce max_samples to {int(dataset_size * 0.7):,}")
        print(f"       3. Use smaller batch_size (8-16)")

    print("\n" + "=" * 80)


def benchmark_dataset_loading(dataset_size: int = 100000):
    """
    Benchmark actual dataset loading (requires data)

    Args:
        dataset_size: Number of compounds to benchmark
    """
    import time
    import psutil
    from src.data.lazy_dataset import LazyMassSpecDataset

    print("=" * 80)
    print("Dataset Loading Benchmark")
    print("=" * 80)

    # Check if data exists
    msp_file = "data/NIST17.MSP"
    if not os.path.exists(msp_file):
        print(f"❌ Data not found: {msp_file}")
        print("   Please download NIST17 dataset first")
        return

    # Memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    print(f"\nCreating LazyMassSpecDataset ({dataset_size:,} samples)...")
    start_time = time.time()

    try:
        dataset = LazyMassSpecDataset(
            msp_file=msp_file,
            mol_files_dir="data/mol_files",
            max_mz=500,
            cache_dir="data/processed/lazy_cache",
            precompute_graphs=False,
            mode='teacher',
            split='train',
            max_samples=dataset_size
        )

        load_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        print(f"\n✅ Dataset loaded successfully")
        print(f"   Samples: {len(dataset):,}")
        print(f"   Load time: {load_time:.2f} seconds")
        print(f"   Memory used: {mem_used:.1f} MB ({mem_used/1024:.2f} GB)")
        print(f"   Per-sample memory: {mem_used/len(dataset):.2f} KB")

        # Test sample loading
        print(f"\nTesting sample loading (10 samples)...")
        sample_times = []

        for i in range(10):
            start = time.time()
            sample = dataset[i]
            sample_times.append(time.time() - start)

        avg_time = sum(sample_times) / len(sample_times)
        print(f"   Average load time: {avg_time*1000:.2f} ms per sample")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


def compare_modes(max_samples: int = 300000):
    """
    Compare lazy vs traditional modes across different dataset sizes

    Args:
        max_samples: Maximum number of samples to compare
    """
    print("=" * 80)
    print("Memory Mode Comparison: Lazy vs Traditional")
    print("=" * 80)

    sizes = [10000, 50000, 100000, 200000, 300000]
    sizes = [s for s in sizes if s <= max_samples]

    print(f"\n{'Samples':>10} | {'Traditional (GB)':>18} | {'Lazy (GB)':>12} | {'Reduction':>10} | {'32GB OK?':>10}")
    print("-" * 80)

    for size in sizes:
        trad = estimate_memory_usage(size, mode='traditional')
        lazy = estimate_memory_usage(size, mode='lazy')

        trad_gb = trad['total_mb'] / 1024
        lazy_gb = lazy['total_mb'] / 1024
        reduction = trad_gb / lazy_gb

        trad_ok = "✅" if trad_gb <= 32 else "❌"
        lazy_ok = "✅" if lazy_gb <= 32 else "❌"

        print(f"{size:>10,} | {trad_gb:>8.1f} {trad_ok:>8} | {lazy_gb:>8.1f} {lazy_ok:>2} | {reduction:>8.1f}x | Lazy: {lazy_ok}")

    print("\nConclusion:")
    print("  - Lazy loading enables full NIST17 (300k) on 32GB RAM")
    print("  - Traditional approach only works for ~50k compounds")
    print("  - 2-3x total memory reduction with lazy loading")


def main():
    parser = argparse.ArgumentParser(
        description="Memory Benchmark Tool for NIST17 Full Dataset"
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['estimate', 'benchmark', 'compare'],
        default='estimate',
        help='Benchmark mode'
    )

    parser.add_argument(
        '--ram_gb',
        type=int,
        default=32,
        help='Available system RAM (GB)'
    )

    parser.add_argument(
        '--dataset_size',
        type=int,
        default=300000,
        help='Number of compounds'
    )

    parser.add_argument(
        '--max_samples',
        type=int,
        default=300000,
        help='Maximum samples for comparison'
    )

    args = parser.parse_args()

    if args.mode == 'estimate':
        estimate_configuration(ram_gb=args.ram_gb, dataset_size=args.dataset_size)

    elif args.mode == 'benchmark':
        benchmark_dataset_loading(dataset_size=args.dataset_size)

    elif args.mode == 'compare':
        compare_modes(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
