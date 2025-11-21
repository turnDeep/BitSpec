#!/usr/bin/env python3
"""
メモリ使用量ベンチマーク

従来のデータセットと遅延読み込みデータセットのメモリ使用量を比較します。
"""

import sys
import os
import psutil
import torch
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.lazy_dataset import (
    LazyMassSpecDataset,
    PrecomputedGraphDataset,
    estimate_dataset_memory
)


def get_memory_usage_mb():
    """現在のメモリ使用量を取得（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_lazy_dataset(msp_file, mol_files_dir, num_samples=1000):
    """遅延読み込みデータセットのベンチマーク"""
    print("\n" + "=" * 60)
    print(f"Lazy Dataset Benchmark ({num_samples} samples)")
    print("=" * 60)

    mem_before = get_memory_usage_mb()
    print(f"Memory before: {mem_before:.1f} MB")

    # データセット作成
    dataset = LazyMassSpecDataset(
        msp_file=msp_file,
        mol_files_dir=mol_files_dir,
        max_mz=500,
        cache_dir=f"data/processed/lazy_cache_benchmark",
        max_samples=num_samples
    )

    mem_after_init = get_memory_usage_mb()
    print(f"Memory after init: {mem_after_init:.1f} MB")
    print(f"Memory used by dataset: {mem_after_init - mem_before:.1f} MB")

    # データアクセステスト
    print("\nAccessing 100 random samples...")
    import random
    indices = random.sample(range(len(dataset)), min(100, len(dataset)))

    mem_before_access = get_memory_usage_mb()
    for idx in indices:
        graph, spectrum, metadata = dataset[idx]

    mem_after_access = get_memory_usage_mb()
    print(f"Memory after accessing samples: {mem_after_access:.1f} MB")
    print(f"Memory increase: {mem_after_access - mem_before_access:.1f} MB")

    return {
        'dataset_memory': mem_after_init - mem_before,
        'access_overhead': mem_after_access - mem_before_access,
        'total_memory': mem_after_access
    }


def benchmark_precomputed_dataset(msp_file, mol_files_dir, num_samples=1000):
    """事前計算データセットのベンチマーク"""
    print("\n" + "=" * 60)
    print(f"Precomputed Dataset Benchmark ({num_samples} samples)")
    print("=" * 60)

    mem_before = get_memory_usage_mb()
    print(f"Memory before: {mem_before:.1f} MB")

    # データセット作成（全データをメモリに保持）
    dataset = PrecomputedGraphDataset(
        msp_file=msp_file,
        mol_files_dir=mol_files_dir,
        cache_dir=f"data/processed/precomputed_cache_benchmark",
        max_mz=500,
        max_samples=num_samples
    )

    mem_after_init = get_memory_usage_mb()
    print(f"Memory after init: {mem_after_init:.1f} MB")
    print(f"Memory used by dataset: {mem_after_init - mem_before:.1f} MB")

    return {
        'dataset_memory': mem_after_init - mem_before,
        'total_memory': mem_after_init
    }


def print_recommendations(total_ram_gb: int):
    """システムRAMに基づく推奨設定を表示"""
    print("\n" + "=" * 60)
    print(f"Recommendations for {total_ram_gb}GB RAM System")
    print("=" * 60)

    available_ram_gb = total_ram_gb * 0.7  # 70% を利用可能と仮定

    scenarios = [
        {
            'name': 'Full NIST17 (300,000 samples)',
            'samples': 300000,
            'estimates': estimate_dataset_memory(300000)
        },
        {
            'name': 'Large subset (100,000 samples)',
            'samples': 100000,
            'estimates': estimate_dataset_memory(100000)
        },
        {
            'name': 'Medium subset (50,000 samples)',
            'samples': 50000,
            'estimates': estimate_dataset_memory(50000)
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        est = scenario['estimates']

        lazy_total = est['lazy_memory_mb'] / 1024 + 10  # +10GB for model & training
        precomputed_total = est['traditional_memory_mb'] / 1024 + 10

        print(f"  Lazy Loading:")
        print(f"    Dataset:  {est['lazy_memory_mb']:.1f} MB")
        print(f"    Total:    ~{lazy_total:.1f} GB (dataset + model + training)")
        if lazy_total < available_ram_gb:
            print(f"    Status:   ✅ RECOMMENDED (fits in {total_ram_gb}GB RAM)")
        else:
            print(f"    Status:   ⚠️  May be tight (needs {lazy_total:.1f}GB)")

        print(f"  Precomputed:")
        print(f"    Dataset:  {est['traditional_memory_mb']:.1f} MB")
        print(f"    Total:    ~{precomputed_total:.1f} GB")
        if precomputed_total < available_ram_gb:
            print(f"    Status:   ✅ OK (faster but uses more memory)")
        else:
            print(f"    Status:   ❌ NOT RECOMMENDED (needs {precomputed_total:.1f}GB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory benchmark for datasets")
    parser.add_argument("--msp_file", type=str,
                        default="data/NIST17.msp",
                        help="Path to MSP file")
    parser.add_argument("--mol_dir", type=str,
                        default="data/mol_files",
                        help="Path to MOL files directory")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples to benchmark")
    parser.add_argument("--ram_gb", type=int, default=32,
                        help="Total system RAM in GB")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["lazy", "precomputed", "all", "estimate"],
                        help="Benchmark mode")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("NEIMS v2.0 Memory Benchmark")
    print("=" * 60)
    print(f"System RAM: {args.ram_gb}GB")
    print(f"Python process PID: {os.getpid()}")

    if args.mode == "estimate":
        # 推定のみ表示
        print_recommendations(args.ram_gb)

    elif args.mode == "lazy" or args.mode == "all":
        # 遅延読み込みベンチマーク
        if Path(args.msp_file).exists():
            lazy_results = benchmark_lazy_dataset(
                args.msp_file,
                args.mol_dir,
                args.samples
            )
        else:
            print(f"\n⚠️  MSP file not found: {args.msp_file}")
            print("Showing estimates only:")
            print_recommendations(args.ram_gb)
            sys.exit(0)

    if args.mode == "precomputed" or args.mode == "all":
        # 事前計算ベンチマーク
        if Path(args.msp_file).exists():
            precomputed_results = benchmark_precomputed_dataset(
                args.msp_file,
                args.mol_dir,
                args.samples
            )
        else:
            print(f"\n⚠️  MSP file not found: {args.msp_file}")

    # 推奨設定を表示
    print_recommendations(args.ram_gb)

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
