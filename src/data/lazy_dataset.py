#!/usr/bin/env python3
# src/data/lazy_dataset.py
"""
メモリ効率的な遅延読み込みデータセット

大規模データセット（NIST17: 30万化合物）を扱うための
メモリ効率的な実装。データは必要に応じてオンザフライで読み込まれます。
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Optional, Dict, Tuple
import json
import h5py
import pickle
from tqdm import tqdm

from .mol_parser import MOLParser, NISTMSPParser
from .features import MolecularFeaturizer, SubstructureFeaturizer


class LazyMassSpecDataset(Dataset):
    """
    遅延読み込み型マススペクトルデータセット

    大規模データセット向けのメモリ効率的な実装:
    - メタデータのみをメモリに保持
    - グラフデータは必要時にオンザフライで読み込み
    - HDF5形式でスペクトルを保存（高速アクセス）

    メモリ使用量:
    - メタデータ: ~100MB (300,000化合物)
    - キャッシュオーバーヘッド: ~50MB
    - 合計: ~150MB（従来の10GBから大幅削減）
    """

    def __init__(
        self,
        msp_file: str,
        mol_files_dir: str,
        max_mz: int = 500,
        mz_bin_size: float = 1.0,
        cache_dir: Optional[str] = None,
        transform=None,
        use_functional_groups: bool = True,
        precompute_graphs: bool = False,
        max_samples: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Args:
            msp_file: NIST MSPファイルのパス
            mol_files_dir: MOLファイルのディレクトリ
            max_mz: 最大m/z値
            mz_bin_size: ビンサイズ
            cache_dir: キャッシュディレクトリ（HDF5とメタデータ）
            transform: データ変換関数
            use_functional_groups: 官能基フィンガープリントを使用するか
            precompute_graphs: グラフを事前計算してキャッシュするか（推奨: False）
            max_samples: サンプリングする最大サンプル数
            random_seed: ランダムシード
        """
        self.msp_file = msp_file
        self.mol_files_dir = Path(mol_files_dir)
        self.max_mz = max_mz
        self.mz_bin_size = mz_bin_size
        self.transform = transform
        self.use_functional_groups = use_functional_groups
        self.precompute_graphs = precompute_graphs
        self.max_samples = max_samples
        self.random_seed = random_seed

        # キャッシュディレクトリ
        if cache_dir is None:
            cache_dir = "data/processed/lazy_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # パーサーと特徴量抽出器（オンザフライ処理用）
        self.msp_parser = NISTMSPParser()
        self.mol_parser = MOLParser()
        self.featurizer = MolecularFeaturizer()
        if self.use_functional_groups:
            self.substructure_featurizer = SubstructureFeaturizer()

        # キャッシュファイルパス
        cache_suffix = f"_n{max_samples}" if max_samples else ""
        self.metadata_file = self.cache_dir / f"metadata{cache_suffix}.json"
        self.spectra_file = self.cache_dir / f"spectra{cache_suffix}.h5"
        self.graphs_file = self.cache_dir / f"graphs{cache_suffix}.pkl" if precompute_graphs else None

        # メタデータの読み込みまたは作成
        if self._cache_exists():
            print(f"Loading metadata from cache: {self.metadata_file}")
            self._load_metadata()
        else:
            print("Building metadata and spectrum cache...")
            self._build_cache()

        print(f"Dataset ready: {len(self)} samples (Memory-efficient mode)")
        print(f"Estimated memory usage: ~{self._estimate_memory_mb():.1f} MB")

    def _cache_exists(self) -> bool:
        """キャッシュが存在し有効かチェック"""
        return (self.metadata_file.exists() and
                self.spectra_file.exists())

    def _build_cache(self):
        """メタデータとスペクトルのキャッシュを構築"""
        # MSPファイルを解析
        print("Parsing MSP file...")
        compounds = self.msp_parser.parse_file(self.msp_file)

        # 利用可能なMOLファイルをフィルタリング
        print("Filtering compounds with available MOL files...")
        available_compounds = []
        for compound in tqdm(compounds, desc="Checking MOL files"):
            compound_id = compound['ID']
            mol_file = self.mol_files_dir / f"ID{compound_id}.MOL"
            if mol_file.exists():
                available_compounds.append(compound)

        print(f"Found {len(available_compounds)} compounds with MOL files")

        # ランダムサンプリング
        if self.max_samples and self.max_samples < len(available_compounds):
            import random
            random.seed(self.random_seed)
            available_compounds = random.sample(available_compounds, self.max_samples)
            print(f"Sampled {self.max_samples} compounds")

        # メタデータとスペクトルを保存
        self.compound_ids = []
        self.metadata_list = []

        print("Building HDF5 spectrum cache...")
        with h5py.File(self.spectra_file, 'w') as h5f:
            num_bins = int(self.max_mz / self.mz_bin_size) + 1

            # HDF5データセットを作成（圧縮あり）
            spectra_dataset = h5f.create_dataset(
                'spectra',
                shape=(len(available_compounds), num_bins),
                dtype='float32',
                compression='gzip',
                compression_opts=4
            )

            for idx, compound in enumerate(tqdm(available_compounds, desc="Processing")):
                compound_id = compound['ID']

                # スペクトルを正規化
                spectrum = self.msp_parser.normalize_spectrum(
                    compound['Spectrum'],
                    max_mz=self.max_mz,
                    mz_bin_size=self.mz_bin_size
                )

                # HDF5に保存
                spectra_dataset[idx] = spectrum

                # メタデータを保存
                self.compound_ids.append(compound_id)
                self.metadata_list.append({
                    'id': compound_id,
                    'name': compound.get('Name', ''),
                    'formula': compound.get('Formula', ''),
                    'mol_weight': compound.get('MW', 0),
                    'cas_no': compound.get('CASNO', ''),
                    'mol_file': str(self.mol_files_dir / f"ID{compound_id}.MOL")
                })

        # メタデータをJSONで保存
        print("Saving metadata...")
        metadata_dict = {
            'compound_ids': self.compound_ids,
            'metadata': self.metadata_list,
            'config': {
                'max_mz': self.max_mz,
                'mz_bin_size': self.mz_bin_size,
                'use_functional_groups': self.use_functional_groups,
                'num_samples': len(self.compound_ids)
            }
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        print(f"Cache built: {len(self.compound_ids)} samples")
        print(f"Spectrum cache: {self.spectra_file} ({self.spectra_file.stat().st_size / 1e6:.1f} MB)")

    def _load_metadata(self):
        """メタデータを読み込み"""
        with open(self.metadata_file, 'r') as f:
            data = json.load(f)

        self.compound_ids = data['compound_ids']
        self.metadata_list = data['metadata']

        print(f"Loaded metadata: {len(self.compound_ids)} samples")

    def _estimate_memory_mb(self) -> float:
        """推定メモリ使用量 (MB)"""
        # メタデータのみをメモリに保持
        metadata_size = len(self.compound_ids) * 0.5  # ~0.5KB per compound
        overhead = 50  # MB
        return metadata_size / 1024 + overhead

    def __len__(self) -> int:
        return len(self.compound_ids)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor, Dict]:
        """
        オンザフライでグラフを生成

        Returns:
            graph_data: グラフデータ
            spectrum: スペクトル [501]
            metadata: メタデータ
        """
        compound_id = self.compound_ids[idx]
        metadata = self.metadata_list[idx]

        # スペクトルをHDF5から読み込み（高速）
        with h5py.File(self.spectra_file, 'r') as h5f:
            spectrum = torch.from_numpy(h5f['spectra'][idx]).float()

        # グラフをオンザフライで生成
        mol_file = metadata['mol_file']
        mol = self.mol_parser.parse_file(mol_file)
        graph_data = self.featurizer.mol_to_graph(mol, y=spectrum)

        # 官能基フィンガープリント
        if self.use_functional_groups:
            fg_fingerprint = self.substructure_featurizer.get_substructure_fingerprint(mol)
            graph_data.functional_groups = torch.tensor(fg_fingerprint, dtype=torch.float32)

        if self.transform:
            graph_data = self.transform(graph_data)

        return graph_data, spectrum, metadata


class PrecomputedGraphDataset(Dataset):
    """
    事前計算グラフデータセット

    グラフを事前計算して保存することで、
    トレーニング時のCPUオーバーヘッドを削減します。

    注意: メモリ使用量は増加しますが、
    小規模データセット（~50,000サンプル）では有効です。
    """

    def __init__(
        self,
        msp_file: str,
        mol_files_dir: str,
        cache_dir: str,
        max_mz: int = 500,
        mz_bin_size: float = 1.0,
        max_samples: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Args:
            msp_file: MSPファイル
            mol_files_dir: MOLファイルディレクトリ
            cache_dir: キャッシュディレクトリ
            max_mz: 最大m/z
            mz_bin_size: ビンサイズ
            max_samples: 最大サンプル数
            random_seed: ランダムシード
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_suffix = f"_n{max_samples}" if max_samples else ""
        self.cache_file = self.cache_dir / f"precomputed_graphs{cache_suffix}.pkl"

        if self.cache_file.exists():
            print(f"Loading precomputed graphs from: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.data_list = pickle.load(f)
            print(f"Loaded {len(self.data_list)} precomputed samples")
        else:
            print("Building precomputed graph cache...")
            # LazyDatasetを使って一度全データを処理
            lazy_dataset = LazyMassSpecDataset(
                msp_file=msp_file,
                mol_files_dir=mol_files_dir,
                cache_dir=cache_dir,
                max_mz=max_mz,
                mz_bin_size=mz_bin_size,
                max_samples=max_samples,
                random_seed=random_seed
            )

            print("Precomputing all graphs (this may take a while)...")
            self.data_list = []
            for i in tqdm(range(len(lazy_dataset)), desc="Precomputing graphs"):
                graph, spectrum, metadata = lazy_dataset[i]
                self.data_list.append((graph, spectrum, metadata))

            print(f"Saving precomputed graphs to: {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data_list, f, protocol=4)

            file_size_mb = self.cache_file.stat().st_size / 1e6
            print(f"Precomputed cache size: {file_size_mb:.1f} MB")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor, Dict]:
        return self.data_list[idx]


def estimate_dataset_memory(num_samples: int) -> Dict[str, float]:
    """
    データセットのメモリ使用量を推定

    Args:
        num_samples: サンプル数

    Returns:
        memory_estimates: メモリ推定値（MB）
    """
    # 1サンプルあたりの推定サイズ
    graph_size_kb = 15  # グラフデータ
    spectrum_size_kb = 2  # スペクトル
    metadata_size_kb = 0.5  # メタデータ

    # 従来のデータセット（全データをメモリに保持）
    traditional_memory_mb = num_samples * (graph_size_kb + spectrum_size_kb + metadata_size_kb) / 1024

    # 遅延読み込みデータセット（メタデータのみ）
    lazy_memory_mb = num_samples * metadata_size_kb / 1024 + 50  # 50MB overhead

    # HDF5キャッシュサイズ
    hdf5_disk_mb = num_samples * spectrum_size_kb / 1024 * 0.3  # 圧縮率 ~30%

    return {
        'traditional_memory_mb': traditional_memory_mb,
        'lazy_memory_mb': lazy_memory_mb,
        'hdf5_disk_mb': hdf5_disk_mb,
        'memory_reduction_ratio': traditional_memory_mb / lazy_memory_mb
    }


if __name__ == "__main__":
    # メモリ使用量の推定
    print("\n=== Memory Usage Estimation ===\n")

    for num_samples in [50000, 100000, 300000]:
        estimates = estimate_dataset_memory(num_samples)
        print(f"Dataset size: {num_samples:,} samples")
        print(f"  Traditional dataset: {estimates['traditional_memory_mb']:.1f} MB")
        print(f"  Lazy dataset:        {estimates['lazy_memory_mb']:.1f} MB")
        print(f"  HDF5 cache (disk):   {estimates['hdf5_disk_mb']:.1f} MB")
        print(f"  Memory reduction:    {estimates['memory_reduction_ratio']:.1f}x")
        print()
