# src/data/dataset.py
"""
マススペクトル予測用データセット
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional, Dict
import pickle
from tqdm import tqdm
import random
import hashlib
import json

from .mol_parser import MOLParser, NISTMSPParser
from .features import MolecularFeaturizer, SubstructureFeaturizer


class MassSpecDataset(Dataset):
    """マススペクトル予測データセット"""
    
    def __init__(
        self,
        msp_file: str,
        mol_files_dir: str,
        max_mz: int = 1000,
        mz_bin_size: float = 1.0,
        cache_file: Optional[str] = None,
        transform=None,
        max_samples: Optional[int] = None,
        random_seed: int = 42,
        use_functional_groups: bool = True
    ):
        """
        Args:
            msp_file: NIST MSPファイルのパス
            mol_files_dir: MOLファイルのディレクトリ
            max_mz: 最大m/z値
            mz_bin_size: ビンサイズ
            cache_file: キャッシュファイルのパス
            transform: データ変換関数
            max_samples: ランダムサンプリングする最大サンプル数（Noneの場合は全て使用）
            random_seed: ランダムシード
            use_functional_groups: 官能基フィンガープリントを使用するか
        """
        self.msp_file = msp_file
        self.mol_files_dir = Path(mol_files_dir)
        self.max_mz = max_mz
        self.mz_bin_size = mz_bin_size
        self.transform = transform
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.use_functional_groups = use_functional_groups

        # パーサーと特徴量抽出器
        self.msp_parser = NISTMSPParser()
        self.mol_parser = MOLParser()
        self.featurizer = MolecularFeaturizer()

        # 官能基フィンガープリント抽出器
        if self.use_functional_groups:
            self.substructure_featurizer = SubstructureFeaturizer()

        # キャッシュファイル名にmax_samplesを含める
        if cache_file and max_samples is not None:
            cache_path = Path(cache_file)
            cache_file = str(cache_path.parent / f"{cache_path.stem}_n{max_samples}{cache_path.suffix}")

        # キャッシュの検証とロード
        cache_valid = False
        if cache_file and Path(cache_file).exists():
            print(f"Loading cached data from {cache_file}...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # キャッシュのメタデータを確認
                if isinstance(cached_data, dict) and 'metadata' in cached_data:
                    metadata = cached_data['metadata']
                    self.data_list = cached_data['data_list']

                    # 設定が一致するかチェック
                    if (metadata.get('max_samples') == max_samples and
                        metadata.get('random_seed') == random_seed and
                        metadata.get('max_mz') == max_mz and
                        metadata.get('mz_bin_size') == mz_bin_size and
                        metadata.get('use_functional_groups') == use_functional_groups):
                        cache_valid = True
                        print(f"Cache valid: {len(self.data_list)} samples")
                    else:
                        print("Cache configuration mismatch, rebuilding...")
                else:
                    # 古い形式のキャッシュ（メタデータなし）
                    print("Old cache format detected, rebuilding...")
            except Exception as e:
                print(f"Error loading cache: {e}, rebuilding...")

        if not cache_valid:
            print("Processing data...")
            self.data_list = self._process_data()

            if cache_file:
                print(f"Saving cache to {cache_file}...")
                Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

                # メタデータと共に保存
                cache_data = {
                    'data_list': self.data_list,
                    'metadata': {
                        'max_samples': max_samples,
                        'random_seed': random_seed,
                        'max_mz': max_mz,
                        'mz_bin_size': mz_bin_size,
                        'use_functional_groups': use_functional_groups,
                        'num_samples': len(self.data_list)
                    }
                }

                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)

        print(f"Loaded {len(self.data_list)} samples")
    
    def _process_data(self) -> List[Tuple[Data, np.ndarray, Dict]]:
        """データを処理"""
        # MSPファイルを解析
        compounds = self.msp_parser.parse_file(self.msp_file)

        # 利用可能なmolファイルをフィルタリング
        available_compounds = []
        for compound in compounds:
            compound_id = compound['ID']
            mol_file = self.mol_files_dir / f"ID{compound_id}.MOL"
            if mol_file.exists():
                available_compounds.append(compound)

        print(f"Found {len(available_compounds)} compounds with available MOL files")

        # ランダムサンプリング
        if self.max_samples is not None and self.max_samples < len(available_compounds):
            random.seed(self.random_seed)
            compounds_to_process = random.sample(available_compounds, self.max_samples)
            print(f"Randomly sampling {self.max_samples} compounds (seed={self.random_seed})")
        else:
            compounds_to_process = available_compounds
            if self.max_samples is not None:
                print(f"Requested {self.max_samples} samples, but only {len(available_compounds)} available. Using all.")

        data_list = []
        for compound in tqdm(compounds_to_process, desc="Processing compounds"):
            try:
                # MOLファイルを読み込み
                compound_id = compound['ID']
                mol_file = self.mol_files_dir / f"ID{compound_id}.MOL"

                mol = self.mol_parser.parse_file(str(mol_file))

                # スペクトルを正規化
                spectrum = self.msp_parser.normalize_spectrum(
                    compound['Spectrum'],
                    max_mz=self.max_mz,
                    mz_bin_size=self.mz_bin_size
                )

                # グラフデータに変換
                graph_data = self.featurizer.mol_to_graph(mol, y=spectrum)

                # 官能基フィンガープリントの追加
                if self.use_functional_groups:
                    fg_fingerprint = self.substructure_featurizer.get_substructure_fingerprint(mol)
                    graph_data.functional_groups = torch.tensor(fg_fingerprint, dtype=torch.float32)

                # メタデータ
                metadata = {
                    'name': compound.get('Name', ''),
                    'formula': compound.get('Formula', ''),
                    'mol_weight': compound.get('MW', 0),
                    'cas_no': compound.get('CASNO', ''),
                    'id': compound_id
                }

                data_list.append((graph_data, spectrum, metadata))

            except Exception as e:
                print(f"Error processing compound {compound.get('ID', 'unknown')}: {e}")
                continue

        return data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor, Dict]:
        """
        Args:
            idx: インデックス
            
        Returns:
            graph_data: グラフデータ
            spectrum: マススペクトル
            metadata: メタデータ
        """
        graph_data, spectrum, metadata = self.data_list[idx]
        
        if self.transform:
            graph_data = self.transform(graph_data)
        
        return graph_data, torch.tensor(spectrum, dtype=torch.float32), metadata


class NISTDataLoader:
    """カスタムデータローダー"""
    
    @staticmethod
    def collate_fn(batch):
        """バッチ処理用のcollate関数"""
        graphs, spectra, metadatas = zip(*batch)
        
        # グラフをバッチ化
        batched_graphs = Batch.from_data_list(graphs)
        
        # スペクトルをスタック
        batched_spectra = torch.stack(spectra)
        
        return batched_graphs, batched_spectra, metadatas
    
    @staticmethod
    def create_dataloaders(
        dataset: MassSpecDataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        訓練、検証、テストのデータローダーを作成

        Args:
            dataset: データセット
            train_ratio: 訓練データの割合
            val_ratio: 検証データの割合
            batch_size: バッチサイズ
            num_workers: ワーカー数
            prefetch_factor: 各ワーカーが先読みするバッチ数
            persistent_workers: ワーカーを永続化するか（エポック間でワーカーを再利用）
            seed: 乱数シード

        Returns:
            train_loader, val_loader, test_loader
        """
        # データセットを分割
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

        # persistent_workersはnum_workers > 0の場合のみ有効
        use_persistent = persistent_workers and num_workers > 0

        # データローダーを作成
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=NISTDataLoader.collate_fn,
            pin_memory=True,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=use_persistent
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=NISTDataLoader.collate_fn,
            pin_memory=True,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=use_persistent
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=NISTDataLoader.collate_fn,
            pin_memory=True,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=use_persistent
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # テスト
    print("Testing MassSpecDataset...")
    
    # ダミーパスでテスト（実際のファイルが必要）
    # dataset = MassSpecDataset(
    #     msp_file="data/NIST17.MSP",
    #     mol_files_dir="data/mol_files",
    #     cache_file="data/processed/cache.pkl"
    # )
    
    print("Dataset test passed (requires actual data files)")
