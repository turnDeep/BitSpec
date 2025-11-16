# src/data/dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import List, Tuple, Dict
import json
from pathlib import Path
from ..features.molecular_features import MolecularFeaturizer

class MassSpectrumDataset(Dataset):
    """マススペクトルデータセット"""
    
    def __init__(self, 
                 data_path: str,
                 max_mz: int = 1000,
                 transform=None):
        """
        Args:
            data_path: データファイルのパス（NIST MSP形式）
            max_mz: 最大m/z値
            transform: データ変換
        """
        self.data_path = Path(data_path)
        self.max_mz = max_mz
        self.transform = transform
        self.featurizer = MolecularFeaturizer()
        
        # データの読み込み
        self.samples = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """MSPファイルからデータを読み込み"""
        samples = []
        
        with open(self.data_path, 'r') as f:
            current_sample = {}
            peaks = []
            
            for line in f:
                line = line.strip()
                
                if line.startswith('NAME:'):
                    if current_sample and peaks:
                        current_sample['spectrum'] = self._peaks_to_spectrum(peaks)
                        samples.append(current_sample)
                        current_sample = {}
                        peaks = []
                    current_sample['name'] = line.split(':', 1)[1].strip()
                    
                elif line.startswith('SMILES:'):
                    current_sample['smiles'] = line.split(':', 1)[1].strip()
                    
                elif line.startswith('INCHIKEY:'):
                    current_sample['inchikey'] = line.split(':', 1)[1].strip()
                    
                elif line.startswith('NUM PEAKS:'):
                    num_peaks = int(line.split(':')[1].strip())
                    
                elif line and not line.startswith('#'):
                    # ピークデータの読み込み
                    try:
                        mz, intensity = line.split()[:2]
                        peaks.append((float(mz), float(intensity)))
                    except:
                        continue
            
            # 最後のサンプルを追加
            if current_sample and peaks:
                current_sample['spectrum'] = self._peaks_to_spectrum(peaks)
                samples.append(current_sample)
        
        print(f"Loaded {len(samples)} samples from {self.data_path}")
        return samples
    
    def _peaks_to_spectrum(self, peaks: List[Tuple[float, float]]) -> np.ndarray:
        """ピークリストをスペクトル配列に変換"""
        spectrum = np.zeros(self.max_mz, dtype=np.float32)
        
        for mz, intensity in peaks:
            if 0 <= mz < self.max_mz:
                bin_idx = int(mz)
                spectrum[bin_idx] = max(spectrum[bin_idx], intensity)
        
        # 正規化
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        
        return spectrum
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        """データサンプルを取得"""
        sample = self.samples[idx]
        
        # SMILESから分子グラフを作成
        mol = Chem.MolFromSmiles(sample['smiles'])
        if mol is None:
            raise ValueError(f"Invalid SMILES: {sample['smiles']}")
        
        # 3D座標の生成（オプション）
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
        
        # 分子グラフの特徴量化
        graph_data = self.featurizer.featurize(mol)
        
        # スペクトルデータ
        spectrum = torch.tensor(sample['spectrum'], dtype=torch.float32)
        
        # データの変換
        if self.transform:
            graph_data = self.transform(graph_data)
        
        return graph_data, spectrum

class MassSpectrumCollator:
    """バッチ処理のためのコレーター"""
    
    def __init__(self):
        pass
    
    def __call__(self, batch: List[Tuple[Data, torch.Tensor]]) -> Tuple[Batch, torch.Tensor]:
        """
        バッチデータの結合
        
        Args:
            batch: [(graph_data, spectrum), ...] のリスト
            
        Returns:
            batched_graphs: バッチ化されたグラフデータ
            batched_spectra: バッチ化されたスペクトル
        """
        graphs, spectra = zip(*batch)
        
        # グラフデータのバッチ化
        batched_graphs = Batch.from_data_list(graphs)
        
        # スペクトルのバッチ化
        batched_spectra = torch.stack(spectra)
        
        return batched_graphs, batched_spectra

def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """データローダーの作成"""
    
    # データセットの作成
    train_dataset = MassSpectrumDataset(train_path, **kwargs)
    val_dataset = MassSpectrumDataset(val_path, **kwargs)
    test_dataset = MassSpectrumDataset(test_path, **kwargs)
    
    # コレーターの作成
    collator = MassSpectrumCollator()
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
