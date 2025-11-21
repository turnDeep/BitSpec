#!/usr/bin/env python3
# src/data/pcqm4m_dataset.py
"""
NEIMS v2.0 PCQM4Mv2 Pretraining Dataset

Loads PCQM4Mv2 (3.8M molecules) for Teacher model pretraining.
Self-supervised task: Bond Masking for learning molecular representations.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)


def download_pcqm4mv2(data_dir: str):
    """
    Download PCQM4Mv2 dataset from OGB

    The dataset will be automatically downloaded using ogb library.
    """
    try:
        from ogb.lsc import PCQM4Mv2Dataset as OGB_PCQM4Mv2

        logger.info(f"Downloading PCQM4Mv2 to {data_dir}...")
        dataset = OGB_PCQM4Mv2(root=data_dir, only_smiles=False)
        logger.info(f"PCQM4Mv2 downloaded: {len(dataset)} molecules")
        return dataset

    except ImportError:
        logger.error("OGB library not installed. Install with: pip install ogb")
        logger.info("Falling back to manual download...")
        return None


def mol_to_graph_with_mask(
    mol: Chem.Mol,
    mask_ratio: float = 0.15
) -> Tuple[Data, torch.Tensor]:
    """
    Convert molecule to graph with bond masking

    Args:
        mol: RDKit molecule
        mask_ratio: Ratio of bonds to mask

    Returns:
        data: PyG Data with masked bonds
        mask_targets: Target bond types for masked bonds
    """
    # Atom features (48-dimensional)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
            atom.IsInRing(),
        ]
        # One-hot encoding
        features += [int(atom.GetAtomicNum() == i) for i in range(1, 37)]
        features += [int(atom.GetTotalDegree() == i) for i in range(7)]
        atom_features.append(features[:48])

    x = torch.tensor(atom_features, dtype=torch.float)

    # Bond features with masking
    edge_index = []
    edge_attr = []
    mask_indices = []
    mask_targets = []

    bonds = list(mol.GetBonds())
    num_bonds = len(bonds)
    num_mask = int(num_bonds * mask_ratio)

    # Randomly select bonds to mask
    if num_mask > 0:
        mask_bond_indices = np.random.choice(num_bonds, num_mask, replace=False)
    else:
        mask_bond_indices = []

    for bond_idx, bond in enumerate(bonds):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type = bond.GetBondTypeAsDouble()
        is_conjugated = bond.GetIsConjugated()
        is_aromatic = bond.GetIsAromatic()
        is_in_ring = bond.IsInRing()

        # Check if this bond should be masked
        is_masked = bond_idx in mask_bond_indices

        if is_masked:
            # Mask: Set bond features to zeros
            edge_features = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # [mask_flag=1]
            mask_targets.append([bond_type, is_conjugated, is_aromatic, is_in_ring])
            mask_indices.append(len(edge_index))  # Store edge index
        else:
            # Normal bond
            edge_features = [bond_type, is_conjugated, is_aromatic, is_in_ring, 0.0, 0.0]

        # Bidirectional edges
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)  # Same features for reverse edge

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Mask targets
    if mask_targets:
        mask_targets = torch.tensor(mask_targets, dtype=torch.float)
        mask_indices = torch.tensor(mask_indices, dtype=torch.long)
    else:
        mask_targets = torch.zeros((0, 4), dtype=torch.float)
        mask_indices = torch.zeros(0, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        mask_indices=mask_indices,
        mask_targets=mask_targets
    )

    return data, mask_targets


def mol_to_ecfp(mol: Chem.Mol, radius: int = 2, n_bits: int = 4096) -> np.ndarray:
    """Generate ECFP4 fingerprint"""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


class PCQM4Mv2Dataset(Dataset):
    """
    PCQM4Mv2 Dataset for Teacher Pretraining

    Self-supervised task: Bond Masking
    - Mask 15% of bonds
    - Predict masked bond types and features
    """

    def __init__(
        self,
        data_config: Dict,
        split: str = 'train',
        mask_ratio: float = 0.15,
        cache_dir: Optional[str] = None,
        download: bool = True
    ):
        """
        Args:
            data_config: Data configuration
            split: 'train' or 'val'
            mask_ratio: Ratio of bonds to mask
            cache_dir: Cache directory
            download: Auto-download if not exists
        """
        self.data_config = data_config
        self.split = split
        self.mask_ratio = mask_ratio

        # Data directory
        data_dir = Path(data_config.get('pcqm4mv2_path', 'data/pcqm4mv2'))
        data_dir.mkdir(parents=True, exist_ok=True)

        # Cache setup
        if cache_dir is None:
            cache_dir = data_config.get('output_dir', 'data/processed')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load OGB dataset
        if download:
            ogb_dataset = download_pcqm4mv2(str(data_dir))
        else:
            ogb_dataset = None

        # Cache file
        cache_file = self.cache_dir / f'pcqm4mv2_{split}_processed.pkl'

        if cache_file.exists():
            logger.info(f"Loading cached PCQM4Mv2 from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.smiles_list = cache_data['smiles']
                self.split_indices = cache_data['indices']
        else:
            logger.info(f"Processing PCQM4Mv2 for {split} split...")
            self.smiles_list, self.split_indices = self._process_ogb_data(
                ogb_dataset, data_dir
            )

            # Save cache
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'smiles': self.smiles_list,
                    'indices': self.split_indices
                }, f)
            logger.info(f"Cached to {cache_file}")

        logger.info(f"Loaded PCQM4Mv2 {split}: {len(self.split_indices)} molecules")

    def _process_ogb_data(
        self,
        ogb_dataset,
        data_dir: Path
    ) -> Tuple[List[str], List[int]]:
        """Process OGB dataset"""
        smiles_list = []

        if ogb_dataset is not None:
            # Use OGB dataset
            logger.info("Processing OGB PCQM4Mv2...")
            for i in range(len(ogb_dataset)):
                graph, _ = ogb_dataset[i]
                # Convert PyG graph back to SMILES (if available)
                # For now, load from OGB's SMILES file
                pass

            # Try loading SMILES directly
            smiles_file = data_dir / 'pcqm4m-v2' / 'raw' / 'data.csv.gz'
            if smiles_file.exists():
                import pandas as pd
                df = pd.read_csv(smiles_file, compression='gzip')
                smiles_list = df['smiles'].tolist()
                logger.info(f"Loaded {len(smiles_list)} SMILES from OGB")

        else:
            # Fallback: Load from custom SMILES file
            smiles_file = data_dir / 'pcqm4mv2_smiles.txt'
            if smiles_file.exists():
                logger.info(f"Loading SMILES from {smiles_file}")
                with open(smiles_file, 'r') as f:
                    smiles_list = [line.strip() for line in f if line.strip()]
            else:
                logger.warning("No PCQM4Mv2 data found. Creating dummy dataset.")
                # Create small dummy dataset for testing
                smiles_list = [
                    'CCO',  # Ethanol
                    'CC(C)O',  # Isopropanol
                    'c1ccccc1',  # Benzene
                ] * 100

        # Split data (train: 90%, val: 10%)
        np.random.seed(42)
        indices = np.random.permutation(len(smiles_list))

        split_idx = int(len(indices) * 0.9)

        if self.split == 'train':
            split_indices = indices[:split_idx].tolist()
        else:  # val
            split_indices = indices[split_idx:].tolist()

        return smiles_list, split_indices

    def __len__(self) -> int:
        return len(self.split_indices)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
                'graph': PyG Data with masked bonds
                'ecfp': ECFP4 fingerprint
                'spectrum': Dummy spectrum (zeros for pretraining)
                'mask_indices': Indices of masked edges
                'bond_targets': Target bond features (for bond masking task)
            }
        """
        real_idx = self.split_indices[idx]
        smiles = self.smiles_list[real_idx]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Fallback to simple molecule
            mol = Chem.MolFromSmiles('CCO')

        # Generate graph with masking
        graph, mask_targets = mol_to_graph_with_mask(mol, self.mask_ratio)

        # Generate ECFP
        ecfp = mol_to_ecfp(mol)

        # Dummy spectrum for pretraining (not used in loss, but needed for consistency)
        dummy_spectrum = torch.zeros(501, dtype=torch.float32)

        return {
            'graph': graph,
            'ecfp': torch.tensor(ecfp, dtype=torch.float32),
            'spectrum': dummy_spectrum,
            'mask_indices': graph.mask_indices,
            'bond_targets': mask_targets,  # Renamed from 'mask_targets' for trainer compatibility
            'smiles': smiles
        }


def collate_fn_pretrain(batch: List[Dict]) -> Dict:
    """Custom collate for pretraining"""
    from torch_geometric.data import Batch

    graphs = [sample['graph'] for sample in batch]
    ecfps = torch.stack([sample['ecfp'] for sample in batch])
    spectra = torch.stack([sample['spectrum'] for sample in batch])

    # Batch graphs
    graph_batch = Batch.from_data_list(graphs)

    # Collect all mask indices and targets
    mask_indices_list = []
    mask_targets_list = []

    edge_offset = 0
    for graph in graphs:
        if graph.mask_indices.numel() > 0:
            # Adjust indices for batched graph
            adjusted_indices = graph.mask_indices + edge_offset
            mask_indices_list.append(adjusted_indices)
            mask_targets_list.append(graph.mask_targets)

        edge_offset += graph.edge_index.size(1)

    if mask_indices_list:
        mask_indices = torch.cat(mask_indices_list, dim=0)
        bond_targets = torch.cat(mask_targets_list, dim=0)
    else:
        mask_indices = torch.zeros(0, dtype=torch.long)
        bond_targets = torch.zeros((0, 4), dtype=torch.float)

    return {
        'graph': graph_batch,
        'ecfp': ecfps,
        'spectrum': spectra,  # Dummy spectra for consistency
        'mask_indices': mask_indices,
        'bond_targets': bond_targets  # Renamed from 'mask_targets' for trainer compatibility
    }
