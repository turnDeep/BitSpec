#!/usr/bin/env python3
# src/data/nist_dataset.py
"""
NEIMS v2.0 NIST EI-MS Dataset Loader

Loads NIST EI-MS data from:
1. NIST17.msp file (if available)
2. MOL files + separate spectrum data

Supports Teacher (GNN+ECFP) and Student (ECFP only) modes.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)


def parse_msp_file(msp_path: str) -> List[Dict]:
    """
    Parse NIST MSP format file

    Returns:
        entries: List of {name, smiles, mw, formula, spectrum}
    """
    entries = []
    current_entry = {}

    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith('Name:'):
                if current_entry:
                    entries.append(current_entry)
                current_entry = {'name': line.split(':', 1)[1].strip()}

            elif line.startswith('MW:'):
                current_entry['mw'] = float(line.split(':')[1].strip())

            elif line.startswith('Formula:'):
                current_entry['formula'] = line.split(':', 1)[1].strip()

            elif line.startswith('SMILES:'):
                current_entry['smiles'] = line.split(':', 1)[1].strip()

            elif line.startswith('Num Peaks:'):
                current_entry['num_peaks'] = int(line.split(':')[1].strip())
                current_entry['peaks'] = []

            elif 'num_peaks' in current_entry and len(current_entry['peaks']) < current_entry['num_peaks']:
                # Parse peak data: "mz intensity; mz intensity; ..."
                if ';' in line:
                    for peak_str in line.split(';'):
                        if peak_str.strip():
                            parts = peak_str.strip().split()
                            if len(parts) >= 2:
                                mz = float(parts[0])
                                intensity = float(parts[1])
                                current_entry['peaks'].append((mz, intensity))

    if current_entry:
        entries.append(current_entry)

    return entries


def peaks_to_spectrum(peaks: List[Tuple[float, float]], max_mz: int = 500) -> np.ndarray:
    """
    Convert peak list to binned spectrum [0, 500] -> [501]

    Args:
        peaks: List of (mz, intensity) tuples
        max_mz: Maximum m/z value

    Returns:
        spectrum: Binned spectrum array [501]
    """
    spectrum = np.zeros(max_mz + 1, dtype=np.float32)

    for mz, intensity in peaks:
        if 0 <= mz <= max_mz:
            bin_idx = int(round(mz))
            spectrum[bin_idx] = max(spectrum[bin_idx], intensity)

    # Normalize to [0, 1]
    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()

    return spectrum


def mol_to_graph(mol: Chem.Mol) -> Data:
    """
    Convert RDKit molecule to PyTorch Geometric graph

    Returns:
        data: PyG Data object with node/edge features
    """
    # Atom features (48-dimensional)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),  # 1
            atom.GetTotalDegree(),  # 1
            atom.GetFormalCharge(),  # 1
            atom.GetTotalNumHs(),  # 1
            atom.GetNumRadicalElectrons(),  # 1
            atom.GetHybridization().real,  # 1
            atom.GetIsAromatic(),  # 1
            atom.IsInRing(),  # 1
        ]
        # One-hot encoding for common properties
        features += [int(atom.GetAtomicNum() == i) for i in range(1, 37)]  # 36
        features += [int(atom.GetTotalDegree() == i) for i in range(7)]  # 7
        atom_features.append(features[:48])  # Ensure 48-dim

    x = torch.tensor(atom_features, dtype=torch.float)

    # Bond features (6-dimensional)
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type = bond.GetBondTypeAsDouble()
        is_conjugated = bond.GetIsConjugated()
        is_aromatic = bond.GetIsAromatic()
        is_in_ring = bond.IsInRing()

        edge_features = [
            bond_type,
            is_conjugated,
            is_aromatic,
            is_in_ring,
            0.0,  # Placeholder
            0.0,  # Placeholder
        ]

        # Bidirectional edges
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def mol_to_ecfp(mol: Chem.Mol, radius: int = 2, n_bits: int = 4096) -> np.ndarray:
    """
    Generate ECFP4 fingerprint

    Returns:
        ecfp: Binary fingerprint [4096]
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def mol_to_count_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Generate Count fingerprint for Student

    Returns:
        count_fp: Count fingerprint [2048]
    """
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    for idx, val in fp.GetNonzeroElements().items():
        arr[idx] = min(val, 255)  # Cap at 255
    return arr / 255.0  # Normalize


class NISTDataset(Dataset):
    """
    NIST EI-MS Dataset for NEIMS v2.0

    Supports:
    - MSP file parsing
    - MOL file loading
    - Teacher mode: Graph + ECFP
    - Student mode: ECFP + Count FP

    Memory Efficient Mode (TODO):
    The config.yaml specifies a memory_efficient_mode with lazy loading
    and HDF5 caching for handling large datasets (300k+ molecules) on
    32GB RAM systems. This feature is currently not implemented.

    To implement:
    1. Use HDF5 for on-disk caching (h5py library)
    2. Implement lazy loading in __getitem__
    3. Generate graphs on-the-fly instead of pre-computing
    4. Add gradient accumulation support in trainer
    5. Periodic cache clearing

    Expected benefits:
    - Memory: 70-100x reduction (5GB vs 17-26GB)
    - Trade-off: ~13% slower training speed
    """

    def __init__(
        self,
        data_config: Dict,
        mode: str = 'teacher',  # 'teacher' or 'student'
        split: str = 'train',   # 'train', 'val', 'test'
        augment: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_config: Data configuration from config.yaml
            mode: 'teacher' (GNN+ECFP) or 'student' (ECFP only)
            split: Data split
            augment: Apply data augmentation
            cache_dir: Cache directory for processed data
        """
        self.data_config = data_config
        self.mode = mode
        self.split = split
        self.augment = augment
        self.max_mz = data_config.get('max_mz', 500)

        # Cache setup
        if cache_dir is None:
            cache_dir = data_config.get('output_dir', 'data/processed')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or process data
        cache_file = self.cache_dir / f'nist_{split}_{mode}.pkl'

        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            logger.info(f"Processing NIST data for {split} split...")
            self.data = self._process_data()
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            logger.info(f"Cached data saved to {cache_file}")

        logger.info(f"Loaded {len(self.data)} samples for {split} split")

    def _process_data(self) -> List[Dict]:
        """Process raw NIST data"""
        data = []

        # Try MSP file first
        msp_path = self.data_config.get('nist_msp_path', 'data/NIST17.msp')
        if os.path.exists(msp_path):
            logger.info(f"Parsing MSP file: {msp_path}")
            entries = parse_msp_file(msp_path)

            for entry in entries:
                if 'smiles' not in entry or 'peaks' not in entry:
                    continue

                mol = Chem.MolFromSmiles(entry['smiles'])
                if mol is None:
                    continue

                spectrum = peaks_to_spectrum(entry['peaks'], self.max_mz)

                sample = {
                    'smiles': entry['smiles'],
                    'spectrum': spectrum,
                    'name': entry.get('name', ''),
                }

                # Add features based on mode
                if self.mode == 'teacher':
                    sample['graph'] = mol_to_graph(mol)
                    sample['ecfp'] = mol_to_ecfp(mol)
                else:  # student
                    sample['ecfp'] = mol_to_ecfp(mol)
                    sample['count_fp'] = mol_to_count_fp(mol)

                data.append(sample)

        else:
            # Load from MOL files
            mol_dir = Path(self.data_config.get('mol_files_dir', 'data/mol_files'))
            logger.info(f"Loading MOL files from: {mol_dir}")

            if mol_dir.exists():
                mol_files = sorted(mol_dir.glob('*.MOL'))
                logger.info(f"Found {len(mol_files)} MOL files")

                for mol_file in mol_files:
                    mol = Chem.MolFromMolFile(str(mol_file), sanitize=True, removeHs=False)
                    if mol is None:
                        continue

                    # Generate dummy spectrum (will be replaced with actual data)
                    # In production, load spectrum from paired data file
                    spectrum = np.zeros(self.max_mz + 1, dtype=np.float32)

                    sample = {
                        'smiles': Chem.MolToSmiles(mol),
                        'spectrum': spectrum,
                        'name': mol_file.stem,
                    }

                    if self.mode == 'teacher':
                        sample['graph'] = mol_to_graph(mol)
                        sample['ecfp'] = mol_to_ecfp(mol)
                    else:
                        sample['ecfp'] = mol_to_ecfp(mol)
                        sample['count_fp'] = mol_to_count_fp(mol)

                    data.append(sample)

        # Split data
        np.random.seed(42)
        indices = np.random.permutation(len(data))

        train_ratio = self.data_config.get('train_split', 0.8)
        val_ratio = self.data_config.get('val_split', 0.1)

        n_train = int(len(data) * train_ratio)
        n_val = int(len(data) * val_ratio)

        if self.split == 'train':
            indices = indices[:n_train]
        elif self.split == 'val':
            indices = indices[n_train:n_train + n_val]
        else:  # test
            indices = indices[n_train + n_val:]

        return [data[i] for i in indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            For teacher: {graph, ecfp, spectrum}
            For student: {ecfp, count_fp, spectrum}
        """
        sample = self.data[idx].copy()

        # Data augmentation (if enabled)
        if self.augment and self.split == 'train':
            augmentation_config = self.data_config.get('augmentation', {})

            # 1. Isotope Substitution
            if augmentation_config.get('isotope', {}).get('enabled', False):
                from src.data.augmentation import isotope_substitution
                probability = augmentation_config['isotope'].get('probability', 0.05)
                modified_smiles = isotope_substitution(sample['smiles'], probability)

                # If SMILES changed, regenerate features
                if modified_smiles != sample['smiles']:
                    mol = Chem.MolFromSmiles(modified_smiles)
                    if mol is not None:
                        sample['smiles'] = modified_smiles
                        if self.mode == 'teacher':
                            sample['graph'] = mol_to_graph(mol)
                            sample['ecfp'] = mol_to_ecfp(mol)
                        else:
                            sample['ecfp'] = mol_to_ecfp(mol)
                            sample['count_fp'] = mol_to_count_fp(mol)

            # 2. Conformer Generation (Teacher pretraining only)
            # Note: Conformer generation is computationally expensive,
            # so we only apply it with low probability or skip for now
            # TODO: Implement conformer-based augmentation for Phase 1 pretraining

            # 3. Label Distribution Smoothing
            if augmentation_config.get('lds', {}).get('enabled', False):
                if np.random.rand() < 0.5:
                    from src.data.augmentation import label_distribution_smoothing
                    sigma = augmentation_config['lds'].get('sigma', 1.5)
                    sample['spectrum'] = label_distribution_smoothing(
                        sample['spectrum'],
                        sigma=sigma
                    )

        # Convert to tensors
        sample['spectrum'] = torch.tensor(sample['spectrum'], dtype=torch.float32)

        if self.mode == 'teacher':
            sample['ecfp'] = torch.tensor(sample['ecfp'], dtype=torch.float32)
        else:
            sample['ecfp'] = torch.tensor(sample['ecfp'], dtype=torch.float32)
            sample['count_fp'] = torch.tensor(sample['count_fp'], dtype=torch.float32)

        return sample


def collate_fn_teacher(batch: List[Dict]) -> Dict:
    """Custom collate for Teacher (handles PyG graphs)"""
    from torch_geometric.data import Batch

    graphs = [sample['graph'] for sample in batch]
    ecfps = torch.stack([sample['ecfp'] for sample in batch])
    spectra = torch.stack([sample['spectrum'] for sample in batch])

    graph_batch = Batch.from_data_list(graphs)

    return {
        'graph': graph_batch,
        'ecfp': ecfps,
        'spectrum': spectra
    }


def collate_fn_student(batch: List[Dict]) -> Dict:
    """Custom collate for Student"""
    ecfps = torch.stack([sample['ecfp'] for sample in batch])
    count_fps = torch.stack([sample['count_fp'] for sample in batch])
    spectra = torch.stack([sample['spectrum'] for sample in batch])

    # Concatenate ECFP + Count FP for Student input
    student_input = torch.cat([ecfps, count_fps], dim=-1)  # [batch, 6144]

    return {
        'input': student_input,
        'ecfp': ecfps,
        'count_fp': count_fps,
        'spectrum': spectra
    }
