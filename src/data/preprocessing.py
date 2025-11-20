#!/usr/bin/env python3
# src/data/preprocessing.py
"""
NEIMS v2.0 Data Preprocessing Utilities

Common utilities for data processing, validation, and normalization.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import logging

logger = logging.getLogger(__name__)


def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string

    Args:
        smiles: SMILES string

    Returns:
        valid: True if valid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize SMILES string

    Args:
        smiles: Input SMILES

    Returns:
        canonical: Canonical SMILES or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def filter_by_molecular_weight(
    smiles: str,
    min_mw: float = 50.0,
    max_mw: float = 1000.0
) -> bool:
    """
    Filter molecules by molecular weight

    Args:
        smiles: SMILES string
        min_mw: Minimum molecular weight
        max_mw: Maximum molecular weight

    Returns:
        valid: True if within range
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mw = Descriptors.MolWt(mol)
        return min_mw <= mw <= max_mw
    except:
        return False


def filter_by_num_atoms(
    smiles: str,
    min_atoms: int = 3,
    max_atoms: int = 100
) -> bool:
    """
    Filter molecules by number of atoms

    Args:
        smiles: SMILES string
        min_atoms: Minimum number of atoms
        max_atoms: Maximum number of atoms

    Returns:
        valid: True if within range
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        num_atoms = mol.GetNumAtoms()
        return min_atoms <= num_atoms <= max_atoms
    except:
        return False


def normalize_spectrum(
    spectrum: np.ndarray,
    method: str = 'max'
) -> np.ndarray:
    """
    Normalize spectrum

    Args:
        spectrum: Input spectrum
        method: 'max' or 'l2'

    Returns:
        normalized: Normalized spectrum
    """
    spectrum = spectrum.astype(np.float32)

    if method == 'max':
        max_val = spectrum.max()
        if max_val > 0:
            return spectrum / max_val
        return spectrum

    elif method == 'l2':
        norm = np.linalg.norm(spectrum)
        if norm > 0:
            return spectrum / norm
        return spectrum

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def remove_noise_peaks(
    spectrum: np.ndarray,
    min_intensity: float = 0.001
) -> np.ndarray:
    """
    Remove noise peaks below threshold

    Args:
        spectrum: Input spectrum
        min_intensity: Minimum intensity threshold

    Returns:
        cleaned: Cleaned spectrum
    """
    spectrum = spectrum.copy()
    spectrum[spectrum < min_intensity] = 0.0
    return spectrum


def peaks_to_spectrum_array(
    peaks: List[Tuple[float, float]],
    max_mz: int = 500,
    bin_size: float = 1.0
) -> np.ndarray:
    """
    Convert peak list to binned spectrum array

    Args:
        peaks: List of (m/z, intensity) tuples
        max_mz: Maximum m/z value
        bin_size: Bin size in m/z units

    Returns:
        spectrum: Binned spectrum [max_mz + 1]
    """
    num_bins = int(max_mz / bin_size) + 1
    spectrum = np.zeros(num_bins, dtype=np.float32)

    for mz, intensity in peaks:
        if 0 <= mz <= max_mz:
            bin_idx = int(round(mz / bin_size))
            if bin_idx < num_bins:
                # Use max intensity if multiple peaks in same bin
                spectrum[bin_idx] = max(spectrum[bin_idx], intensity)

    return spectrum


def spectrum_to_peaks(
    spectrum: np.ndarray,
    threshold: float = 0.01,
    max_peaks: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    Convert spectrum array to peak list

    Args:
        spectrum: Spectrum array
        threshold: Minimum intensity threshold
        max_peaks: Maximum number of peaks to return

    Returns:
        peaks: List of (m/z, intensity) tuples
    """
    # Find peaks above threshold
    peak_indices = np.where(spectrum > threshold)[0]
    peaks = [(float(idx), float(spectrum[idx])) for idx in peak_indices]

    # Sort by intensity (descending)
    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

    # Limit number of peaks
    if max_peaks is not None:
        peaks = peaks[:max_peaks]

    # Sort by m/z (ascending)
    peaks = sorted(peaks, key=lambda x: x[0])

    return peaks


def compute_spectrum_statistics(spectrum: np.ndarray) -> dict:
    """
    Compute spectrum statistics

    Args:
        spectrum: Spectrum array

    Returns:
        stats: Dictionary of statistics
    """
    num_peaks = np.count_nonzero(spectrum)
    max_intensity = spectrum.max()
    mean_intensity = spectrum[spectrum > 0].mean() if num_peaks > 0 else 0.0
    base_peak_mz = spectrum.argmax()

    return {
        'num_peaks': int(num_peaks),
        'max_intensity': float(max_intensity),
        'mean_intensity': float(mean_intensity),
        'base_peak_mz': int(base_peak_mz),
        'total_intensity': float(spectrum.sum())
    }


def compute_molecular_descriptors(smiles: str) -> dict:
    """
    Compute molecular descriptors

    Args:
        smiles: SMILES string

    Returns:
        descriptors: Dictionary of descriptors
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'tpsa': rdMolDescriptors.CalcTPSA(mol),
        }
    except Exception as e:
        logger.warning(f"Failed to compute descriptors for {smiles}: {e}")
        return {}


def split_dataset(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset indices into train/val/test

    Args:
        n_samples: Total number of samples
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        train_idx, val_idx, test_idx
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    np.random.seed(seed)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    logger.info(f"Dataset split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return train_idx, val_idx, test_idx


def batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    Move batch data to device

    Args:
        batch: Batch dictionary
        device: Target device

    Returns:
        batch: Batch on device
    """
    batch_on_device = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_on_device[key] = value.to(device, non_blocking=True)
        elif hasattr(value, 'to'):  # PyG Data object
            batch_on_device[key] = value.to(device)
        else:
            batch_on_device[key] = value

    return batch_on_device


def compute_class_weights(
    spectra: np.ndarray,
    num_bins: int = 10
) -> np.ndarray:
    """
    Compute class weights for imbalanced spectrum bins

    Args:
        spectra: Array of spectra [N, 501]
        num_bins: Number of intensity bins

    Returns:
        weights: Class weights [501]
    """
    # Count non-zero occurrences per m/z bin
    counts = np.count_nonzero(spectra, axis=0) + 1  # Add 1 to avoid division by zero

    # Inverse frequency weighting
    weights = 1.0 / counts
    weights = weights / weights.sum()  # Normalize

    return weights.astype(np.float32)


def augment_spectrum_noise(
    spectrum: np.ndarray,
    noise_level: float = 0.01
) -> np.ndarray:
    """
    Add Gaussian noise to spectrum for augmentation

    Args:
        spectrum: Input spectrum
        noise_level: Noise standard deviation

    Returns:
        augmented: Augmented spectrum
    """
    noise = np.random.normal(0, noise_level, spectrum.shape)
    augmented = spectrum + noise
    augmented = np.clip(augmented, 0, 1)  # Keep in [0, 1]
    return augmented.astype(np.float32)
