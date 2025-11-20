#!/usr/bin/env python3
# src/data/augmentation.py
"""
NEIMS v2.0 Data Augmentation

Implements LDS, Isotope Substitution, and Conformer Generation.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.ndimage import gaussian_filter1d


def label_distribution_smoothing(
    spectrum: np.ndarray,
    sigma: float = 1.5
) -> np.ndarray:
    """
    Label Distribution Smoothing (LDS)
    
    Applies Gaussian smoothing to smooth neighboring m/z peaks.
    
    Args:
        spectrum: Input spectrum [501]
        sigma: Gaussian sigma in m/z units
        
    Returns:
        smoothed: Smoothed spectrum [501]
    """
    smoothed = gaussian_filter1d(spectrum, sigma=sigma, mode='constant')
    return smoothed


def isotope_substitution(smiles: str, probability: float = 0.05) -> str:
    """
    Isotope Substitution: C12 -> C13
    
    Args:
        smiles: Input SMILES
        probability: Probability of substitution per molecule
        
    Returns:
        modified_smiles: SMILES with isotope substitution
    """
    if np.random.rand() > probability:
        return smiles
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    
    # Find carbon atoms
    carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
    
    if len(carbon_indices) == 0:
        return smiles
    
    # Randomly select 1-2 carbons
    num_substitutions = min(np.random.randint(1, 3), len(carbon_indices))
    selected_carbons = np.random.choice(carbon_indices, num_substitutions, replace=False)
    
    # Set isotope to C13
    for idx in selected_carbons:
        mol.GetAtomWithIdx(int(idx)).SetIsotope(13)
    
    return Chem.MolToSmiles(mol)


def generate_conformers(smiles: str, num_conformers: int = 5) -> list:
    """
    Generate 3D conformers
    
    Args:
        smiles: Input SMILES
        num_conformers: Number of conformers to generate
        
    Returns:
        conformer_mols: List of RDKit molecules with 3D coordinates
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    mol = Chem.AddHs(mol)
    
    # Generate conformers
    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_conformers,
        randomSeed=42,
        pruneRmsThresh=0.5
    )
    
    # Optimize geometry
    for conf_id in range(mol.GetNumConformers()):
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
    
    return [mol] * mol.GetNumConformers()
