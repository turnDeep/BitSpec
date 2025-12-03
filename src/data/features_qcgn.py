#!/usr/bin/env python3
# src/data/features_qcgn.py
"""
NExtIMS v4.2: QC-GN2oMS2-style Minimal Feature Extractor

Implements minimal node and edge features following QC-GN2oMS2 design:
- Node features: 16 dimensions (15 atom types + 1 ionization energy)
- Edge features: 3 dimensions (bond_order + BDE + in_ring)

This replaces the v2.0 48-dim node / 6-dim edge approach with a minimal
configuration optimized for MS/MS prediction, adapted for EI-MS.

Design Philosophy:
- Minimize feature dimensions to reduce overfitting
- Use only essential chemical information
- Leverage GATv2Conv's attention mechanism for feature learning
- Pre-compute BDE from BonDNet for edge enrichment

References:
- QC-GN2oMS2: https://github.com/PNNL-m-q/QC-GN2oMS2
- Paper: "Quantum Chemistry-Augmented Graph Neural Network for Accurate
          Prediction of Molecular Spectral Properties"
- Code: qcgnoms/qc2.py (lines 82-96 for node features)
"""

import numpy as np
import torch
from rdkit import Chem
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QCGNFeaturizer:
    """
    QC-GN2oMS2-style minimal molecular featurizer for v4.2

    Node Features (16-dim):
        - 15 atom types (one-hot via mass matching):
          C, O, H, N, S, Si, P, Cl, F, I, Se, As, B, Br, Sn
        - 1 ionization energy: 70.0 eV (fixed for EI-MS)

    Edge Features (3-dim):
        - bond_order: 1.0 (single), 2.0 (double), 3.0 (triple), 1.5 (aromatic)
        - BDE: Bond Dissociation Energy (kcal/mol), normalized [0, 1]
        - in_ring: Binary indicator

    Key Differences from v2.0:
        - Node: 48-dim → 16-dim (67% reduction)
        - Edge: 6-dim → 3-dim (50% reduction)
        - Removes: formal charge, hybridization, chirality, conjugation
        - Adds: BDE (critical for fragmentation prediction)
    """

    # Atom masses for one-hot encoding (matching QC-GN2oMS2)
    ATOM_MASSES = np.array([
        12.011,  # C  (0)
        15.999,  # O  (1)
        1.008,   # H  (2)
        14.007,  # N  (3)
        32.067,  # S  (4)
        28.086,  # Si (5)
        30.974,  # P  (6)
        35.453,  # Cl (7)
        18.998,  # F  (8)
        126.904, # I  (9)
        78.96,   # Se (10)
        74.922,  # As (11)
        10.812,  # B  (12)
        79.904,  # Br (13)
        118.711  # Sn (14)
    ], dtype=np.float32)

    # Ionization energy for EI-MS (70 eV is standard)
    IONIZATION_ENERGY = 70.0  # eV

    # BDE normalization range (kcal/mol)
    BDE_MIN = 50.0   # Weak bonds (e.g., peroxides)
    BDE_MAX = 200.0  # Strong bonds (e.g., alkynes)

    def __init__(
        self,
        use_bde: bool = True,
        bde_min: float = 50.0,
        bde_max: float = 200.0
    ):
        """
        Initialize QC-GN featurizer

        Args:
            use_bde: Whether to include BDE in edge features
            bde_min: Minimum BDE for normalization (kcal/mol)
            bde_max: Maximum BDE for normalization (kcal/mol)
        """
        self.use_bde = use_bde
        self.bde_min = bde_min
        self.bde_max = bde_max

        logger.info("QCGNFeaturizer initialized:")
        logger.info(f"  Node features: 16-dim (15 atom types + ionization energy)")
        logger.info(f"  Edge features: 3-dim (bond_order + BDE + in_ring)")
        logger.info(f"  BDE enabled: {use_bde}")
        logger.info(f"  BDE range: {bde_min}-{bde_max} kcal/mol")

    def get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        """
        Get minimal node features (16-dim)

        Implementation matches QC-GN2oMS2 (qc2.py lines 82-96):
        - Mass-based one-hot encoding for 15 atom types
        - Fixed ionization energy (70 eV for EI-MS)

        Args:
            atom: RDKit Atom object

        Returns:
            features: [16] numpy array
        """
        features = np.zeros(16, dtype=np.float32)

        # Atom type via mass matching (15-dim one-hot)
        atom_mass = atom.GetMass()
        mass_diffs = np.abs(self.ATOM_MASSES - atom_mass)
        mass_idx = np.argmin(mass_diffs)

        # Only mark as known atom if mass matches within tolerance
        if mass_diffs[mass_idx] < 0.1:  # Tolerance: 0.1 amu
            features[mass_idx] = 1.0
        else:
            # Unknown atom type - all zeros (will be rare in NIST17)
            logger.debug(f"Unknown atom with mass {atom_mass:.3f} (symbol: {atom.GetSymbol()})")

        # Ionization energy (16th dimension)
        features[15] = self.IONIZATION_ENERGY

        return features

    def get_edge_features(
        self,
        bond: Chem.Bond,
        bde_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Get minimal edge features (3-dim)

        Args:
            bond: RDKit Bond object
            bde_value: BDE value in kcal/mol (optional)

        Returns:
            features: [3] numpy array
                [bond_order, BDE_normalized, in_ring]
        """
        features = np.zeros(3, dtype=np.float32)

        # Bond order (dimension 0)
        bond_type_map = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 2.0,
            Chem.BondType.TRIPLE: 3.0,
            Chem.BondType.AROMATIC: 1.5
        }
        features[0] = bond_type_map.get(bond.GetBondType(), 1.0)

        # BDE normalized (dimension 1)
        if self.use_bde and bde_value is not None:
            # Normalize to [0, 1]
            bde_normalized = (bde_value - self.bde_min) / (self.bde_max - self.bde_min)
            # Clip to [0, 1]
            bde_normalized = np.clip(bde_normalized, 0.0, 1.0)
            features[1] = bde_normalized
        else:
            # No BDE available - use default (0.5 = mid-range)
            features[1] = 0.5

        # In ring (dimension 2)
        features[2] = float(bond.IsInRing())

        return features

    def normalize_bde(self, bde: float) -> float:
        """
        Normalize BDE to [0, 1] range

        Args:
            bde: BDE value in kcal/mol

        Returns:
            bde_normalized: Normalized BDE [0, 1]
        """
        bde_norm = (bde - self.bde_min) / (self.bde_max - self.bde_min)
        return np.clip(bde_norm, 0.0, 1.0)

    def denormalize_bde(self, bde_normalized: float) -> float:
        """
        Denormalize BDE from [0, 1] to kcal/mol

        Args:
            bde_normalized: Normalized BDE [0, 1]

        Returns:
            bde: BDE value in kcal/mol
        """
        return bde_normalized * (self.bde_max - self.bde_min) + self.bde_min

    def get_node_dim(self) -> int:
        """Get node feature dimension"""
        return 16

    def get_edge_dim(self) -> int:
        """Get edge feature dimension"""
        return 3

    def validate_molecule(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """
        Validate if molecule is compatible with featurizer

        Args:
            mol: RDKit Mol object

        Returns:
            (is_valid, error_message)
        """
        if mol is None:
            return False, "Invalid molecule (None)"

        # Check for atoms
        if mol.GetNumAtoms() == 0:
            return False, "No atoms in molecule"

        # Check for unsupported elements
        unsupported_elements = []
        for atom in mol.GetAtoms():
            atom_mass = atom.GetMass()
            mass_diffs = np.abs(self.ATOM_MASSES - atom_mass)
            if mass_diffs.min() >= 0.1:
                unsupported_elements.append(atom.GetSymbol())

        if unsupported_elements:
            unique_elements = sorted(set(unsupported_elements))
            return False, f"Unsupported elements: {', '.join(unique_elements)}"

        return True, "OK"

    def get_feature_info(self) -> Dict:
        """
        Get featurizer information

        Returns:
            info: Dictionary with feature dimensions and settings
        """
        return {
            'node_dim': 16,
            'edge_dim': 3,
            'atom_types': [
                'C', 'O', 'H', 'N', 'S', 'Si', 'P', 'Cl',
                'F', 'I', 'Se', 'As', 'B', 'Br', 'Sn'
            ],
            'ionization_energy_eV': self.IONIZATION_ENERGY,
            'use_bde': self.use_bde,
            'bde_range_kcal_mol': (self.bde_min, self.bde_max),
            'edge_features': ['bond_order', 'BDE_normalized', 'in_ring']
        }


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Initialize featurizer
    featurizer = QCGNFeaturizer()

    # Print feature info
    info = featurizer.get_feature_info()
    print("\n" + "="*60)
    print("QC-GN Featurizer Information")
    print("="*60)
    print(f"Node dimension: {info['node_dim']}")
    print(f"Edge dimension: {info['edge_dim']}")
    print(f"Atom types: {', '.join(info['atom_types'])}")
    print(f"Ionization energy: {info['ionization_energy_eV']} eV")
    print(f"BDE enabled: {info['use_bde']}")
    print(f"BDE range: {info['bde_range_kcal_mol'][0]}-{info['bde_range_kcal_mol'][1]} kcal/mol")
    print(f"Edge features: {', '.join(info['edge_features'])}")

    # Test molecules
    test_smiles = [
        ("c1ccccc1", "Benzene"),
        ("CC(=O)O", "Acetic acid"),
        ("CCO", "Ethanol"),
        ("CCCCCCCC", "Octane")
    ]

    print("\n" + "="*60)
    print("Testing Molecules")
    print("="*60)

    for smiles, name in test_smiles:
        mol = Chem.MolFromSmiles(smiles)

        # Validate
        is_valid, msg = featurizer.validate_molecule(mol)
        print(f"\n{name}: {smiles}")
        print(f"  Valid: {is_valid} ({msg})")

        if not is_valid:
            continue

        # Node features
        print(f"  Atoms: {mol.GetNumAtoms()}")
        for i, atom in enumerate(mol.GetAtoms()):
            features = featurizer.get_atom_features(atom)
            atom_type_idx = np.argmax(features[:15])
            atom_type = info['atom_types'][atom_type_idx] if features[atom_type_idx] > 0 else "Unknown"
            print(f"    Atom {i} ({atom.GetSymbol()}): type={atom_type}, features shape={features.shape}")

        # Edge features (without BDE)
        print(f"  Bonds: {mol.GetNumBonds()}")
        for i, bond in enumerate(mol.GetBonds()):
            features = featurizer.get_edge_features(bond, bde_value=85.0)  # Sample BDE
            print(f"    Bond {i}: order={features[0]:.1f}, BDE_norm={features[1]:.3f}, in_ring={int(features[2])}")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
