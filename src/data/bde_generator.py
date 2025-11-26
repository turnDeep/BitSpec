#!/usr/bin/env python3
# src/data/bde_generator.py
"""
NEIMS v2.0 BDE (Bond Dissociation Energy) Generator

Uses ALFABET (A machine-Learning derived, Fast, Accurate Bond dissociation Enthalpy Tool)
to generate BDE values for pretraining.

ALFABET: https://github.com/NREL/alfabet
Paper: https://www.nature.com/articles/s41467-020-16201-z
Dataset: 290,664 BDEs from 42,577 molecules (C, H, O, N only)

Installation:
    pip install alfabet

Auto-Download:
    ALFABET model weights (~50MB) are automatically downloaded from
    https://github.com/pstjohn/alfabet-models/releases on first use.
    The model files are cached in your home directory for future use.
"""

import torch
import numpy as np
from rdkit import Chem
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import ALFABET
try:
    from alfabet import model as alfabet_model
    ALFABET_AVAILABLE = True
    logger.info("ALFABET successfully imported")
except ImportError:
    ALFABET_AVAILABLE = False
    alfabet_model = None
    logger.warning("ALFABET not available. Install with: pip install alfabet")
except Exception as e:
    ALFABET_AVAILABLE = False
    alfabet_model = None
    logger.warning(f"ALFABET import failed: {e}. Install with: pip install alfabet")


class BDEGenerator:
    """
    BDE (Bond Dissociation Energy) Generator using ALFABET

    Generates BDE values for all bonds in a molecule for use in pretraining.
    BDE values are normalized to [0, 1] range for neural network training.

    QC-GN2oMS2 Comparison:
    - QC-GN2oMS2: Uses BDE as static edge features
    - NExtIMS v2.0: Uses BDE as pretraining task target (this class)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        bde_min: float = 50.0,   # Typical minimum BDE (kcal/mol)
        bde_max: float = 120.0,  # Typical maximum BDE (kcal/mol)
        fallback_bde: float = 85.0  # Default BDE if ALFABET fails
    ):
        """
        Args:
            cache_dir: Directory to cache computed BDEs
            use_cache: Whether to use cached BDEs
            bde_min: Minimum BDE for normalization (kcal/mol)
            bde_max: Maximum BDE for normalization (kcal/mol)
            fallback_bde: Fallback BDE value if prediction fails
        """
        self.use_cache = use_cache
        self.bde_min = bde_min
        self.bde_max = bde_max
        self.fallback_bde = fallback_bde

        # Setup cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "bde_cache.pkl"

            # Load existing cache
            if self.use_cache and self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded BDE cache with {len(self.cache)} entries")
            else:
                self.cache = {}
        else:
            self.cache_dir = None
            self.cache = {}

        # Initialize ALFABET model reference
        if ALFABET_AVAILABLE and alfabet_model is not None:
            try:
                # ALFABET model is ready to use after import
                # Model weights are automatically downloaded on first use
                self.use_alfabet = True
                logger.info("ALFABET model ready for predictions")
            except Exception as e:
                logger.error(f"Failed to initialize ALFABET: {e}")
                logger.warning("Falling back to rule-based BDE estimation")
                self.use_alfabet = False
        else:
            self.use_alfabet = False
            logger.warning("ALFABET not available, using rule-based BDE estimation")

    def normalize_bde(self, bde: float) -> float:
        """Normalize BDE to [0, 1] range"""
        return (bde - self.bde_min) / (self.bde_max - self.bde_min)

    def denormalize_bde(self, bde_normalized: float) -> float:
        """Denormalize BDE from [0, 1] to kcal/mol"""
        return bde_normalized * (self.bde_max - self.bde_min) + self.bde_min

    def predict_bde(self, mol: Chem.Mol, use_cache: bool = True) -> Dict[int, float]:
        """
        Predict BDE for all bonds in molecule

        Args:
            mol: RDKit molecule
            use_cache: Whether to use cached results

        Returns:
            bde_dict: {bond_idx: BDE (kcal/mol)}
        """
        # Generate cache key from SMILES
        smiles = Chem.MolToSmiles(mol)

        # Check cache
        if use_cache and self.use_cache and smiles in self.cache:
            return self.cache[smiles]

        # Predict BDE
        if self.use_alfabet and alfabet_model is not None:
            try:
                # ALFABET prediction
                # alfabet.model.predict() expects a list of SMILES strings
                # Returns DataFrame with columns: molecule, bond_index, bond_type,
                # fragment1, fragment2, bde_pred, is_valid
                predictions = alfabet_model.predict([smiles], verbose=False)

                # Convert DataFrame to dict {bond_idx: BDE}
                bde_dict = {}
                for _, row in predictions.iterrows():
                    if row['is_valid']:
                        bond_idx = int(row['bond_index'])
                        bde_value = float(row['bde_pred'])
                        bde_dict[bond_idx] = bde_value
                    else:
                        # Use fallback for invalid predictions
                        bond_idx = int(row['bond_index'])
                        bde_dict[bond_idx] = self.fallback_bde

                # If no valid predictions, fall back to rule-based
                if not bde_dict:
                    logger.debug(f"No valid ALFABET predictions for {smiles}, using rule-based estimation")
                    bde_dict = self._rule_based_bde(mol)

            except Exception as e:
                logger.debug(f"ALFABET prediction failed: {e}, using rule-based estimation")
                bde_dict = self._rule_based_bde(mol)
        else:
            # Rule-based estimation
            bde_dict = self._rule_based_bde(mol)

        # Cache result
        if self.use_cache:
            self.cache[smiles] = bde_dict

        return bde_dict

    def _rule_based_bde(self, mol: Chem.Mol) -> Dict[int, float]:
        """
        Rule-based BDE estimation (fallback when ALFABET unavailable)

        Based on bond type and chemical environment:
        - Single bonds: 80-90 kcal/mol
        - Double bonds: 100-110 kcal/mol
        - Triple bonds: 110-120 kcal/mol
        - Aromatic bonds: 90-100 kcal/mol
        - Weak bonds (next to heteroatoms): -10 kcal/mol
        """
        bde_dict = {}

        for bond_idx, bond in enumerate(mol.GetBonds()):
            # Base BDE by bond type
            bond_type = bond.GetBondTypeAsDouble()

            if bond_type == 1.0:  # Single bond
                base_bde = 85.0
            elif bond_type == 2.0:  # Double bond
                base_bde = 105.0
            elif bond_type == 3.0:  # Triple bond
                base_bde = 115.0
            elif bond.GetIsAromatic():  # Aromatic
                base_bde = 95.0
            else:
                base_bde = self.fallback_bde

            # Adjust for chemical environment
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            # Weaken bonds next to heteroatoms (O, N)
            if begin_atom.GetAtomicNum() in [7, 8] or end_atom.GetAtomicNum() in [7, 8]:
                base_bde -= 10.0

            # Weaken bonds in conjugated systems
            if bond.GetIsConjugated():
                base_bde -= 5.0

            bde_dict[bond_idx] = base_bde

        return bde_dict

    def get_bde_edge_features(
        self,
        mol: Chem.Mol,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Get BDE values as edge features for PyG graph

        Args:
            mol: RDKit molecule
            normalize: Whether to normalize BDEs to [0, 1]

        Returns:
            bde_features: [num_edges, 1] tensor (bidirectional)
        """
        bde_dict = self.predict_bde(mol)
        num_bonds = len(list(mol.GetBonds()))

        # Create bidirectional edge features
        bde_features = []
        for bond_idx in range(num_bonds):
            bde = bde_dict.get(bond_idx, self.fallback_bde)

            if normalize:
                bde = self.normalize_bde(bde)

            # Bidirectional: same BDE for both directions
            bde_features.append([bde])
            bde_features.append([bde])

        return torch.tensor(bde_features, dtype=torch.float32)

    def get_bde_targets(
        self,
        mol: Chem.Mol,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Get BDE values as regression targets for pretraining

        Args:
            mol: RDKit molecule
            normalize: Whether to normalize BDEs

        Returns:
            bde_targets: [num_edges, 1] tensor (bidirectional)
        """
        return self.get_bde_edge_features(mol, normalize=normalize)

    def save_cache(self):
        """Save BDE cache to disk"""
        if self.cache_dir and self.use_cache:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved BDE cache with {len(self.cache)} entries")

    def __del__(self):
        """Destructor: save cache on cleanup"""
        if hasattr(self, 'cache_dir') and self.cache_dir:
            self.save_cache()


class BDEFeatureAugmenter:
    """
    Augments existing molecular graphs with BDE features

    This class adds BDE information to existing PyG Data objects.
    """

    def __init__(self, bde_generator: BDEGenerator):
        self.bde_gen = bde_generator

    def augment_graph(self, graph_data, mol: Chem.Mol) -> None:
        """
        Add BDE features to existing PyG graph (in-place)

        Args:
            graph_data: PyG Data object
            mol: RDKit molecule
        """
        bde_features = self.bde_gen.get_bde_edge_features(mol, normalize=True)

        # Concatenate BDE features to existing edge_attr
        if graph_data.edge_attr is not None:
            graph_data.edge_attr = torch.cat([
                graph_data.edge_attr,
                bde_features
            ], dim=-1)
        else:
            graph_data.edge_attr = bde_features

    def add_bde_targets(self, graph_data, mol: Chem.Mol) -> None:
        """
        Add BDE regression targets to PyG graph (in-place)

        Args:
            graph_data: PyG Data object
            mol: RDKit molecule
        """
        bde_targets = self.bde_gen.get_bde_targets(mol, normalize=True)
        graph_data.bde_targets = bde_targets


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create BDE generator
    bde_gen = BDEGenerator(
        cache_dir="data/processed/bde_cache",
        use_cache=True
    )

    # Test molecule: Aspirin
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)

    # Predict BDEs
    bde_dict = bde_gen.predict_bde(mol)

    print(f"\nBDE prediction for {smiles}:")
    print(f"Number of bonds: {len(bde_dict)}")
    for bond_idx, bde in bde_dict.items():
        bond = mol.GetBondWithIdx(bond_idx)
        print(f"  Bond {bond_idx} ({bond.GetBondType()}): {bde:.2f} kcal/mol")

    # Get edge features
    bde_features = bde_gen.get_bde_edge_features(mol)
    print(f"\nBDE edge features shape: {bde_features.shape}")

    # Save cache
    bde_gen.save_cache()
