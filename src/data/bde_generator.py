#!/usr/bin/env python3
# src/data/bde_generator.py
"""
NEIMS v2.0 BDE (Bond Dissociation Energy) Generator

Uses ALFABET (A machine-Learning derived, Fast, Accurate Bond dissociation Enthalpy Tool)
to generate BDE values for pretraining.

Phase 0 Integration:
- Prefers HDF5 cache created by scripts/precompute_bde.py
- Falls back to direct ALFABET prediction if cache miss
- Supports both Phase 0 workflow (HDF5) and legacy workflow (direct ALFABET)

ALFABET: https://github.com/NREL/alfabet
Paper: https://www.nature.com/articles/s41467-020-16201-z
Dataset: 290,664 BDEs from 42,577 molecules (C, H, O, N only)
"""

import os
import torch
import numpy as np
from rdkit import Chem
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import HDF5
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logger.warning("h5py not available. Install with: pip install h5py")

# Try to import ALFABET (only needed if not using Phase 0 HDF5 cache)
try:
    from alfabet import model as alfabet_model
    ALFABET_AVAILABLE = True
    logger.info("ALFABET successfully imported")
except ImportError:
    ALFABET_AVAILABLE = False
    logger.warning("ALFABET not available. Install with: pip install alfabet")


class BDEGenerator:
    """
    BDE (Bond Dissociation Energy) Generator using ALFABET

    Generates BDE values for all bonds in a molecule for use in pretraining.
    BDE values are normalized to [0, 1] range for neural network training.

    Phase 0 Integration:
    - Prefers HDF5 cache (data/processed/bde_cache/bde_cache.h5) from Phase 0
    - Falls back to direct ALFABET prediction if cache miss
    - Most efficient: Run Phase 0 once, then use HDF5 cache for all training

    Cyclic Bond Handling (ALFABET Limitation):
    - ALFABET only predicts acyclic (non-ring) bonds
    - Aromatic ring bonds: 120 kcal/mol (benzene ~124, maximum stability)
    - Aliphatic ring bonds: 85 kcal/mol (cyclohexane ~83, similar to acyclic)
    - This distinction is critical for EI-MS: aromatic rings rarely fragment at 70eV

    QC-GN2oMS2 Comparison:
    - QC-GN2oMS2 (MS/MS): Uses BDE as static edge features, cyclic bonds = 0
    - NExtIMS v2.0 (EI-MS): Uses BDE as pretraining task target (this class)
      * Learns to predict BDE from molecular structure
      * Cyclic bonds properly handled (aromatic=120, aliphatic=85)
      * Inference does not require ALFABET (knowledge embedded in GNN)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        use_hdf5: bool = True,  # NEW: Use HDF5 cache from Phase 0
        bde_min: float = 50.0,   # Typical minimum BDE (kcal/mol)
        bde_max: float = 120.0,  # Typical maximum BDE (kcal/mol)
        fallback_bde: float = 85.0  # Default BDE for acyclic C-C & aliphatic rings
    ):
        """
        Args:
            cache_dir: Directory containing BDE cache
            use_cache: Whether to use cached BDEs
            use_hdf5: Whether to use HDF5 cache from Phase 0 (recommended)
            bde_min: Minimum BDE for normalization (kcal/mol)
            bde_max: Maximum BDE for normalization (kcal/mol)
            fallback_bde: Fallback BDE value if ALFABET prediction fails
                - Used for acyclic C-C bonds: ~85 kcal/mol (typical)
                - Used for aliphatic rings: ~83-85 kcal/mol (cyclohexane)
                - Aromatic rings use 120 kcal/mol (see predict_bde method)
        """
        self.use_cache = use_cache
        self.use_hdf5 = use_hdf5
        self.bde_min = bde_min
        self.bde_max = bde_max
        self.fallback_bde = fallback_bde

        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = Path("data/processed/bde_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Try to load HDF5 cache (from Phase 0)
        self.hdf5_file = None
        self.hdf5_cache_loaded = False
        if self.use_hdf5 and HDF5_AVAILABLE:
            hdf5_path = self.cache_dir / "bde_cache.h5"
            if hdf5_path.exists():
                try:
                    self.hdf5_file = h5py.File(str(hdf5_path), 'r')
                    num_molecules = self.hdf5_file.attrs.get('num_molecules', len(self.hdf5_file))
                    num_bde_values = self.hdf5_file.attrs.get('num_bde_values', 0)
                    logger.info(f"Loaded HDF5 BDE cache: {num_molecules:,} molecules, {num_bde_values:,} BDE values")
                    logger.info(f"  Cache file: {hdf5_path}")
                    self.hdf5_cache_loaded = True
                except Exception as e:
                    logger.warning(f"Failed to load HDF5 cache: {e}")
                    self.hdf5_file = None
            else:
                logger.info(f"HDF5 cache not found at {hdf5_path}")
                logger.info("Run Phase 0 to create cache: python scripts/precompute_bde.py")

        # Legacy pickle cache (fallback)
        self.cache_file = self.cache_dir / "bde_cache.pkl"
        if self.use_cache and self.cache_file.exists() and not self.hdf5_cache_loaded:
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded pickle BDE cache with {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load pickle cache: {e}")
                self.cache = {}
        else:
            self.cache = {}

        # Initialize ALFABET predictor (only if not using HDF5 cache)
        self.predictor = None
        if not self.hdf5_cache_loaded and ALFABET_AVAILABLE:
            try:
                # ALFABET uses a singleton model, just import it
                self.predictor = alfabet_model
                logger.info("ALFABET predictor initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ALFABET: {e}")
                logger.warning("Falling back to rule-based BDE estimation")
                self.predictor = None
        elif not self.hdf5_cache_loaded:
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

        Priority:
        1. HDF5 cache (from Phase 0) - fastest
        2. Pickle cache (legacy) - fast
        3. ALFABET direct prediction - slow (acyclic bonds only)
        4. Rule-based supplementation for cyclic bonds (ALFABET limitation)
        5. Full rule-based estimation - fallback

        BDE values for cyclic bonds (ALFABET does not predict):
        - Aromatic rings: 120 kcal/mol (benzene ~124, max stability for GNN)
        - Aliphatic rings: 85 kcal/mol (cyclohexane ~83, similar to acyclic)

        Args:
            mol: RDKit molecule
            use_cache: Whether to use cached results

        Returns:
            bde_dict: {bond_idx: BDE (kcal/mol)}
        """
        # Generate cache key from SMILES
        smiles = Chem.MolToSmiles(mol)

        # Priority 1: Check HDF5 cache (Phase 0)
        if use_cache and self.use_cache and self.hdf5_file is not None:
            if smiles in self.hdf5_file:
                try:
                    bde_group = self.hdf5_file[smiles]
                    bde_dict = {}
                    for bond_idx_str in bde_group.keys():
                        bond_idx = int(bond_idx_str)
                        bde_value = float(bde_group[bond_idx_str][()])
                        bde_dict[bond_idx] = bde_value
                    return bde_dict
                except Exception as e:
                    logger.warning(f"Failed to read from HDF5 cache: {e}")

        # Priority 2: Check pickle cache (legacy)
        if use_cache and self.use_cache and smiles in self.cache:
            return self.cache[smiles]

        # Priority 3: ALFABET direct prediction
        bde_dict = {}
        if self.predictor is not None:
            try:
                # ALFABET expects a list of SMILES
                predictions = self.predictor.predict([smiles])
                if predictions and len(predictions) > 0:
                    alfabet_predictions = predictions[0]
                    if alfabet_predictions:
                        bde_dict = alfabet_predictions
            except Exception as e:
                logger.debug(f"ALFABET prediction failed: {e}, using rule-based estimation")

        # Priority 4: Supplement missing bonds (typically cyclic bonds)
        # ALFABET only predicts acyclic bonds, so we need to fill in cyclic bonds
        num_bonds = mol.GetNumBonds()
        if len(bde_dict) < num_bonds:
            for bond_idx, bond in enumerate(mol.GetBonds()):
                if bond_idx not in bde_dict:
                    # Bond not predicted by ALFABET (likely cyclic)
                    if bond.IsInRing():
                        if bond.GetIsAromatic():
                            # Aromatic ring: highly stable in EI-MS
                            # Benzene C-C: ~124 kcal/mol (experimental)
                            # Use 120 to signal maximum stability to GNN
                            bde_dict[bond_idx] = 120.0
                        else:
                            # Aliphatic ring: similar to acyclic
                            # Cyclohexane C-C: ~83 kcal/mol
                            bde_dict[bond_idx] = 85.0
                    else:
                        # Acyclic but ALFABET failed: use fallback
                        bde_dict[bond_idx] = self.fallback_bde

        # Priority 5: If still no predictions, use full rule-based estimation
        if not bde_dict:
            bde_dict = self._rule_based_bde(mol)

        # Cache result
        if self.use_cache:
            self.cache[smiles] = bde_dict

        return bde_dict

    def _rule_based_bde(self, mol: Chem.Mol) -> Dict[int, float]:
        """
        Rule-based BDE estimation (fallback when ALFABET unavailable)

        Based on bond type and chemical environment for EI-MS (70eV):
        - Single bonds: 80-90 kcal/mol (acyclic C-C: ~85)
        - Double bonds: 100-110 kcal/mol
        - Triple bonds: 110-120 kcal/mol
        - Aromatic bonds: 120 kcal/mol (benzene C-C: ~124, rarely breaks in EI-MS)
        - Aliphatic rings: 83-85 kcal/mol (cyclohexane: ~83)
        - Weak bonds (next to heteroatoms): -10 kcal/mol

        References:
        - Blanksby & Ellison, Acc. Chem. Res. 2003, 36, 255-263
        - Aromatic ring stability in EI-MS: fragmentation occurs at β-bond to ring
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
                # Aromatic C-C bonds: ~124-147 kcal/mol (experimental)
                # Use 120 (bde_max) to signal maximum stability to GNN
                # EI-MS (70eV): aromatic rings rarely break, produce prominent M+
                base_bde = 120.0
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
            # Failsafe: predict_bde should cover all bonds, but use fallback just in case
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
        """Destructor: save cache on cleanup and close HDF5"""
        # Close HDF5 file
        if hasattr(self, 'hdf5_file') and self.hdf5_file is not None:
            try:
                self.hdf5_file.close()
            except Exception:
                pass

        # Save pickle cache
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
