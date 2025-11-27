#!/usr/bin/env python3
"""
Test script for BDE generator modifications

Tests the updated BDE values for:
- Aromatic rings: 120 kcal/mol
- Aliphatic rings: 85 kcal/mol
- Acyclic bonds: 85 kcal/mol (or ALFABET prediction)
"""

import sys
sys.path.insert(0, '/home/user/NExtIMS')

from rdkit import Chem
from src.data.bde_generator import BDEGenerator
import logging

logging.basicConfig(level=logging.INFO)

def test_bde_generator():
    """Test BDE generator with various molecular structures"""

    # Initialize BDE generator (without ALFABET for testing)
    bde_gen = BDEGenerator(
        cache_dir="data/processed/bde_cache",
        use_cache=False,  # Disable cache for testing
        use_hdf5=False    # Disable HDF5 for testing
    )

    print("=" * 80)
    print("Testing BDE Generator Modifications")
    print("=" * 80)

    # Test 1: Benzene (aromatic ring)
    print("\n[Test 1] Benzene (c1ccccc1) - Expected: all bonds = 120 kcal/mol")
    mol = Chem.MolFromSmiles("c1ccccc1")
    bde_dict = bde_gen.predict_bde(mol)
    print(f"  Number of bonds: {mol.GetNumBonds()}")
    print(f"  BDE predictions:")
    for bond_idx, bde in bde_dict.items():
        bond = mol.GetBondWithIdx(bond_idx)
        print(f"    Bond {bond_idx}: {bde:.1f} kcal/mol (aromatic={bond.GetIsAromatic()}, in_ring={bond.IsInRing()})")

    # Verify
    expected = 120.0
    all_correct = all(abs(bde - expected) < 0.1 for bde in bde_dict.values())
    print(f"  ✓ PASS" if all_correct else f"  ✗ FAIL")

    # Test 2: Cyclohexane (aliphatic ring)
    print("\n[Test 2] Cyclohexane (C1CCCCC1) - Expected: all bonds = 85 kcal/mol")
    mol = Chem.MolFromSmiles("C1CCCCC1")
    bde_dict = bde_gen.predict_bde(mol)
    print(f"  Number of bonds: {mol.GetNumBonds()}")
    print(f"  BDE predictions:")
    for bond_idx, bde in bde_dict.items():
        bond = mol.GetBondWithIdx(bond_idx)
        print(f"    Bond {bond_idx}: {bde:.1f} kcal/mol (aromatic={bond.GetIsAromatic()}, in_ring={bond.IsInRing()})")

    # Verify
    expected = 85.0
    all_correct = all(abs(bde - expected) < 0.1 for bde in bde_dict.values())
    print(f"  ✓ PASS" if all_correct else f"  ✗ FAIL")

    # Test 3: Hexane (acyclic)
    print("\n[Test 3] Hexane (CCCCCC) - Expected: all bonds = 85 kcal/mol")
    mol = Chem.MolFromSmiles("CCCCCC")
    bde_dict = bde_gen.predict_bde(mol)
    print(f"  Number of bonds: {mol.GetNumBonds()}")
    print(f"  BDE predictions:")
    for bond_idx, bde in bde_dict.items():
        bond = mol.GetBondWithIdx(bond_idx)
        print(f"    Bond {bond_idx}: {bde:.1f} kcal/mol (aromatic={bond.GetIsAromatic()}, in_ring={bond.IsInRing()})")

    # Verify
    expected = 85.0
    all_correct = all(abs(bde - expected) < 0.1 for bde in bde_dict.values())
    print(f"  ✓ PASS" if all_correct else f"  ✗ FAIL")

    # Test 4: Toluene (mixed: aromatic ring + acyclic side chain)
    print("\n[Test 4] Toluene (Cc1ccccc1) - Expected: aromatic=120, methyl-phenyl=85")
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    bde_dict = bde_gen.predict_bde(mol)
    print(f"  Number of bonds: {mol.GetNumBonds()}")
    print(f"  BDE predictions:")
    aromatic_bonds = []
    acyclic_bonds = []
    for bond_idx, bde in bde_dict.items():
        bond = mol.GetBondWithIdx(bond_idx)
        is_aromatic = bond.GetIsAromatic()
        in_ring = bond.IsInRing()
        print(f"    Bond {bond_idx}: {bde:.1f} kcal/mol (aromatic={is_aromatic}, in_ring={in_ring})")
        if is_aromatic:
            aromatic_bonds.append(bde)
        else:
            acyclic_bonds.append(bde)

    # Verify
    aromatic_correct = all(abs(bde - 120.0) < 0.1 for bde in aromatic_bonds)
    acyclic_correct = all(abs(bde - 85.0) < 0.1 for bde in acyclic_bonds)
    print(f"  Aromatic bonds: {len(aromatic_bonds)} (expected ~120) - {'✓ PASS' if aromatic_correct else '✗ FAIL'}")
    print(f"  Acyclic bonds: {len(acyclic_bonds)} (expected ~85) - {'✓ PASS' if acyclic_correct else '✗ FAIL'}")

    # Test 5: Pyridine (aromatic heterocycle)
    print("\n[Test 5] Pyridine (c1ccncc1) - Expected: all aromatic bonds = 120 kcal/mol")
    mol = Chem.MolFromSmiles("c1ccncc1")
    bde_dict = bde_gen.predict_bde(mol)
    print(f"  Number of bonds: {mol.GetNumBonds()}")
    print(f"  BDE predictions:")
    for bond_idx, bde in bde_dict.items():
        bond = mol.GetBondWithIdx(bond_idx)
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        atom_symbols = f"{begin_atom.GetSymbol()}-{end_atom.GetSymbol()}"
        print(f"    Bond {bond_idx} ({atom_symbols}): {bde:.1f} kcal/mol (aromatic={bond.GetIsAromatic()})")

    # Test 6: Normalization
    print("\n[Test 6] Normalization test")
    print(f"  BDE range: {bde_gen.bde_min} - {bde_gen.bde_max} kcal/mol")
    print(f"  Normalized 85.0: {bde_gen.normalize_bde(85.0):.3f}")
    print(f"  Normalized 120.0: {bde_gen.normalize_bde(120.0):.3f}")
    print(f"  Expected: 85→0.5, 120→1.0")

    norm_85 = bde_gen.normalize_bde(85.0)
    norm_120 = bde_gen.normalize_bde(120.0)
    print(f"  ✓ PASS" if (abs(norm_85 - 0.5) < 0.01 and abs(norm_120 - 1.0) < 0.01) else f"  ✗ FAIL")

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print("All tests completed. Check results above.")
    print("\nKey improvements:")
    print("  1. Aromatic rings: 95 → 120 kcal/mol (closer to experimental ~124)")
    print("  2. Aliphatic rings: 85 kcal/mol (cyclohexane ~83)")
    print("  3. Acyclic bonds: 85 kcal/mol (standard C-C)")
    print("  4. Normalization: aromatic=1.0 (max), others=0.5 (mid)")
    print("  5. EI-MS relevance: aromatic rings stable at 70eV")

if __name__ == "__main__":
    test_bde_generator()
