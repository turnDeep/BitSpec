#!/usr/bin/env python
"""
Test script to verify MOL file and NIST17.MSP ID mapping

This script tests that the dataset correctly maps MOL files (e.g., ID200001.MOL)
to their corresponding entries in NIST17.MSP based on the ID field.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.mol_parser import NISTMSPParser, MOLParser
from src.data.dataset import MassSpecDataset


def test_id_mapping():
    """Test that MOL files are correctly mapped to MSP entries by ID"""
    print("=" * 80)
    print("Testing MOL file and NIST17.MSP ID Mapping")
    print("=" * 80)

    # Parse MSP file
    print("\n1. Parsing NIST17.MSP file...")
    msp_parser = NISTMSPParser()
    compounds = msp_parser.parse_file('data/NIST17.MSP')
    print(f"   Found {len(compounds)} compounds in MSP file")

    # Check MOL files directory
    print("\n2. Checking MOL files directory...")
    mol_dir = Path('data/mol_files')
    mol_files = list(mol_dir.glob('*.MOL'))
    print(f"   Found {len(mol_files)} MOL files")

    # Test ID mapping
    print("\n3. Testing ID mapping...")
    matched = 0
    sample_matches = []

    for compound in compounds[:10]:  # Test first 10
        compound_id = compound.get('ID')
        if compound_id:
            mol_file = mol_dir / f"ID{compound_id}.MOL"
            if mol_file.exists():
                matched += 1
                sample_matches.append({
                    'id': compound_id,
                    'name': compound.get('Name', 'Unknown'),
                    'mol_file': mol_file.name
                })

    print(f"   Successfully mapped {matched}/10 compounds")

    # Show sample mappings
    print("\n4. Sample mappings:")
    for match in sample_matches[:5]:
        print(f"   ID {match['id']}: {match['name']}")
        print(f"   → {match['mol_file']}")

    # Test dataset loading
    print("\n5. Testing MassSpecDataset loading...")
    try:
        dataset = MassSpecDataset(
            msp_file='data/NIST17.MSP',
            mol_files_dir='data/mol_files',
            max_mz=1000,
            mz_bin_size=1.0
        )
        print(f"   ✓ Dataset loaded successfully with {len(dataset)} samples")

        # Test first sample
        if len(dataset) > 0:
            graph_data, spectrum, metadata = dataset[0]
            print(f"\n6. Sample data (ID {metadata['id']}):")
            print(f"   Name: {metadata['name']}")
            print(f"   Formula: {metadata['formula']}")
            print(f"   Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
            print(f"   Node features: {graph_data.x.shape}")
            print(f"   Spectrum: {spectrum.shape}")
            print(f"   Non-zero peaks: {(spectrum > 0).sum()}")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print("\n" + "=" * 80)
    print("ID Mapping Test Complete!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    test_id_mapping()
