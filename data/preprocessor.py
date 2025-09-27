import torch
from rdkit import Chem
from torch_geometric.data import Data
import numpy as np

# --- Feature Extraction Helpers ---

def get_atom_features(atom):
    """
    Extracts features for a single atom.

    Args:
        atom (rdkit.Chem.rdchem.Atom): An RDKit atom object.

    Returns:
        list: A list of numerical features for the atom.
    """
    # Based on design-doc.md
    # Using simple features for now, can be extended with one-hot encodings.
    features = [
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetDegree(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]
    return features

def get_bond_features(bond):
    """
    Extracts features for a single bond.

    Args:
        bond (rdkit.Chem.rdchem.Bond): An RDKit bond object.

    Returns:
        list: A list of numerical features for the bond.
    """
    # Based on design-doc.md
    features = [
        int(bond.GetBondType()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]
    # Note: Stereo chemistry is more complex and omitted for now.
    return features

# --- Main Preprocessing Function ---

def mol_to_graph_data(mol, spectrum_data):
    """
    Converts an RDKit molecule and its spectrum data into a PyG Data object.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The input molecule.
        spectrum_data (dict): The corresponding spectrum data from the MSP parser.
                              Should contain at least a 'peaks' numpy array.

    Returns:
        torch_geometric.data.Data: A graph data object ready for GCN, or None if fails.
    """
    if mol is None:
        return None

    # --- Node (Atom) Features ---
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # --- Edge (Bond) Index and Features ---
    if mol.GetNumBonds() > 0:
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Add edges in both directions for undirected graph
            edge_indices.append((i, j))
            edge_indices.append((j, i))

            bond_feats = get_bond_features(bond)
            edge_features.append(bond_feats)
            edge_features.append(bond_feats) # Same features for the other direction

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # Handle molecules with no bonds (e.g., single atoms)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float) # Match bond feature dim

    # --- Target (Spectrum) Data ---
    # We need to convert the sparse peak list into a dense vector for training.
    # The output dimension is 1000, representing m/z values from 1 to 1000.
    peaks = spectrum_data.get('peaks')
    if peaks is None or peaks.shape[0] == 0:
        return None # Skip if no peak data is available

    y = torch.zeros(1000, dtype=torch.float)
    # Round m/z values to the nearest integer to use as indices
    mz_indices = np.round(peaks[:, 0]).astype(int)
    intensities = peaks[:, 1]

    # Filter indices to be within the valid range [1, 1000]
    valid_mask = (mz_indices >= 1) & (mz_indices <= 1000)
    mz_indices = mz_indices[valid_mask]
    intensities = intensities[valid_mask]

    # Assign intensities to the dense vector. Indices are 1-based, so subtract 1.
    # Explicitly convert numpy arrays to torch tensors to avoid numpy version issues.
    y[torch.from_numpy(mz_indices - 1)] = torch.from_numpy(intensities)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Add other useful info to the data object
    data.mol_id = spectrum_data.get('id', '')
    data.mw = float(spectrum_data.get('mw', 0.0))

    return data

if __name__ == '__main__':
    # --- Example Usage ---
    # This test combines the parser, loader, and preprocessor.

    # To run this script directly, we need to add the project root to the Python path
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from data.msp_parser import load_all_spectra
    from data.mol_loader import load_mol_files
    import random

    # 1. Load data
    print("Loading all data...")
    spectra_list = load_all_spectra('NIST17.MSP')
    molecules_dict = load_mol_files('mol_files')

    # 2. Create a mapping from spectrum ID to spectrum data
    spectra_dict = {spec['id']: spec for spec in spectra_list if 'id' in spec}

    # 3. Find common IDs between spectra and molecules
    common_ids = list(set(spectra_dict.keys()) & set(molecules_dict.keys()))
    print(f"\nFound {len(common_ids)} common entries between MSP and MOL files.")

    if common_ids:
        # 4. Pick a random sample to preprocess
        sample_id = random.choice(common_ids)
        sample_mol = molecules_dict[sample_id]
        sample_spectrum = spectra_dict[sample_id]

        print(f"\n--- Preprocessing sample ID: {sample_id} ---")
        print(f"Molecule: {Chem.MolToSmiles(sample_mol)}")
        print(f"Spectrum Name: {sample_spectrum.get('name')}")

        # 5. Convert to graph data
        graph_data = mol_to_graph_data(sample_mol, sample_spectrum)

        if graph_data:
            print("\n--- Generated Graph Data ---")
            print(graph_data)
            print(f"Node features shape (x): {graph_data.x.shape}")
            print(f"Edge index shape (edge_index): {graph_data.edge_index.shape}")
            print(f"Edge features shape (edge_attr): {graph_data.edge_attr.shape}")
            print(f"Target vector shape (y): {graph_data.y.shape}")
            print(f"Sum of intensities in y: {graph_data.y.sum():.2f}")
            print(f"Number of non-zero peaks in y: {torch.count_nonzero(graph_data.y)}")
        else:
            print("\nFailed to preprocess the sample.")
    else:
        print("\nNo common data to preprocess. Check file IDs and formats.")