import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import tqdm

def load_mol_files(mol_dir):
    """
    Loads all MOL files from a directory into a dictionary of RDKit molecules.

    Args:
        mol_dir (str): Path to the directory containing MOL files.

    Returns:
        dict: A dictionary mapping the molecule ID (from filename) to the
              RDKit molecule object.
              Example: {'200001': <rdkit.Chem.rdchem.Mol object>, ...}
    """
    molecules = {}
    if not os.path.isdir(mol_dir):
        print(f"Error: Directory not found at {mol_dir}")
        return molecules

    mol_files = [f for f in os.listdir(mol_dir) if f.upper().endswith('.MOL')]

    print(f"Loading molecule files from {mol_dir}...")
    for filename in tqdm.tqdm(mol_files, desc="Loading MOL files"):
        filepath = os.path.join(mol_dir, filename)

        # Extract ID from filename like 'ID200001.MOL'
        try:
            # Assumes format IDXXXXXX.MOL
            mol_id = filename[2:-4]
            if not mol_id.isdigit():
                 # Fallback for other potential naming conventions
                 mol_id = os.path.splitext(filename)[0]
        except IndexError:
            print(f"Warning: Could not parse ID from filename '{filename}'. Skipping.")
            continue

        try:
            # removeHs=False to keep hydrogen atoms, which might be important for some features.
            mol = Chem.MolFromMolFile(filepath, removeHs=False)
            if mol is not None:
                molecules[mol_id] = mol
            else:
                print(f"Warning: Failed to load molecule from '{filename}'. RDKit returned None.")
        except Exception as e:
            print(f"Warning: Error reading file '{filename}': {e}. Skipping.")

    print(f"Successfully loaded {len(molecules)} molecules.")
    return molecules

if __name__ == '__main__':
    # Example usage:
    # This script can be run directly to test the MOL loader.
    mol_directory = 'mol_files'

    if os.path.exists(mol_directory):
        loaded_molecules = load_mol_files(mol_directory)

        if loaded_molecules:
            print(f"\nTotal molecules loaded: {len(loaded_molecules)}")

            # Print details of a few loaded molecules as a sample
            sample_ids = list(loaded_molecules.keys())[:3]
            for mol_id in sample_ids:
                mol = loaded_molecules[mol_id]
                print(f"\n--- Molecule ID: {mol_id} ---")
                print(f"  Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
                print(f"  Num Atoms: {mol.GetNumAtoms()}")
                print(f"  Mol Weight: {rdMolDescriptors.CalcExactMolWt(mol)}")
        else:
            print("No molecules were loaded. The directory might be empty or files are corrupted.")
    else:
        print(f"Error: The directory '{mol_directory}' was not found.")
        print("Please ensure the script is run from the project root and the mol_files directory is present.")