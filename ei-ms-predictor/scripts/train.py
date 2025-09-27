import argparse
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
import sys
from tqdm import tqdm

# Add the project root to the Python path to allow importing from `src`
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.device_manager import DeviceManager
from src.models.graph_transformer import EIMSPredictor
from src.data.mol_processor import MOLProcessor
from src.data.msp_parser import MSPParser
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train the EI-MS Predictor model.")
    parser.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='auto', help='Execution device')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--msp_path', type=str, default='data/NIST17.msp', help="Path to the MSP spectral data file")
    parser.add_argument('--mol_dir', type=str, default='data/mol_files/', help="Directory containing MOL files")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help="Directory to save model checkpoints")
    args = parser.parse_args()

    # Ensure checkpoint directory exists
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # --- Device Setup ---
    device_manager = DeviceManager(force_cpu=(args.device == 'cpu'))
    device = device_manager.get_device()
    print(f"Using device: {device}")

    # --- Model Configuration ---
    config = type('Config', (), {
        'hidden_dim': 256,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.1,
        'max_peaks': 200,
        'max_mz': 1000,
        'max_intensity': 100,
        'atom_list': ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
    })()

    # --- Data Preparation ---
    mol_processor = MOLProcessor()

    # The spec implies a custom Dataset. For now, we'll create the data list directly.
    # A full implementation should use the EIMSDataset class in dataloader.py
    print("Processing MOL files and matching with spectra...")

    # 1. Parse spectra (placeholder, as parser is not fully implemented)
    msp_parser = MSPParser(args.msp_path)
    # In a real implementation, you'd get a list of spectra:
    # all_spectra = msp_parser.parse()
    # and a mapping:
    # cas_to_spectrum = {s['cas']: s for s in all_spectra}

    # 2. Process MOL files and create a dataset
    mol_files = list(Path(args.mol_dir).glob('*.MOL'))
    dataset = []
    for mol_file in tqdm(mol_files, desc="Processing MOL files"):
        try:
            graph_data = mol_processor.process_mol_file(str(mol_file))

            # Placeholder for matching graph_data to its spectrum
            # This requires an identifier in the MOL file (e.g., CAS number in the header)
            # that can be used to look up the spectrum in the MSP data.
            # For now, we'll just add dummy target data.
            graph_data.y_mz = torch.randn(config.max_peaks)
            graph_data.y_intensity = torch.randn(config.max_peaks)

            dataset.append(graph_data)
        except Exception as e:
            print(f"Skipping {mol_file.name} due to error: {e}")

    if not dataset:
        print("No data could be loaded. Exiting.")
        return

    # 3. Split data and create DataLoaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Model, Trainer, and Training ---
    model = EIMSPredictor(config).to(device)

    # The spec mentions torch.compile, which is a great optimization for PyTorch 2.x
    try:
        model = torch.compile(model)
        print("Model compiled successfully (PyTorch 2.x).")
    except Exception:
        print("Could not compile model (requires PyTorch 2.x).")

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        epochs=args.epochs
    )

    trainer.train()

    # --- Save Final Model ---
    final_model_path = Path(args.checkpoint_dir) / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
