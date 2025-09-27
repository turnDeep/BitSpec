import argparse
import torch
from pathlib import Path
import json
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.device_manager import DeviceManager
from src.models.graph_transformer import EIMSPredictor
from src.data.mol_processor import MOLProcessor
from torch_geometric.data import Batch

def predict(mol_file, model_path, device_choice='auto'):
    """
    Predicts the mass spectrum for a single MOL file.
    """
    # --- Device Setup ---
    device_manager = DeviceManager(force_cpu=(device_choice == 'cpu'))
    device = device_manager.get_device()
    print(f"Using device: {device}")

    # --- Model Configuration (should match training) ---
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

    # --- Load Model ---
    model = EIMSPredictor(config)
    try:
        # Load the state dict, which might be from a compiled model
        state_dict = torch.load(model_path, map_location=device)

        # If the model was compiled, the keys will be prefixed. We need to remove the prefix.
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                name = k[10:] # remove `_orig_mod.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)

    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
    model.to(device)
    model.eval()

    # Inference optimization with torch.compile
    try:
        model = torch.compile(model)
        print("Model compiled for inference.")
    except Exception:
        print("Could not compile model for inference.")

    # --- Process Input Molecule ---
    print(f"Processing molecule: {mol_file}")
    mol_processor = MOLProcessor()
    try:
        mol_data = mol_processor.process_mol_file(mol_file)
    except Exception as e:
        print(f"Error processing MOL file: {e}")
        sys.exit(1)

    # Create a batch for the single molecule
    mol_batch = Batch.from_data_list([mol_data]).to(device)

    # --- Prediction ---
    with torch.no_grad():
        mz_values, intensities = model(mol_batch)

    # --- Format Output ---
    # Results are on the device, move to CPU and numpy for post-processing
    mz_values = mz_values.cpu().numpy().flatten()
    intensities = intensities.cpu().numpy().flatten()

    # Filter out low-intensity peaks and create a list of {'mz': val, 'intensity': val}
    # Also sort by m/z value for clean output
    spectrum_peaks = []
    for mz, intensity in zip(mz_values, intensities):
        if intensity > 1.0: # Intensity threshold
            spectrum_peaks.append({'mz': round(float(mz), 4), 'intensity': round(float(intensity), 4)})

    # Sort by m/z
    spectrum_peaks.sort(key=lambda p: p['mz'])

    result = {
        'mol_file': Path(mol_file).name,
        'predicted_spectrum': spectrum_peaks
    }

    return result

def main():
    parser = argparse.ArgumentParser(description="Predict a mass spectrum from a MOL file.")
    parser.add_argument('--mol_file', type=str, required=True, help="Path to the input MOL file.")
    parser.add_argument('--model_path', type=str, default='checkpoints/final_model.pth', help="Path to the trained model checkpoint.")
    parser.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='auto', help="Execution device.")
    parser.add_argument('--output', type=str, default='prediction.json', help="Path to save the output JSON file.")
    args = parser.parse_args()

    # Run prediction
    prediction_result = predict(args.mol_file, args.model_path, args.device)

    # Save result to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(prediction_result, f, indent=2)

    print(f"Prediction saved to {output_path}")
    print(f"Predicted {len(prediction_result['predicted_spectrum'])} peaks.")

if __name__ == "__main__":
    main()
