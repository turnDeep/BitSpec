import argparse
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
import sys
import json
from tqdm import tqdm
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.device_manager import DeviceManager
from src.models.graph_transformer import EIMSPredictor
from src.data.mol_processor import MOLProcessor
from src.evaluation.metrics import EvaluationMetrics

def evaluate(model, dataloader, device):
    """
    Runs the evaluation loop.
    """
    model.eval()
    all_metrics = []
    metrics_calculator = EvaluationMetrics()

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            data = data.to(device)

            pred_mz, pred_intensity = model(data)

            # The dataloader should provide true spectra (y) for evaluation.
            # This is a placeholder, assuming data.y_mz and data.y_intensity exist.
            true_mz = data.y_mz
            true_intensity = data.y_intensity

            # Calculate metrics for each item in the batch
            for i in range(data.num_graphs):
                metrics = metrics_calculator.calculate_all(
                    pred_mz[i], pred_intensity[i],
                    true_mz[i], true_intensity[i]
                )
                metrics['mol_file'] = data.path[i] if hasattr(data, 'path') else 'N/A'
                all_metrics.append(metrics)

    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained EI-MS Predictor model.")
    parser.add_argument('--model_path', type=str, default='checkpoints/final_model.pth', help="Path to the trained model checkpoint.")
    parser.add_argument('--mol_dir', type=str, default='data/mol_files/', help="Directory containing MOL files for the test set.")
    # Add other args for test data selection if necessary
    parser.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='auto', help="Execution device.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--output_csv', type=str, default='results/evaluation_summary.csv', help="Path to save the evaluation results CSV.")
    args = parser.parse_args()

    # --- Device and Model Setup ---
    device_manager = DeviceManager(force_cpu=(args.device == 'cpu'))
    device = device_manager.get_device()
    config = type('Config', (), {
        'hidden_dim': 256, 'n_heads': 8, 'n_layers': 6, 'dropout': 0.1,
        'max_peaks': 200, 'max_mz': 1000, 'max_intensity': 100,
        'atom_list': ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
    })()
    model = EIMSPredictor(config)

    # Load the state dict, cleaning keys if the model was compiled
    try:
        state_dict = torch.load(args.model_path, map_location=device)
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
        print(f"Error: Model checkpoint not found at {args.model_path}")
        sys.exit(1)

    model.to(device)

    # --- Data Loading (Placeholder) ---
    # This should load a pre-defined test/validation set.
    # For now, we re-use the logic from train.py to load all molecules.
    mol_processor = MOLProcessor()
    mol_files = list(Path(args.mol_dir).glob('*.MOL'))
    dataset = []
    for mol_file in mol_files:
        try:
            graph_data = mol_processor.process_mol_file(str(mol_file))
            graph_data.y_mz = torch.rand(config.max_peaks) * 1000 # Dummy true data
            graph_data.y_intensity = torch.rand(config.max_peaks) * 100 # Dummy true data
            dataset.append(graph_data)
        except Exception as e:
            print(f"Skipping {mol_file.name}: {e}")

    if not dataset:
        print("No data to evaluate. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Run Evaluation ---
    evaluation_results = evaluate(model, dataloader, device)

    # --- Summarize and Save Results ---
    if not evaluation_results:
        print("Evaluation did not produce any results.")
        return

    df = pd.DataFrame(evaluation_results)

    # Save detailed results
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Detailed evaluation results saved to {output_path}")

    # Print summary
    avg_cosine = df['cosine_similarity'].mean()
    print("\n--- Evaluation Summary ---")
    print(f"Average Cosine Similarity: {avg_cosine:.4f}")
    print("--------------------------\n")

if __name__ == "__main__":
    main()
