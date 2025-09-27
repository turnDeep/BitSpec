import torch
import os
import random
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset

# Import all custom modules
from utils.config import config
from data.msp_parser import load_all_spectra
from data.mol_loader import load_mol_files
from data.preprocessor import mol_to_graph_data
from models.gcn_encoder import GCNEncoder
from models.bitnet_decoder import BitNetDecoder
from training.loss import WeightedCosineLoss
from training.trainer import Trainer

class BitSpecGCN(torch.nn.Module):
    """A wrapper class that combines the GCNEncoder and BitNetDecoder."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # The preprocessor creates a dense vector of 1000 bins for the target y
        # The decoder predicts (mz, intensity), but for this loss function, we only need intensity.
        self.output_type = 'intensity_vector'

    def forward(self, x, edge_index, batch):
        embedding = self.encoder(x, edge_index, batch)
        _, pred_intensities = self.decoder(embedding)
        return pred_intensities

def load_and_preprocess_data(cfg):
    """
    Loads, preprocesses, and caches the dataset.
    If a cached version exists, it's loaded directly. Otherwise, data is
    processed from scratch and cached for future use.
    """
    cache_dir = cfg['data']['cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'preprocessed_data.pt')

    if os.path.exists(cache_path):
        print(f"Loading preprocessed data from cache: {cache_path}")
        return torch.load(cache_path)

    print("No cache found. Starting preprocessing from scratch...")

    # 1. Load raw data
    spectra_list = load_all_spectra(cfg['data']['msp_path'])
    molecules_dict = load_mol_files(cfg['data']['mol_dir'])
    spectra_dict = {spec['id']: spec for spec in spectra_list if 'id' in spec}

    # 2. Find common entries and preprocess
    common_ids = list(set(spectra_dict.keys()) & set(molecules_dict.keys()))
    print(f"Found {len(common_ids)} common entries between MSP and MOL files.")

    processed_data = []
    for mol_id in common_ids:
        mol = molecules_dict[mol_id]
        spectrum = spectra_dict[mol_id]
        graph_data = mol_to_graph_data(mol, spectrum)

        if graph_data is not None:
            # The target `y` needs to be 2D for the DataLoader to batch correctly
            graph_data.y = graph_data.y.unsqueeze(0)
            processed_data.append(graph_data)

    print(f"Successfully preprocessed {len(processed_data)} data points.")

    # 3. Cache the processed data
    torch.save(processed_data, cache_path)
    print(f"Saved preprocessed data to cache: {cache_path}")

    return processed_data

def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """The main function to run the entire training pipeline."""
    set_seed()

    # 1. Load and preprocess data
    dataset = load_and_preprocess_data(config)
    if not dataset:
        print("No data available to train. Exiting.")
        return

    # 2. Split dataset
    split_ratios = config['data']['split_ratios']
    train_size = int(split_ratios[0] * len(dataset))
    val_size = int(split_ratios[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # 3. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['run']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], num_workers=config['run']['num_workers'])
    # test_loader will be used for final evaluation after training

    # 4. Initialize model and training components
    device_str = config['run']['device']
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = GCNEncoder(**config['model']['encoder'])
    decoder = BitNetDecoder(**config['model']['decoder'])
    model = BitSpecGCN(encoder, decoder)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler_cfg = config['training']['scheduler']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_cfg['t_0'], T_mult=scheduler_cfg['t_mult'], eta_min=scheduler_cfg['eta_min']
    )

    loss_fn = WeightedCosineLoss(max_mz=config['model']['decoder']['output_bins'])

    # 5. Initialize and run Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=os.path.join(config['run']['save_dir'], 'checkpoints')
    )

    trainer.train(num_epochs=config['training']['epochs'])

    print("\nTraining pipeline finished.")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best model saved in '{trainer.save_dir}'")

    # --- Start Evaluation on Test Set ---
    print("\n--- Starting Evaluation on Test Set ---")
    evaluate_on_test_set(model, test_dataset, device, config)


def evaluate_on_test_set(model, test_dataset, device, cfg):
    """Loads the best model and evaluates it on the test set."""

    checkpoint_dir = os.path.join(cfg['run']['save_dir'], 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        print(f"Error: Best model checkpoint not found at {checkpoint_path}.")
        return

    print(f"Loading best model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'])

    all_preds = []
    all_trues = []
    all_mol_ids = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_intensities = model(batch.x, batch.edge_index, batch.batch)
            all_preds.append(pred_intensities.cpu())
            all_trues.append(batch.y.cpu())
            all_mol_ids.extend(batch.mol_id)

    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    # Calculate and print metrics
    from evaluation.metrics import weighted_cosine_similarity, top_n_accuracy

    similarities = weighted_cosine_similarity(all_preds, all_trues)
    mean_similarity = similarities.mean().item()
    print(f"\nAverage Weighted Cosine Similarity: {mean_similarity:.4f}")

    eval_cfg = cfg['evaluation']['top_n_accuracy']
    accuracy = top_n_accuracy(all_preds, all_trues, n=eval_cfg['n'], tolerance=eval_cfg['tolerance'])
    print(f"Top-{eval_cfg['n']} Peak Accuracy (tolerance={eval_cfg['tolerance']}): {accuracy:.4f}")

    # Generate and save plots
    from evaluation.visualizer import plot_spectrum_comparison
    plot_dir = os.path.join(cfg['run']['save_dir'], 'evaluation_plots')
    os.makedirs(plot_dir, exist_ok=True)

    num_plots = min(5, len(test_dataset))
    print(f"\nGenerating {num_plots} comparison plots...")
    for i in range(num_plots):
        idx = random.randint(0, len(all_preds) - 1)
        plot_spectrum_comparison(
            pred_intensities=all_preds[idx],
            true_intensities=all_trues[idx],
            mol_id=all_mol_ids[idx],
            save_path=os.path.join(plot_dir, f'comparison_plot_{all_mol_ids[idx]}.png')
        )
    print(f"Saved plots to '{plot_dir}'.")


if __name__ == '__main__':
    main()