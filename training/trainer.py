import torch
import torch.nn as nn
import os
import tqdm
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    """
    A class to encapsulate the training and evaluation loop for the BitSpec-GCN model.
    It includes support for mixed-precision training, model checkpointing, and progress logging.
    """
    def __init__(self, model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, save_dir='checkpoints'):
        """
        Args:
            model (nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler: The learning rate scheduler.
            loss_fn: The loss function.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            device (torch.device): The device to run training on (e.g., 'cuda' or 'cpu').
            save_dir (str): Directory to save model checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda')) # Mixed precision scaler
        self.best_val_loss = float('inf')

        self.model.to(self.device)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Trainer initialized. Checkpoints will be saved in '{self.save_dir}'")

    def _run_batch(self, batch, is_training):
        """Processes a single batch of data, for either training or evaluation."""
        batch = batch.to(self.device)

        # autocast enables mixed-precision computation
        with autocast(enabled=(self.device.type == 'cuda')):
            pred_intensities = self.model(batch.x, batch.edge_index, batch.batch)
            true_intensities = batch.y
            loss = self.loss_fn(pred_intensities, true_intensities)

        if is_training:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss.item()

    def train_epoch(self):
        """Runs one full epoch of training."""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm.tqdm(self.train_loader, desc="Training", leave=False, unit="batch")
        for batch in progress_bar:
            loss = self._run_batch(batch, is_training=True)
            total_loss += loss
            progress_bar.set_postfix(loss=f'{loss:.4f}')

        return total_loss / len(self.train_loader)

    def evaluate(self):
        """Runs one full epoch of validation."""
        self.model.eval()
        total_loss = 0

        progress_bar = tqdm.tqdm(self.val_loader, desc="Validating", leave=False, unit="batch")
        with torch.no_grad():
            for batch in progress_bar:
                loss = self._run_batch(batch, is_training=False)
                total_loss += loss
                progress_bar.set_postfix(loss=f'{loss:.4f}')

        return total_loss / len(self.val_loader)

    def train(self, num_epochs):
        """The main training loop."""
        print(f"Starting training for {num_epochs} epochs on device: {self.device}")
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, 'best_model.pth')
                print(f"  -> New best model saved with validation loss: {val_loss:.4f}")

        self._save_checkpoint(num_epochs, 'last_model.pth')
        print("Training finished.")

    def _save_checkpoint(self, epoch, filename):
        """Saves a model checkpoint."""
        filepath = os.path.join(self.save_dir, filename)
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(state, filepath)


if __name__ == '__main__':
    # --- Example Usage & Test ---
    print("--- Testing Trainer Class ---")

    # To run this script directly, we need to add the project root to the Python path
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from models.gcn_encoder import GCNEncoder
    from models.bitnet_decoder import BitNetDecoder
    from training.loss import WeightedCosineLoss
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data

    # 1. Define a wrapper model that combines Encoder and Decoder
    class BitSpecGCN(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x, edge_index, batch):
            embedding = self.encoder(x, edge_index, batch)
            # The decoder returns (mz, intensity), but our loss only needs intensity.
            _, pred_intensities = self.decoder(embedding)
            return pred_intensities

    # 2. Set up dummy data and parameters
    input_dim = 10
    hidden_dim = 32
    output_dim = 64
    num_bins = 100
    batch_size = 8

    # Create a list of dummy graphs
    data_list = []
    for _ in range(32):
        num_nodes = torch.randint(5, 15, (1,)).item()
        x = torch.randn(num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), dtype=torch.long)
        y = torch.rand(num_bins) * 100 # Target intensity vector
        # y needs to be 2D [1, num_bins] so the DataLoader batches it correctly to [batch_size, num_bins]
        data_list.append(Data(x=x, edge_index=edge_index, y=y.unsqueeze(0)))

    # Create DataLoaders
    train_loader = DataLoader(data_list[:24], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[24:], batch_size=batch_size)

    # 3. Instantiate all components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = GCNEncoder(input_dim, hidden_dim, output_dim)
    decoder = BitNetDecoder(output_dim, hidden_dim, num_bins)
    model = BitSpecGCN(encoder, decoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = WeightedCosineLoss(max_mz=num_bins)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # 4. Instantiate and run the Trainer
    trainer = Trainer(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device)

    # Run for just one epoch to test the whole pipeline
    trainer.train(num_epochs=1)

    # 5. Check if checkpoints were created
    assert os.path.exists(os.path.join('checkpoints', 'best_model.pth')), "Best model checkpoint was not created."
    assert os.path.exists(os.path.join('checkpoints', 'last_model.pth')), "Last model checkpoint was not created."
    print("\nCheckpoints created successfully.")

    # Clean up created directory
    import shutil
    shutil.rmtree('checkpoints')
    print("Cleaned up checkpoints directory.")

    print("\n--- Trainer class test completed successfully! ---")