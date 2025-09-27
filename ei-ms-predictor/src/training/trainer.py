import torch
from tqdm import tqdm
from .losses import HybridLoss
from .optimizer import create_optimizer, create_scheduler

class Trainer:
    """
    Handles the model training and validation loops.
    """
    def __init__(self, model, device, train_loader, val_loader, learning_rate, epochs):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = learning_rate
        self.epochs = epochs

        self.optimizer = create_optimizer(model, lr=self.lr)
        self.scheduler = create_scheduler(self.optimizer) # Using default scheduler params
        self.criterion = HybridLoss()

    def train(self):
        """
        The main training loop over all epochs.
        """
        print("Starting training...")
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.validate_one_epoch(epoch)

            # Placeholder for saving the best model
            # torch.save(self.model.state_dict(), 'best_model.pth')

        print("Training finished.")

    def train_one_epoch(self, epoch_num):
        """
        Runs a single epoch of training.
        """
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_num+1}/{self.epochs} [T]")
        for data in progress_bar:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            pred_mz, pred_intensity = self.model(data)

            # Loss calculation
            # The loss function requires true spectra, which should be on the `data` object.
            # This part depends on the dataloader correctly attaching `y` (true spectra).
            # e.g., loss = self.criterion(pred_mz, pred_intensity, data.y_mz, data.y_intensity)

            # Placeholder loss
            loss = torch.tensor(0.0, requires_grad=True) # Replace with real loss

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Gradient clipping
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch_num+1} Training - Average Loss: {avg_loss:.4f}")

    def validate_one_epoch(self, epoch_num):
        """
        Runs a single epoch of validation.
        """
        self.model.eval()
        total_loss = 0

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch_num+1}/{self.epochs} [V]")
        with torch.no_grad():
            for data in progress_bar:
                data = data.to(self.device)

                pred_mz, pred_intensity = self.model(data)

                # Placeholder loss
                loss = torch.tensor(0.0) # Replace with real loss

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {epoch_num+1} Validation - Average Loss: {avg_loss:.4f}")
