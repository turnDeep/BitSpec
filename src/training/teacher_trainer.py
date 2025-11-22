#!/usr/bin/env python3
# src/training/teacher_trainer.py
"""
NEIMS v2.0 Teacher Trainer

Handles Teacher model training for both pretraining (Bond Masking)
and finetuning (Spectrum Prediction with MC Dropout).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from src.training.losses import TeacherLoss
from src.training.schedulers import WarmupCosineScheduler


class TeacherTrainer:
    """
    Teacher Model Trainer

    Supports:
    - Phase 1: Pretraining on PCQM4Mv2 (Bond Masking)
    - Phase 2: Finetuning on NIST EI-MS (MC Dropout)
    """

    def __init__(
        self,
        model,
        config: Dict,
        device: str = 'cuda',
        phase: str = 'pretrain'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.phase = phase

        # Get phase-specific config
        if phase == 'pretrain':
            self.train_config = config['training']['teacher_pretrain']
        else:  # finetune
            self.train_config = config['training']['teacher_finetune']

        # Loss function
        lambda_bond = self.train_config.get('bond_masking', {}).get('lambda_bond', 0.1) \
                      if phase == 'pretrain' else 0.0
        self.criterion = TeacherLoss(lambda_bond=lambda_bond)

        # Optimizer
        self.optimizer = self._setup_optimizer()

        # Scheduler
        self.scheduler = self._setup_scheduler()

        # Mixed precision training
        self.use_amp = self.train_config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.gradient_clip = self.train_config.get('gradient_clip', 1.0)

        # Logging
        self.logger = logging.getLogger(__name__)

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def _setup_optimizer(self):
        """Setup optimizer"""
        optimizer_name = self.train_config.get('optimizer', 'AdamW')
        lr = self.train_config.get('learning_rate', 1e-4)
        weight_decay = self.train_config.get('weight_decay', 1e-5)

        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_name = self.train_config.get('scheduler', 'CosineAnnealingWarmRestarts')

        if scheduler_name == 'CosineAnnealingWarmRestarts':
            T_0 = self.train_config.get('scheduler_t0', 10)
            T_mult = self.train_config.get('scheduler_tmult', 2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=1e-6
            )
        elif scheduler_name == 'WarmupCosine':
            warmup_epochs = self.train_config.get('warmup_epochs', 5)
            max_epochs = self.train_config.get('num_epochs', 100)
            scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch

        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_spectrum_loss = 0.0
        total_bond_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Move batch to device
            graph_data = batch['graph'].to(self.device)
            ecfp = batch['ecfp'].to(self.device)

            # Handle different phases
            bond_predictions = None
            bond_targets = None

            if self.phase == 'pretrain':
                # Pretraining: Use dummy spectrum (all zeros) since we focus on bond masking
                batch_size = ecfp.size(0)
                target_spectrum = torch.zeros((batch_size, 501), device=self.device)

                # Bond targets for pretraining
                if 'mask_targets' in batch:
                    bond_targets = batch['mask_targets'].to(self.device)
            else:
                # Finetuning: Use actual spectrum
                target_spectrum = batch['spectrum'].to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                # Model forward with bond predictions for pretraining
                if self.phase == 'pretrain' and bond_targets is not None:
                    model_output = self.model(graph_data, ecfp, dropout=True, return_bond_predictions=True)
                    if isinstance(model_output, tuple):
                        predicted_spectrum, bond_predictions = model_output
                    else:
                        predicted_spectrum = model_output
                else:
                    predicted_spectrum = self.model(graph_data, ecfp, dropout=True)

                # Compute loss
                loss, loss_dict = self.criterion(
                    predicted_spectrum,
                    target_spectrum,
                    bond_predictions,
                    bond_targets
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                self.optimizer.step()

            # Update metrics
            total_loss += loss_dict['total_loss']
            total_spectrum_loss += loss_dict['spectrum_loss']
            if 'bond_loss' in loss_dict:
                total_bond_loss += loss_dict['bond_loss']
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'spec': f"{loss_dict['spectrum_loss']:.4f}"
            })

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # Compute averages
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_spectrum_loss': total_spectrum_loss / num_batches,
        }

        if self.phase == 'pretrain':
            metrics['train_bond_loss'] = total_bond_loss / num_batches

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        use_mc_dropout: bool = False
    ) -> Dict[str, float]:
        """
        Validate model

        Args:
            val_loader: Validation data loader
            use_mc_dropout: Whether to use MC Dropout for uncertainty

        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_spectrum_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            graph_data = batch['graph'].to(self.device)
            ecfp = batch['ecfp'].to(self.device)

            # Handle different phases
            if self.phase == 'pretrain':
                # Pretraining: Use dummy spectrum
                batch_size = ecfp.size(0)
                target_spectrum = torch.zeros((batch_size, 501), device=self.device)
            else:
                # Finetuning: Use actual spectrum
                target_spectrum = batch['spectrum'].to(self.device)

            with autocast(enabled=self.use_amp):
                if use_mc_dropout and self.phase == 'finetune':
                    # MC Dropout uncertainty estimation
                    predicted_spectrum, uncertainty = self.model.predict_with_uncertainty(
                        graph_data,
                        ecfp
                    )
                else:
                    # Standard forward pass
                    predicted_spectrum = self.model(graph_data, ecfp, dropout=False)

                # Compute loss
                loss, loss_dict = self.criterion(
                    predicted_spectrum,
                    target_spectrum,
                    None,
                    None
                )

            total_loss += loss_dict['total_loss']
            total_spectrum_loss += loss_dict['spectrum_loss']
            num_batches += 1

        metrics = {
            'val_loss': total_loss / num_batches,
            'val_spectrum_loss': total_spectrum_loss / num_batches,
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str,
        save_interval: int = 10
    ):
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N epochs
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting {self.phase} training for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            use_mc = (self.phase == 'finetune')
            val_metrics = self.validate(val_loader, use_mc_dropout=use_mc)

            # Log metrics
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )

            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.save_checkpoint(
                    checkpoint_dir / f'best_{self.phase}_teacher.pt',
                    epoch,
                    val_metrics
                )
                self.logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f})")

            # Periodic checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(
                    checkpoint_dir / f'{self.phase}_teacher_epoch_{epoch}.pt',
                    epoch,
                    val_metrics
                )

        self.logger.info(
            f"Training complete. Best epoch: {self.best_epoch}, "
            f"Best val loss: {self.best_val_loss:.4f}"
        )

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict
    ):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'phase': self.phase
        }, path)

        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"Checkpoint loaded: {path}")

        return checkpoint['epoch'], checkpoint['metrics']
