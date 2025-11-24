#!/usr/bin/env python3
# src/training/student_trainer.py
"""
NEIMS v2.0 Student Trainer

Handles Student model training with knowledge distillation from Teacher.
Implements uncertainty-aware KD with GradNorm adaptive weighting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Optional

from src.training.losses import StudentDistillationLoss, GradNormWeighting
from src.training.schedulers import TemperatureScheduler
from src.models.modules import FeatureProjection
from src.models.moe import load_balancing_loss, entropy_regularization


class StudentTrainer:
    """
    Student Model Trainer with Knowledge Distillation

    Implements NEIMS v2.0 complete distillation strategy:
    - Uncertainty-aware soft labels (MC Dropout)
    - Multi-objective loss (Hard + Soft + Feature + Load + Entropy)
    - GradNorm adaptive weighting
    - Temperature annealing
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        config: Dict,
        device: str = 'cuda'
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.teacher.eval()  # Teacher in eval mode

        self.config = config
        self.device = device

        # Get distillation config
        self.train_config = config['training']['student_distill']
        self.distill_config = self.train_config['distillation']

        # Loss function
        self.criterion = StudentDistillationLoss(
            alpha_init=self.distill_config['alpha_init'],
            beta_init=self.distill_config['beta_init'],
            gamma_init=self.distill_config['gamma_init'],
            delta_load=self.distill_config['delta_load'],
            delta_entropy=self.distill_config['delta_entropy'],
            use_lds=self.distill_config.get('use_lds', True),
            lds_sigma=self.distill_config.get('lds_sigma', 1.5)
        )

        # Feature projection for KD
        self.feature_projection = FeatureProjection(
            student_dim=config['model']['student']['input_dim'],
            teacher_dim=config['model']['teacher']['ecfp']['mlp_output']
        ).to(device)

        # Temperature scheduler
        self.temp_scheduler = TemperatureScheduler(
            T_init=self.distill_config['temperature_init'],
            T_min=self.distill_config['temperature_min'],
            max_epochs=self.train_config['num_epochs'],
            schedule=self.distill_config['temperature_schedule']
        )

        # GradNorm (after warmup)
        self.use_gradnorm = self.distill_config.get('use_gradnorm', True)
        self.warmup_epochs = self.distill_config.get('warmup_epochs', 15)
        self.gradnorm = None  # Initialize after first epoch

        # Optimizer
        self.optimizer = self._setup_optimizer()

        # Scheduler
        self.scheduler = self._setup_scheduler()

        # Mixed precision
        self.use_amp = self.train_config.get('use_amp', True)
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Gradient clipping
        self.gradient_clip = self.train_config.get('gradient_clip', 0.5)

        # Logging
        self.logger = logging.getLogger(__name__)

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # Track initial losses for GradNorm
        self.initial_losses = None

    def _setup_optimizer(self):
        """Setup optimizer"""
        lr = self.train_config.get('learning_rate', 5e-4)
        weight_decay = self.train_config.get('weight_decay', 1e-4)

        # Separate parameter groups for student and feature projection
        params = [
            {'params': self.student.parameters()},
            {'params': self.feature_projection.parameters()}
        ]

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        return optimizer

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_name = self.train_config.get('scheduler', 'OneCycleLR')

        if scheduler_name == 'OneCycleLR':
            max_lr = self.train_config.get('max_lr', 1e-3)
            num_epochs = self.train_config['num_epochs']
            # Estimate steps per epoch (will update later)
            steps_per_epoch = 1000  # Placeholder

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.train_config.get('pct_start', 0.1)
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with knowledge distillation"""
        self.student.train()
        self.teacher.eval()

        # Get temperature for this epoch
        temperature = self.temp_scheduler.get_temperature(epoch - 1)

        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        total_feature_loss = 0.0
        total_load_loss = 0.0
        total_entropy_loss = 0.0
        num_batches = 0

        # Initialize GradNorm if needed (and not already initialized)
        if self.use_gradnorm and epoch >= self.warmup_epochs and self.gradnorm is None:
             # We need initial losses. If we are resuming or starting late, we might not have them.
             # We will rely on the first batch of this epoch to set them if they are missing.
             pass

        # Expert usage tracking
        expert_counts = torch.zeros(4, device=self.device)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} (T={temperature:.2f})")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            ecfp_count_fp = batch['ecfp_count_fp'].to(self.device)
            nist_spectrum = batch['spectrum'].to(self.device)
            graph_data = batch.get('graph')  # For teacher
            ecfp = batch.get('ecfp')  # For teacher

            # Teacher prediction with uncertainty (no gradients)
            with torch.no_grad():
                if graph_data is not None:
                    graph_data = graph_data.to(self.device)
                    ecfp = ecfp.to(self.device)

                    teacher_mean, teacher_std = self.teacher.predict_with_uncertainty(
                        graph_data,
                        ecfp
                    )
                    teacher_features = self.teacher.get_ecfp_embedding(ecfp)
                else:
                    # Fallback: use NIST as teacher target
                    teacher_mean = nist_spectrum
                    teacher_std = torch.zeros_like(nist_spectrum)
                    teacher_features = torch.zeros(
                        nist_spectrum.size(0),
                        512,
                        device=self.device
                    )

            # Student forward pass
            with autocast('cuda', enabled=self.use_amp):
                # Student prediction
                student_output, expert_weights, expert_indices = self.student(
                    ecfp_count_fp
                )

                # Student features (for feature distillation)
                student_features_raw = self.student.get_hidden_features(ecfp_count_fp)
                student_features = self.feature_projection(student_features_raw)

                # Compute distillation loss components explicitly for GradNorm
                loss_hard_tensor = torch.nn.functional.mse_loss(student_output, nist_spectrum)

                loss_soft_tensor = self.criterion._compute_soft_loss(
                    student_output,
                    teacher_mean,
                    teacher_std,
                    temperature
                )

                loss_feature_tensor = torch.nn.functional.mse_loss(student_features, teacher_features)

                loss_load_tensor = load_balancing_loss(expert_weights, expert_indices)
                loss_entropy_tensor = entropy_regularization(expert_weights)

                # Total Loss
                loss = (
                    self.criterion.alpha * loss_hard_tensor +
                    self.criterion.beta * loss_soft_tensor +
                    self.criterion.gamma * loss_feature_tensor +
                    self.criterion.delta_load * loss_load_tensor +
                    self.criterion.delta_entropy * loss_entropy_tensor
                )

                loss_dict = {
                    'total_loss': loss.item(),
                    'hard_loss': loss_hard_tensor.item(),
                    'soft_loss': loss_soft_tensor.item(),
                    'feature_loss': loss_feature_tensor.item(),
                    'load_loss': loss_load_tensor.item(),
                    'entropy_loss': loss_entropy_tensor.item(),
                    'alpha': self.criterion.alpha,
                    'beta': self.criterion.beta,
                    'gamma': self.criterion.gamma
                }

            # ✅ NaN/Inf check BEFORE backward() to prevent weight corruption
            if (torch.isnan(loss) or torch.isinf(loss) or
                torch.isnan(loss_hard_tensor) or torch.isinf(loss_hard_tensor) or
                torch.isnan(loss_soft_tensor) or torch.isinf(loss_soft_tensor) or
                torch.isnan(loss_feature_tensor) or torch.isinf(loss_feature_tensor)):
                self.logger.warning(f"⚠️ NaN or Inf detected BEFORE backward at epoch {epoch}, batch {batch_idx}")
                self.logger.warning(f"Loss components: {loss_dict}")
                self.logger.warning(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
                self.logger.warning("Skipping backward/step to preserve model weights...")
                continue  # Skip this batch entirely (no backward, no step)

            # Backward pass
            self.optimizer.zero_grad()

            # Define shared parameters for GradNorm (Gate + Experts)
            # This targets the main body of the Student model.
            shared_params = list(self.student.gate.parameters()) + list(self.student.experts.parameters())

            if self.use_amp:
                self.scaler.scale(loss).backward(retain_graph=self.use_gradnorm)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.student.parameters()) + list(self.feature_projection.parameters()),
                    self.gradient_clip
                )

                # GradNorm Update
                if self.use_gradnorm and epoch >= self.warmup_epochs:
                     if self.initial_losses is None:
                         self.initial_losses = {
                             'hard': loss_dict['hard_loss'],
                             'soft': loss_dict['soft_loss'],
                             'feature': loss_dict['feature_loss']
                         }

                     if self.gradnorm is None:
                        self.gradnorm = GradNormWeighting(
                            self.initial_losses,
                            alpha=1.5,
                            gradient_clip_range=tuple(self.distill_config['gradient_clip_range'])
                        )

                     current_loss_tensors = {
                         'hard': loss_hard_tensor,
                         'soft': loss_soft_tensor,
                         'feature': loss_feature_tensor
                     }

                     current_weights = {
                         'alpha': self.criterion.alpha,
                         'beta': self.criterion.beta,
                         'gamma': self.criterion.gamma
                     }

                     new_weights = self.gradnorm.compute_weights(
                         current_loss_tensors,
                         current_weights,
                         shared_params
                     )

                     self.criterion.update_weights(
                         new_weights['alpha'],
                         new_weights['beta'],
                         new_weights['gamma']
                     )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward(retain_graph=self.use_gradnorm)
                torch.nn.utils.clip_grad_norm_(
                    list(self.student.parameters()) + list(self.feature_projection.parameters()),
                    self.gradient_clip
                )

                # GradNorm Update
                if self.use_gradnorm and epoch >= self.warmup_epochs:
                     if self.initial_losses is None:
                         self.initial_losses = {
                             'hard': loss_dict['hard_loss'],
                             'soft': loss_dict['soft_loss'],
                             'feature': loss_dict['feature_loss']
                         }

                     if self.gradnorm is None:
                        self.gradnorm = GradNormWeighting(
                            self.initial_losses,
                            alpha=1.5,
                            gradient_clip_range=tuple(self.distill_config['gradient_clip_range'])
                        )

                     current_loss_tensors = {
                         'hard': loss_hard_tensor,
                         'soft': loss_soft_tensor,
                         'feature': loss_feature_tensor
                     }

                     current_weights = {
                         'alpha': self.criterion.alpha,
                         'beta': self.criterion.beta,
                         'gamma': self.criterion.gamma
                     }

                     new_weights = self.gradnorm.compute_weights(
                         current_loss_tensors,
                         current_weights,
                         shared_params
                     )

                     self.criterion.update_weights(
                         new_weights['alpha'],
                         new_weights['beta'],
                         new_weights['gamma']
                     )

                self.optimizer.step()

            # Update expert bias (auxiliary-loss-free load balancing)
            expert_usage = torch.bincount(
                expert_indices.flatten(),
                minlength=4
            ).float()
            expert_counts += expert_usage
            self.student.update_expert_bias(expert_usage)

            # Scheduler step (if OneCycleLR)
            if self.scheduler and isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()

            # Update metrics (NaN check already done before backward)
            total_loss += loss_dict['total_loss']
            total_hard_loss += loss_dict['hard_loss']
            total_soft_loss += loss_dict['soft_loss']
            total_feature_loss += loss_dict['feature_loss']
            total_load_loss += loss_dict['load_loss']
            total_entropy_loss += loss_dict['entropy_loss']
            num_batches += 1

            # Store initial losses for GradNorm (if not set yet)
            if self.initial_losses is None and batch_idx == 0:
                self.initial_losses = {
                    'hard': loss_dict['hard_loss'],
                    'soft': loss_dict['soft_loss'],
                    'feature': loss_dict['feature_loss']
                }

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'α': f"{loss_dict['alpha']:.2f}",
                'β': f"{loss_dict['beta']:.2f}",
                'γ': f"{loss_dict['gamma']:.2f}"
            })

        # Compute averages
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_hard_loss': total_hard_loss / num_batches,
            'train_soft_loss': total_soft_loss / num_batches,
            'train_feature_loss': total_feature_loss / num_batches,
            'train_load_loss': total_load_loss / num_batches,
            'train_entropy_loss': total_entropy_loss / num_batches,
            'temperature': temperature,
            'alpha': self.criterion.alpha,
            'beta': self.criterion.beta,
            'gamma': self.criterion.gamma,
            'expert_usage': expert_counts.cpu().numpy() / expert_counts.sum().item()
        }

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate student model"""
        self.student.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            ecfp_count_fp = batch['ecfp_count_fp'].to(self.device)
            nist_spectrum = batch['spectrum'].to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                student_output, expert_weights, expert_indices = self.student(
                    ecfp_count_fp
                )

                # Simple MSE loss for validation
                loss = torch.nn.functional.mse_loss(student_output, nist_spectrum)

            total_loss += loss.item()
            num_batches += 1

        metrics = {
            'val_loss': total_loss / num_batches
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str,
        save_interval: int = 5
    ):
        """Full training loop"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting Student distillation for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Log metrics
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"T: {train_metrics['temperature']:.2f}, "
                f"Expert Usage: {train_metrics['expert_usage']}"
            )

            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.save_checkpoint(
                    checkpoint_dir / 'best_student.pt',
                    epoch,
                    val_metrics
                )
                self.logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f})")

            # Periodic checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(
                    checkpoint_dir / f'student_epoch_{epoch}.pt',
                    epoch,
                    val_metrics
                )

        self.logger.info(
            f"Training complete. Best epoch: {self.best_epoch}, "
            f"Best val loss: {self.best_val_loss:.4f}"
        )

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'feature_projection_state_dict': self.feature_projection.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'loss_weights': {
                'alpha': self.criterion.alpha,
                'beta': self.criterion.beta,
                'gamma': self.criterion.gamma
            }
        }, path)

        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.feature_projection.load_state_dict(checkpoint['feature_projection_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore loss weights
        if 'loss_weights' in checkpoint:
            self.criterion.alpha = checkpoint['loss_weights']['alpha']
            self.criterion.beta = checkpoint['loss_weights']['beta']
            self.criterion.gamma = checkpoint['loss_weights']['gamma']

        self.logger.info(f"Checkpoint loaded: {path}")

        return checkpoint['epoch'], checkpoint['metrics']
