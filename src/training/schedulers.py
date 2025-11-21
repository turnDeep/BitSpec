#!/usr/bin/env python3
# src/training/schedulers.py
"""
NEIMS v2.0 Schedulers

Temperature annealing and learning rate schedulers for knowledge distillation.
"""

import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class TemperatureScheduler:
    """
    Temperature Annealing Scheduler for Knowledge Distillation

    Gradually decreases temperature from T_init to T_min over training.
    """
    def __init__(
        self,
        T_init: float = 4.0,
        T_min: float = 1.0,
        max_epochs: int = 150,
        schedule: str = 'cosine'
    ):
        self.T_init = T_init
        self.T_min = T_min
        self.max_epochs = max_epochs
        self.schedule = schedule

    def get_temperature(self, epoch: int) -> float:
        """
        Get temperature for current epoch

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            temperature: Current temperature value
        """
        progress = epoch / self.max_epochs

        if self.schedule == 'cosine':
            # Cosine annealing
            T = self.T_min + (self.T_init - self.T_min) * \
                0.5 * (1 + np.cos(np.pi * progress))

        elif self.schedule == 'linear':
            # Linear decay
            T = self.T_init - (self.T_init - self.T_min) * progress

        elif self.schedule == 'exponential':
            # Exponential decay
            decay_rate = -np.log(self.T_min / self.T_init) / self.max_epochs
            T = self.T_init * np.exp(-decay_rate * epoch)

        elif self.schedule == 'step':
            # Step decay (halve every 1/3 of training)
            if progress < 1/3:
                T = self.T_init
            elif progress < 2/3:
                T = (self.T_init + self.T_min) / 2
            else:
                T = self.T_min

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return max(T, self.T_min)


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning Rate Scheduler with Warmup and Cosine Annealing

    Combines linear warmup with cosine annealing decay.
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / \
                       (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))

            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]


class AdaptiveLRScheduler:
    """
    Adaptive Learning Rate Scheduler

    Reduces LR when validation loss plateaus.
    """
    def __init__(
        self,
        optimizer,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 5,
        threshold: float = 1e-4,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr

        self.best_val = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0

    def step(self, val_loss):
        """
        Update learning rate based on validation loss

        Args:
            val_loss: Current validation loss
        """
        # Check if improvement
        if self.mode == 'min':
            is_better = val_loss < (self.best_val - self.threshold)
        else:
            is_better = val_loss > (self.best_val + self.threshold)

        if is_better:
            self.best_val = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Reduce LR if no improvement for patience epochs
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        """Reduce learning rate"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}")


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1
):
    """
    Create a schedule with linear warmup and linear decay

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        last_epoch: The index of last epoch

    Returns:
        scheduler: LR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=0.0,
    power=1.0,
    last_epoch=-1
):
    """
    Create a schedule with polynomial warmup and decay

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        lr_end: Final learning rate
        power: Polynomial power
        last_epoch: The index of last epoch

    Returns:
        scheduler: LR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        lr_range = 1.0 - lr_end
        pct_remaining = 1 - (current_step - num_warmup_steps) / \
                        (num_training_steps - num_warmup_steps)
        return lr_end + lr_range * pct_remaining ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
