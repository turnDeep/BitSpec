#!/usr/bin/env python3
# src/training/early_stopping.py
"""
Early Stopping Utility

Monitors validation loss and stops training when no improvement is observed.
"""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early Stopping to prevent overfitting

    Monitors a metric (e.g., validation loss) and stops training
    when the metric stops improving.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' (lower is better) or 'max' (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = logging.getLogger(__name__)

    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop

        Args:
            current_score: Current metric value (e.g., val_loss)

        Returns:
            early_stop: True if training should stop
        """
        if self.best_score is None:
            # First epoch
            self.best_score = current_score
            self.logger.info(f"Early stopping initialized with score: {current_score:.6f}")
            return False

        # Check if improved
        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:  # max
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            self.logger.info(
                f"Early stopping: Improvement detected. "
                f"Best score: {self.best_score:.6f}, Counter reset."
            )
        else:
            self.counter += 1
            self.logger.info(
                f"Early stopping: No improvement. "
                f"Counter: {self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(
                    f"Early stopping triggered! "
                    f"No improvement for {self.patience} epochs. "
                    f"Best score: {self.best_score:.6f}"
                )

        return self.early_stop

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger.info("Early stopping reset")
