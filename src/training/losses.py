#!/usr/bin/env python3
# src/training/losses.py
"""
NEIMS v2.0 Loss Functions

Implements all loss functions for Teacher and Student training,
including knowledge distillation, load balancing, and entropy regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from src.models.moe import load_balancing_loss, entropy_regularization


class TeacherLoss(nn.Module):
    """
    Teacher Training Loss: L_teacher = L_spectrum + λ_bond * L_bond_masking
    """
    def __init__(self, lambda_bond: float = 0.1):
        super().__init__()
        self.lambda_bond = lambda_bond
        self.spectrum_loss = nn.MSELoss()

    def forward(
        self,
        predicted_spectrum,
        target_spectrum,
        bond_predictions=None,
        bond_targets=None
    ):
        """
        Args:
            predicted_spectrum: [batch_size, 501]
            target_spectrum: [batch_size, 501]
            bond_predictions: [E, 1] (optional, for pretraining)
            bond_targets: [E] (optional, for pretraining)

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        # Spectrum prediction loss
        loss_spectrum = self.spectrum_loss(predicted_spectrum, target_spectrum)

        loss_dict = {'spectrum_loss': loss_spectrum.item()}

        # Bond masking loss (pretraining only)
        if bond_predictions is not None and bond_targets is not None:
            loss_bond = F.binary_cross_entropy(
                bond_predictions.squeeze(),
                bond_targets.float()
            )
            loss = loss_spectrum + self.lambda_bond * loss_bond
            loss_dict['bond_loss'] = loss_bond.item()
        else:
            loss = loss_spectrum

        loss_dict['total_loss'] = loss.item()

        return loss, loss_dict


class StudentDistillationLoss(nn.Module):
    """
    Student Knowledge Distillation Loss:
    L_student = α*L_hard + β*L_soft + γ*L_feature + δ_load*L_load + δ_entropy*L_entropy
    """
    def __init__(
        self,
        alpha_init: float = 0.3,
        beta_init: float = 0.5,
        gamma_init: float = 0.2,
        delta_load: float = 0.01,
        delta_entropy: float = 0.001,
        use_lds: bool = True,
        lds_sigma: float = 1.5
    ):
        super().__init__()
        # Initial loss weights
        self.alpha = alpha_init
        self.beta = beta_init
        self.gamma = gamma_init
        self.delta_load = delta_load
        self.delta_entropy = delta_entropy

        # Label Distribution Smoothing
        self.use_lds = use_lds
        if use_lds:
            from src.models.modules import GaussianSmoothing
            self.lds = GaussianSmoothing(sigma=lds_sigma)

    def forward(
        self,
        student_output,
        teacher_mean,
        teacher_std,
        nist_spectrum,
        student_features,
        teacher_features,
        expert_weights,
        expert_indices,
        temperature: float = 1.0
    ) -> tuple:
        """
        Compute complete distillation loss

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        # L1: Hard Label Loss
        loss_hard = F.mse_loss(student_output, nist_spectrum)

        # L2: Soft Label Loss (Uncertainty-Aware)
        loss_soft = self._compute_soft_loss(
            student_output,
            teacher_mean,
            teacher_std,
            temperature
        )

        # L3: Feature-Level Distillation
        loss_feature = F.mse_loss(student_features, teacher_features)

        # L4: Load Balancing Loss
        loss_load = load_balancing_loss(expert_weights, expert_indices)

        # L5: Entropy Regularization
        loss_entropy = entropy_regularization(expert_weights)

        # Total loss
        loss = (
            self.alpha * loss_hard +
            self.beta * loss_soft +
            self.gamma * loss_feature +
            self.delta_load * loss_load +
            self.delta_entropy * loss_entropy
        )

        loss_dict = {
            'total_loss': loss.item(),
            'hard_loss': loss_hard.item(),
            'soft_loss': loss_soft.item(),
            'feature_loss': loss_feature.item(),
            'load_loss': loss_load.item(),
            'entropy_loss': loss_entropy.item(),
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
        }

        return loss, loss_dict

    def _compute_soft_loss(
        self,
        student_output,
        teacher_mean,
        teacher_std,
        temperature
    ):
        """
        Compute uncertainty-aware soft label loss with temperature scaling
        """
        # Apply LDS to teacher output
        if self.use_lds:
            teacher_smoothed = self.lds(teacher_mean)
        else:
            teacher_smoothed = teacher_mean

        # Compute confidence weights (inverse uncertainty)
        confidence = 1.0 / (1.0 + teacher_std)
        confidence = confidence / confidence.sum(dim=-1, keepdim=True)

        # Temperature scaling
        teacher_soft = teacher_smoothed / temperature
        student_scaled = student_output / temperature

        # Confidence-weighted MSE loss
        loss = F.mse_loss(
            student_scaled * confidence,
            teacher_soft * confidence
        ) * (temperature ** 2)

        return loss

    def update_weights(self, alpha, beta, gamma):
        """Update loss weights (for GradNorm)"""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class GradNormWeighting:
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing

    Automatically balances multiple loss terms by equalizing gradient magnitudes.
    """
    def __init__(
        self,
        initial_losses: Dict[str, float],
        alpha: float = 1.5,
        gradient_clip_range: tuple = (0.5, 2.0)
    ):
        self.initial_losses = initial_losses
        self.alpha = alpha
        self.clip_min, self.clip_max = gradient_clip_range

    def compute_weights(
        self,
        current_losses: Dict[str, torch.Tensor], # Now accepts Tensors
        loss_weights: Dict[str, float],
        model_params
    ) -> Dict[str, float]:
        """
        Compute updated loss weights using GradNorm

        Args:
            current_losses: Current loss tensors (with grad) {'hard': ..., 'soft': ..., 'feature': ...}
            loss_weights: Current weights {'alpha': ..., 'beta': ..., 'gamma': ...}
            model_params: Model parameters (iterable) for gradient computation

        Returns:
            updated_weights: Updated loss weights
        """
        # Ensure model_params is a list
        model_params = list(model_params)

        # Compute gradient norms for each loss
        grad_norms = {}

        for loss_name, loss_tensor in current_losses.items():
            # Weighted loss is what's usually backpropped, but GradNorm typically looks at
            # gradients of the *unweighted* individual losses w.r.t shared weights.
            # Here we assume loss_tensor is the unweighted loss.

            # Compute gradients w.r.t. model_params
            # We use retain_graph=True because we might need the graph for subsequent backward() or other losses
            grads = torch.autograd.grad(
                loss_tensor,
                model_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )

            # Compute L2 norm of gradients
            # Filter out None grads (params unused by this specific loss)
            valid_grads = [g for g in grads if g is not None]
            if not valid_grads:
                 grad_norm = torch.tensor(0.0, device=loss_tensor.device)
            else:
                 grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in valid_grads))

            grad_norms[loss_name] = grad_norm.item()

        # Compute average gradient norm
        avg_grad_norm = sum(grad_norms.values()) / max(len(grad_norms), 1)

        # Compute loss ratios
        loss_ratios = {}
        for loss_name, loss_tensor in current_losses.items():
            loss_value = loss_tensor.item()
            initial_loss = self.initial_losses[loss_name]
            loss_ratio = loss_value / (initial_loss + 1e-8)
            loss_ratios[loss_name] = loss_ratio ** self.alpha

        # Compute target gradient norms
        avg_loss_ratio = sum(loss_ratios.values()) / max(len(loss_ratios), 1)
        target_grad_norms = {
            name: avg_grad_norm * (ratio / (avg_loss_ratio + 1e-8))
            for name, ratio in loss_ratios.items()
        }

        # Update weights
        updated_weights = {}
        for loss_name, weight_name in [('hard', 'alpha'), ('soft', 'beta'), ('feature', 'gamma')]:
            if loss_name in grad_norms and loss_name in target_grad_norms:
                current_weight = loss_weights[weight_name]
                grad_norm = grad_norms[loss_name]
                target_norm = target_grad_norms[loss_name]

                # Compute weight update ratio
                ratio = torch.clamp(
                    torch.tensor(target_norm / (grad_norm + 1e-8)),
                    self.clip_min,
                    self.clip_max
                )

                updated_weights[weight_name] = current_weight * ratio.item()
            else:
                updated_weights[weight_name] = loss_weights.get(weight_name, 0.0)

        # Normalize weights to sum to 1.0
        total = sum(updated_weights.values())
        if total > 0:
            updated_weights = {k: v / total for k, v in updated_weights.items()}

        return updated_weights


def compute_spectral_angle_loss(predicted, target):
    """
    Spectral Angle Mapper (SAM) Loss

    Measures the angle between predicted and target spectra.
    Useful for spectrum similarity.
    """
    # Normalize vectors
    pred_norm = F.normalize(predicted, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)

    # Compute cosine similarity
    cos_sim = (pred_norm * target_norm).sum(dim=-1)

    # Clamp to avoid numerical issues
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # Convert to angle (in radians)
    angle = torch.acos(cos_sim)

    # Average angle
    loss = angle.mean()

    return loss


def compute_peak_loss(predicted, target, threshold=0.05):
    """
    Peak-Focused Loss

    Emphasizes correct prediction of significant peaks.
    """
    # Identify significant peaks in target
    peak_mask = target > (target.max(dim=-1, keepdim=True)[0] * threshold)

    # Compute weighted MSE (higher weight on peaks)
    weights = torch.where(peak_mask, torch.tensor(10.0), torch.tensor(1.0)).to(predicted.device)

    loss = (weights * (predicted - target) ** 2).mean()

    return loss
