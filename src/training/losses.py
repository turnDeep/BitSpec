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
    Teacher Training Loss:
    - Bond Masking: L_teacher = L_spectrum + λ_bond * L_bond_masking
    - BDE Regression (NEW): L_teacher = λ_bde * L_bde_regression

    QC-GN2oMS2 vs NExtIMS v2.0:
    - QC-GN2oMS2: No pretraining loss (BDE as input only)
    - NExtIMS v2.0: BDE regression pretraining loss (THIS CLASS)
    """
    def __init__(self, lambda_bond: float = 0.1, lambda_bde: float = 1.0):
        super().__init__()
        self.lambda_bond = lambda_bond  # Bond Masking weight
        self.lambda_bde = lambda_bde    # BDE Regression weight (NEW)
        self.spectrum_loss = nn.MSELoss()

    def forward(
        self,
        predicted_spectrum,
        target_spectrum,
        bond_predictions=None,
        bond_targets=None,
        bde_predictions=None,
        bde_targets=None
    ):
        """
        Args:
            predicted_spectrum: [batch_size, 501]
            target_spectrum: [batch_size, 501]
            bond_predictions: [num_masked_bonds, 4] (optional, Bond Masking pretraining)
            bond_targets: [num_masked_bonds, 4] (optional, Bond Masking pretraining)
            bde_predictions: [num_edges, 1] (optional, BDE Regression pretraining, NEW)
            bde_targets: [num_edges, 1] (optional, BDE Regression pretraining, NEW)

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        # Spectrum prediction loss
        loss_spectrum = self.spectrum_loss(predicted_spectrum, target_spectrum)

        loss_dict = {'spectrum_loss': loss_spectrum.item()}

        # BDE regression loss (NEW: for BDE pretraining)
        if bde_predictions is not None and bde_targets is not None:
            if bde_predictions.numel() > 0 and bde_targets.numel() > 0:
                # MSE loss for BDE regression
                loss_bde = F.mse_loss(bde_predictions, bde_targets)
                loss = self.lambda_bde * loss_bde

                loss_dict['bde_loss'] = loss_bde.item()

                # Additional MAE metric for monitoring
                mae_bde = F.l1_loss(bde_predictions, bde_targets)
                loss_dict['bde_mae'] = mae_bde.item()
            else:
                # No edges in this batch
                loss = torch.tensor(0.0, device=predicted_spectrum.device)
                loss_dict['bde_loss'] = 0.0
                loss_dict['bde_mae'] = 0.0

        # Bond masking loss (original pretraining)
        elif bond_predictions is not None and bond_targets is not None:
            # MSE loss for bond feature prediction (bond_type, conjugated, aromatic, in_ring)
            if bond_predictions.numel() > 0 and bond_targets.numel() > 0:
                loss_bond = F.mse_loss(bond_predictions, bond_targets)
                loss = loss_spectrum + self.lambda_bond * loss_bond
                loss_dict['bond_loss'] = loss_bond.item()
            else:
                # No masked bonds in this batch
                loss = loss_spectrum
                loss_dict['bond_loss'] = 0.0

        # Spectrum-only loss (finetuning)
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

        # ✅ Constrain weights to prevent extreme values (GradNorm paper recommendation)
        # This prevents the issue where one loss dominates and others go to near-zero
        WEIGHT_CONSTRAINTS = {
            'alpha': (0.05, 0.60),   # Hard Loss: 5-60%
            'beta': (0.20, 0.80),    # Soft Loss: 20-80%
            'gamma': (0.05, 0.50)    # Feature Loss: 5-50%
        }

        for weight_name, (min_val, max_val) in WEIGHT_CONSTRAINTS.items():
            if weight_name in updated_weights:
                updated_weights[weight_name] = max(min_val, min(max_val, updated_weights[weight_name]))

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


class MultitaskTeacherLoss(nn.Module):
    """
    Multitask Teacher Loss for NIST17 Direct Training

    Combines spectrum prediction (primary task) with BDE regression (auxiliary task)
    to leverage BonDNet's high-quality BDE knowledge.

    Architecture: Option B (Recommended)
        - BonDNet: Trained on BDE-db2 (531K BDEs) → High-accuracy BDE predictor
        - NEIMS Teacher: Direct NIST17 training with:
            * Primary Task: Spectrum prediction (306K spectra)
            * Auxiliary Task: BDE regression (distill from BonDNet)

    Loss Function:
        L_total = L_spectrum + λ_bde * L_bde_auxiliary

    Benefits:
        1. Task-specific architectures (no task transfer)
        2. BonDNet knowledge distillation via auxiliary task
        3. Improved fragmentation pattern understanding
        4. Expected performance: Recall@10 = 96-97%
    """
    def __init__(
        self,
        lambda_spectrum: float = 1.0,
        lambda_bde: float = 0.1,
        use_peak_loss: bool = False,
        peak_threshold: float = 0.05
    ):
        """
        Args:
            lambda_spectrum: Weight for spectrum prediction loss (primary task)
            lambda_bde: Weight for BDE regression loss (auxiliary task)
            use_peak_loss: Whether to use peak-focused loss for spectrum
            peak_threshold: Threshold for peak detection (if use_peak_loss=True)
        """
        super().__init__()
        self.lambda_spectrum = lambda_spectrum
        self.lambda_bde = lambda_bde
        self.use_peak_loss = use_peak_loss
        self.peak_threshold = peak_threshold

        # Loss functions
        self.spectrum_loss_fn = nn.MSELoss()
        self.bde_loss_fn = nn.MSELoss()

    def forward(
        self,
        predicted_spectrum: torch.Tensor,
        target_spectrum: torch.Tensor,
        predicted_bde: Optional[torch.Tensor] = None,
        target_bde: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Compute multitask loss

        Args:
            predicted_spectrum: [batch_size, 501] - Predicted mass spectrum
            target_spectrum: [batch_size, 501] - Target mass spectrum (NIST17)
            predicted_bde: [num_edges, 1] - Predicted BDE values (optional)
            target_bde: [num_edges, 1] - Target BDE from BonDNet (optional)

        Returns:
            loss: Total weighted loss
            loss_dict: Dictionary of individual losses
        """
        # Primary Task: Spectrum Prediction
        if self.use_peak_loss:
            loss_spectrum = compute_peak_loss(
                predicted_spectrum,
                target_spectrum,
                threshold=self.peak_threshold
            )
        else:
            loss_spectrum = self.spectrum_loss_fn(predicted_spectrum, target_spectrum)

        loss_dict = {
            'spectrum_loss': loss_spectrum.item(),
            'lambda_spectrum': self.lambda_spectrum
        }

        # Total loss starts with spectrum
        total_loss = self.lambda_spectrum * loss_spectrum

        # Auxiliary Task: BDE Regression (distill from BonDNet)
        if predicted_bde is not None and target_bde is not None:
            if predicted_bde.numel() > 0 and target_bde.numel() > 0:
                # MSE loss for BDE regression
                loss_bde = self.bde_loss_fn(predicted_bde, target_bde)

                # Add to total loss
                total_loss += self.lambda_bde * loss_bde

                # Logging metrics
                loss_dict['bde_loss'] = loss_bde.item()
                loss_dict['lambda_bde'] = self.lambda_bde

                # MAE for monitoring BDE quality
                mae_bde = F.l1_loss(predicted_bde, target_bde)
                loss_dict['bde_mae'] = mae_bde.item()

                # R² coefficient for BDE prediction quality
                ss_res = ((predicted_bde - target_bde) ** 2).sum()
                ss_tot = ((target_bde - target_bde.mean()) ** 2).sum()
                r2_bde = 1.0 - (ss_res / (ss_tot + 1e-8))
                loss_dict['bde_r2'] = r2_bde.item()
            else:
                # No edges in this batch
                loss_dict['bde_loss'] = 0.0
                loss_dict['bde_mae'] = 0.0
                loss_dict['bde_r2'] = 0.0
                loss_dict['lambda_bde'] = self.lambda_bde
        else:
            # BDE auxiliary task not used (spectrum-only training)
            loss_dict['bde_loss'] = None
            loss_dict['bde_mae'] = None
            loss_dict['bde_r2'] = None

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def update_lambda_bde(self, new_lambda: float):
        """Update BDE loss weight (for curriculum learning or annealing)"""
        self.lambda_bde = new_lambda
