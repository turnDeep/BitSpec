#!/usr/bin/env python3
# src/models/moe.py
"""
Mixture of Experts (MoE) Components

Implements load balancing and routing utilities for MoE architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_balancing_loss(expert_weights, expert_indices, num_experts=4):
    """
    Load Balancing Loss (Switch Transformer style)

    Encourages uniform distribution of samples across experts.

    Args:
        expert_weights: Gating weights [batch_size, num_experts]
        expert_indices: Selected expert indices [batch_size] or [batch_size, top_k]
        num_experts: Total number of experts

    Returns:
        loss: Load balancing loss scalar
    """
    batch_size = expert_weights.size(0)

    # Flatten expert indices if top-k routing
    if expert_indices.dim() > 1:
        expert_indices = expert_indices.flatten()

    # Compute expert selection frequency
    expert_counts = torch.bincount(expert_indices, minlength=num_experts).float()
    expert_freq = expert_counts / expert_counts.sum()

    # Compute average gating weight per expert
    expert_avg_weight = expert_weights.mean(dim=0)

    # Load balancing loss: num_experts * sum(freq * avg_weight)
    loss = num_experts * (expert_freq * expert_avg_weight).sum()

    return loss


def entropy_regularization(expert_weights):
    """
    Entropy Regularization

    Maximizes entropy of expert selection to prevent collapse.

    Args:
        expert_weights: Gating weights [batch_size, num_experts]

    Returns:
        loss: Negative entropy (to minimize, i.e., maximize entropy)
    """
    eps = 1e-8
    entropy = -(expert_weights * torch.log(expert_weights + eps)).sum(dim=-1)

    # Return negative entropy (we want to maximize entropy, i.e., minimize negative entropy)
    return -entropy.mean()


class AuxiliaryLossFreeLoadBalancing(nn.Module):
    """
    Auxiliary-Loss-Free Load Balancing

    Uses dynamic biasing of expert logits to achieve load balancing
    without additional loss terms.
    """
    def __init__(self, num_experts=4, momentum=0.9, bias_scale=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.momentum = momentum
        self.bias_scale = bias_scale

        # Exponential moving average of expert loads
        self.register_buffer('expert_ema', torch.zeros(num_experts))

    def update(self, expert_indices):
        """
        Update expert load statistics

        Args:
            expert_indices: Selected expert indices [batch_size] or [batch_size, top_k]

        Returns:
            bias: Bias to add to expert logits [num_experts]
        """
        # Flatten if top-k
        if expert_indices.dim() > 1:
            expert_indices = expert_indices.flatten()

        # Count expert usage
        expert_counts = torch.bincount(expert_indices, minlength=self.num_experts).float()

        # Update EMA
        self.expert_ema = self.momentum * self.expert_ema + (1 - self.momentum) * expert_counts

        # Compute target load (uniform)
        target_load = self.expert_ema.sum() / self.num_experts

        # Bias: discourage overused experts, encourage underused ones
        bias = -self.bias_scale * (self.expert_ema - target_load)

        return bias

    def forward(self, expert_indices):
        """Alias for update()"""
        return self.update(expert_indices)


def compute_expert_usage_stats(expert_indices, num_experts=4):
    """
    Compute expert usage statistics for logging

    Args:
        expert_indices: Selected expert indices [batch_size] or [batch_size, top_k]
        num_experts: Total number of experts

    Returns:
        stats: Dictionary with usage statistics
    """
    if expert_indices.dim() > 1:
        expert_indices = expert_indices.flatten()

    expert_counts = torch.bincount(expert_indices, minlength=num_experts).float()
    total = expert_counts.sum()

    stats = {
        'expert_counts': expert_counts.cpu().numpy(),
        'expert_percentages': (expert_counts / total * 100).cpu().numpy(),
        'entropy': compute_entropy(expert_counts / total),
        'balance_score': compute_balance_score(expert_counts)
    }

    return stats


def compute_entropy(distribution):
    """Compute entropy of a distribution"""
    eps = 1e-8
    distribution = distribution + eps
    entropy = -(distribution * torch.log(distribution)).sum()
    return entropy.item()


def compute_balance_score(counts):
    """
    Compute balance score (0-1, higher is better)

    Perfect balance = 1.0, complete imbalance = 0.0
    """
    total = counts.sum()
    if total == 0:
        return 0.0

    # Compute coefficient of variation
    mean = counts.mean()
    std = counts.std()

    if mean == 0:
        return 0.0

    cv = std / mean  # Coefficient of variation

    # Convert to balance score (0-1, where 1 is perfect balance)
    balance = 1.0 / (1.0 + cv)

    return balance.item()


class TopKRouter(nn.Module):
    """
    Top-K Router for MoE

    Selects top-k experts based on gating scores.
    """
    def __init__(self, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, gate_logits):
        """
        Args:
            gate_logits: Gating logits [batch_size, num_experts]

        Returns:
            expert_weights: Normalized weights for top-k [batch_size, top_k]
            expert_indices: Indices of top-k experts [batch_size, top_k]
            all_weights: All expert weights (softmax) [batch_size, num_experts]
        """
        # Softmax over all experts
        all_weights = F.softmax(gate_logits, dim=-1)

        # Select top-k
        top_k_weights, top_k_indices = torch.topk(all_weights, self.top_k, dim=-1)

        # Renormalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return top_k_weights, top_k_indices, all_weights


def sparse_dispatcher(inputs, expert_indices, num_experts):
    """
    Sparse Dispatcher: Route inputs to experts

    Args:
        inputs: Input tensor [batch_size, dim]
        expert_indices: Selected expert indices [batch_size] (top-1 only)
        num_experts: Number of experts

    Returns:
        dispatched: List of inputs for each expert
    """
    dispatched = []

    for expert_id in range(num_experts):
        mask = (expert_indices == expert_id)
        if mask.any():
            dispatched.append(inputs[mask])
        else:
            # No samples for this expert
            dispatched.append(None)

    return dispatched
