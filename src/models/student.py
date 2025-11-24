#!/usr/bin/env python3
# src/models/student.py
"""
NEIMS v2.0 Student Model: MoE-Residual MLP Architecture

Lightweight student model using Mixture of Experts (MoE) with residual connections
for fast inference (~10ms per molecule).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

# Import BidirectionalModule
try:
    from src.models.modules import BidirectionalModule
except ImportError:
    # Fallback if running as script
    from models.modules import BidirectionalModule


class GateNetwork(nn.Module):
    """
    Gate Network: Routes inputs to appropriate experts

    Uses softmax with top-k routing to select the most relevant experts.
    """
    def __init__(
        self,
        input_dim: int = 6144,
        hidden_dims: list = [512, 128],
        num_experts: int = 4,
        top_k: int = 2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_experts))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            expert_weights: Softmax weights for top-k experts [batch_size, top_k]
            expert_indices: Indices of top-k experts [batch_size, top_k]
            all_weights: All expert weights (for loss computation) [batch_size, num_experts]
        """
        logits = self.mlp(x)  # [batch_size, num_experts]
        all_weights = F.softmax(logits, dim=-1)

        # Top-k routing
        top_k_weights, top_k_indices = torch.topk(all_weights, self.top_k, dim=-1)

        # Renormalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return top_k_weights, top_k_indices, all_weights


class ResidualBlock(nn.Module):
    """
    Residual Block with LayerNorm and GELU activation
    """
    def __init__(self, dim: int = 6144, use_layer_norm: bool = True):
        super().__init__()
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 3),
            nn.GELU(),
            nn.Linear(dim // 3, dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input [batch_size, dim]

        Returns:
            output: Residual output [batch_size, dim]
        """
        identity = x

        if self.use_layer_norm:
            x = self.norm(x)

        x = self.mlp(x)

        return x + identity


class ExpertNetwork(nn.Module):
    """
    Expert Network: Deep Residual MLP

    Each expert specializes in a specific type of molecule:
    - Expert 0: Aromatic compounds
    - Expert 1: Aliphatic compounds
    - Expert 2: Heterocyclic compounds
    - Expert 3: General/Mixed
    """
    def __init__(
        self,
        input_dim: int = 6144,
        num_residual_blocks: int = 6,
        use_layer_norm: bool = True
    ):
        super().__init__()

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(input_dim, use_layer_norm)
            for _ in range(num_residual_blocks)
        ])

    def forward(self, x):
        """
        Args:
            x: Input [batch_size, input_dim]

        Returns:
            output: Expert output [batch_size, input_dim]
        """
        for block in self.blocks:
            x = block(x)
        return x


class StudentModel(nn.Module):
    """
    NEIMS v2.0 Student Model: MoE-Residual MLP

    Fast inference model using Mixture of Experts with residual connections.
    Designed for production deployment with ~10ms inference time.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Extract configuration
        student_cfg = config['model']['student']
        gate_cfg = student_cfg['gate']
        expert_cfg = student_cfg['experts']
        pred_cfg = student_cfg['prediction']

        self.input_dim = student_cfg['input_dim']
        self.num_experts = expert_cfg['num_experts']
        self.top_k = gate_cfg['top_k']

        # Gate Network
        self.gate = GateNetwork(
            input_dim=self.input_dim,
            hidden_dims=gate_cfg['hidden_dims'],
            num_experts=self.num_experts,
            top_k=self.top_k
        )

        # Expert Networks
        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim=self.input_dim,
                num_residual_blocks=expert_cfg['residual_blocks_per_expert'],
                use_layer_norm=expert_cfg['use_layer_norm']
            )
            for _ in range(self.num_experts)
        ])

        # Expert bias for auxiliary-loss-free load balancing
        self.register_buffer('expert_bias', torch.zeros(self.num_experts))
        self.register_buffer('expert_load_history', torch.zeros(self.num_experts))

        # Prediction Head
        hidden_dims = pred_cfg['hidden_dims']
        output_dim = pred_cfg['output_dim']
        dropout = pred_cfg['dropout']
        use_bidirectional = pred_cfg.get('use_bidirectional', False)

        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        if use_bidirectional:
            layers.append(BidirectionalModule(prev_dim, output_dim))
        else:
            layers.append(nn.Linear(prev_dim, output_dim))

        self.prediction_head = nn.Sequential(*layers)

    def forward(self, ecfp_count_fp):
        """
        Forward pass

        Args:
            ecfp_count_fp: Concatenated ECFP4 + Count FP [batch_size, 6144]

        Returns:
            spectrum: Predicted spectrum [batch_size, output_dim]
            expert_weights: All expert weights [batch_size, num_experts]
            expert_indices: Selected expert indices [batch_size, top_k]
        """
        # Gate decision (with bias for load balancing)
        gate_logits = self.gate.mlp(ecfp_count_fp)  # [batch_size, num_experts]
        gate_logits = gate_logits + self.expert_bias.unsqueeze(0)  # Apply bias

        # Get expert routing from biased logits
        all_weights = F.softmax(gate_logits, dim=-1)

        # Top-k routing
        top_k_weights, expert_indices = torch.topk(all_weights, self.top_k, dim=-1)

        # Renormalize top-k weights
        expert_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Compute expert outputs (only for selected experts)
        batch_size = ecfp_count_fp.size(0)
        combined = torch.zeros(batch_size, self.input_dim, device=ecfp_count_fp.device)

        for i in range(self.top_k):
            # Get expert index for each batch element
            indices = expert_indices[:, i]  # [batch_size]
            weights = expert_weights[:, i].unsqueeze(-1)  # [batch_size, 1]

            # Compute expert outputs (batched)
            for expert_id in range(self.num_experts):
                mask = (indices == expert_id)
                if mask.any():
                    expert_input = ecfp_count_fp[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    combined[mask] += weights[mask] * expert_output

        # Prediction head
        spectrum = self.prediction_head(combined)  # [batch_size, output_dim]

        return spectrum, all_weights, expert_indices

    def update_expert_bias(self, expert_counts):
        """
        Update expert bias for auxiliary-loss-free load balancing

        Args:
            expert_counts: Counts of how many times each expert was used
        """
        # Exponential moving average of expert usage
        self.expert_load_history = 0.9 * self.expert_load_history + 0.1 * expert_counts.float()

        # Compute target load (uniform distribution)
        target_load = expert_counts.sum().float() / self.num_experts

        # Update bias to discourage overused experts
        self.expert_bias = -0.1 * (self.expert_load_history - target_load)

    def get_hidden_features(self, ecfp_count_fp):
        """
        Get hidden features for feature-level distillation

        Args:
            ecfp_count_fp: Input features [batch_size, 6144]

        Returns:
            features: Hidden features before prediction head [batch_size, 6144]
        """
        # Gate decision (with bias for load balancing)
        gate_logits = self.gate.mlp(ecfp_count_fp)
        gate_logits = gate_logits + self.expert_bias.unsqueeze(0)

        # Get expert routing from biased logits
        all_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, expert_indices = torch.topk(all_weights, self.top_k, dim=-1)
        expert_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        batch_size = ecfp_count_fp.size(0)
        combined = torch.zeros(batch_size, self.input_dim, device=ecfp_count_fp.device)

        for i in range(self.top_k):
            indices = expert_indices[:, i]
            weights = expert_weights[:, i].unsqueeze(-1)

            for expert_id in range(self.num_experts):
                mask = (indices == expert_id)
                if mask.any():
                    expert_input = ecfp_count_fp[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    combined[mask] += weights[mask] * expert_output

        return combined


def compute_count_fingerprint(smiles: str, fingerprint_size: int = 2048) -> np.ndarray:
    """
    Compute Count Fingerprint from SMILES

    Count fingerprints record the number of times each substructure appears,
    providing richer information than binary fingerprints.

    Args:
        smiles: SMILES string
        fingerprint_size: Fingerprint size (default: 2048)

    Returns:
        count_fp: Count fingerprint array
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fingerprint_size)

    # Morgan fingerprint with counts
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fingerprint_size)
    fp = mfpgen.GetCountFingerprint(mol)

    # Convert to numpy array
    count_fp = np.zeros(fingerprint_size)
    for idx, count in fp.GetNonzeroElements().items():
        count_fp[idx] = min(count, 255)  # Cap at 255

    return count_fp


def prepare_student_input(smiles: str) -> torch.Tensor:
    """
    Prepare input for Student model: ECFP4 + Count FP

    Args:
        smiles: SMILES string

    Returns:
        input_tensor: Concatenated ECFP4 + Count FP [6144]
    """
    from src.models.teacher import compute_ecfp4

    ecfp4 = compute_ecfp4(smiles, fingerprint_size=4096)
    count_fp = compute_count_fingerprint(smiles, fingerprint_size=2048)

    input_array = np.concatenate([ecfp4, count_fp])
    return torch.from_numpy(input_array).float()
