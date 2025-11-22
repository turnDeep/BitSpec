#!/usr/bin/env python3
# src/models/teacher.py
"""
NEIMS v2.0 Teacher Model: GNN+ECFP Hybrid Architecture

Teacher model combines Graph Neural Networks (GINEConv) with ECFP fingerprints
for high-accuracy mass spectrum prediction with uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import PairNorm
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
import numpy as np
from typing import Tuple, Optional

# Import BidirectionalModule
try:
    from src.models.modules import BidirectionalModule
except ImportError:
    # Fallback if running as script
    from models.modules import BidirectionalModule


class BondBreakingAttention(nn.Module):
    """
    Bond-Breaking Attention Module

    Predicts which bonds are likely to break during ionization,
    helping the model focus on fragmentation patterns.
    """
    def __init__(self, node_dim: int = 256, edge_dim: int = 128):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, node_features, edge_index, edge_attr):
        """
        Args:
            node_features: [N, node_dim]
            edge_index: [2, E]
            edge_attr: [E, edge_dim]

        Returns:
            bond_probs: [E, 1] - Bond breaking probabilities
        """
        row, col = edge_index
        node_i = node_features[row]  # [E, node_dim]
        node_j = node_features[col]  # [E, node_dim]

        # Concatenate node and edge features
        combined = torch.cat([node_i, node_j, edge_attr], dim=-1)  # [E, 2*node_dim + edge_dim]

        # Predict bond breaking probability
        bond_probs = self.attention_mlp(combined)  # [E, 1]

        return bond_probs


class GNNBranch(nn.Module):
    """
    GNN Branch: GINEConv with DropEdge and PairNorm
    """
    def __init__(
        self,
        node_features: int = 48,
        edge_features: int = 6,
        hidden_dim: int = 256,
        edge_dim: int = 128,
        num_layers: int = 8,
        dropout: float = 0.3,
        drop_edge: float = 0.2,
        use_pair_norm: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.drop_edge = drop_edge
        self.use_pair_norm = use_pair_norm

        # Input embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, edge_dim)

        # GINEConv layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.pair_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            if use_pair_norm:
                self.pair_norms.append(PairNorm(scale=1.0))

        # Global pooling attention
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch, dropout=False, return_node_features=False):
        """
        Args:
            x: Node features [N, node_features]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_features]
            batch: Batch vector [N]
            dropout: Whether to apply dropout (for MC Dropout)
            return_node_features: Whether to return node features for bond prediction

        Returns:
            graph_embedding: [batch_size, 768] (mean + max + attention pooling)
            node_features: [N, hidden_dim] (optional, only if return_node_features=True)
            edge_index_processed: [2, E'] (optional, only if return_node_features=True, E' after DropEdge)
            edge_attr_emb: [E', edge_dim] (optional, only if return_node_features=True)
            edge_mask: [E] boolean mask (optional, only if return_node_features=True, shows which edges were kept)
        """
        # Embed inputs
        x = self.node_embedding(x)
        edge_attr_emb = self.edge_embedding(edge_attr)

        # Apply DropEdge if training
        edge_mask = None
        if dropout and self.training:
            edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.drop_edge
            edge_index = edge_index[:, edge_mask]
            edge_attr_emb = edge_attr_emb[edge_mask]
        else:
            # No DropEdge applied, all edges are kept
            edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

        # GNN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr_emb)
            x = self.batch_norms[i](x)
            if self.use_pair_norm:
                x = self.pair_norms[i](x, batch)
            x = F.relu(x)
            if dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling: mean + max + attention
        mean_pool = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        max_pool = global_max_pool(x, batch)    # [batch_size, hidden_dim]

        # Attention pooling
        attention_scores = self.attention_pooling(x)  # [N, 1]
        attention_weights = torch.exp(attention_scores)
        attention_weights = attention_weights / (global_add_pool(attention_weights, batch)[batch] + 1e-8)
        attention_pool = global_add_pool(x * attention_weights, batch)  # [batch_size, hidden_dim]

        # Concatenate all pooling methods
        graph_embedding = torch.cat([mean_pool, max_pool, attention_pool], dim=-1)  # [batch_size, 768]

        if return_node_features:
            return graph_embedding, x, edge_index, edge_attr_emb, edge_mask
        return graph_embedding


class ECFPBranch(nn.Module):
    """
    ECFP Branch: Extended Connectivity Fingerprint processing
    """
    def __init__(
        self,
        fingerprint_size: int = 4096,
        mlp_hidden: int = 1024,
        mlp_output: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fingerprint_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_output),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, ecfp):
        """
        Args:
            ecfp: ECFP4 fingerprint [batch_size, fingerprint_size]

        Returns:
            ecfp_embedding: [batch_size, mlp_output]
        """
        return self.mlp(ecfp)


class TeacherModel(nn.Module):
    """
    NEIMS v2.0 Teacher Model: GNN+ECFP Hybrid

    High-capacity model for training only. Uses MC Dropout for uncertainty estimation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Extract configuration
        gnn_cfg = config['model']['teacher']['gnn']
        ecfp_cfg = config['model']['teacher']['ecfp']
        pred_cfg = config['model']['teacher']['prediction']
        mc_cfg = config['model']['teacher']['mc_dropout']
        common_cfg = config['model']['common']

        # GNN Branch
        self.gnn_branch = GNNBranch(
            node_features=common_cfg['node_features'],
            edge_features=common_cfg['edge_features'],
            hidden_dim=gnn_cfg['hidden_dim'],
            edge_dim=gnn_cfg['edge_dim'],
            num_layers=gnn_cfg['num_layers'],
            dropout=gnn_cfg['dropout'],
            drop_edge=gnn_cfg['drop_edge'],
            use_pair_norm=gnn_cfg['use_pair_norm']
        )

        # Bond-Breaking Attention (optional)
        self.use_bond_breaking = gnn_cfg.get('use_bond_breaking', False)
        if self.use_bond_breaking:
            self.bond_breaking = BondBreakingAttention(
                node_dim=gnn_cfg['hidden_dim'],
                edge_dim=gnn_cfg['edge_dim']
            )
            # Bond feature prediction head for pretraining
            # Predicts: bond_type, is_conjugated, is_aromatic, is_in_ring
            self.bond_feature_head = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 4)
            )

        # ECFP Branch
        self.ecfp_branch = ECFPBranch(
            fingerprint_size=ecfp_cfg['fingerprint_size'],
            mlp_hidden=ecfp_cfg['mlp_hidden'],
            mlp_output=ecfp_cfg['mlp_output'],
            dropout=ecfp_cfg['dropout']
        )

        # Fusion
        fusion_dim = config['model']['teacher']['fusion_dim']  # GNN(768) + ECFP(512) = 1280

        # Prediction Head
        hidden_dims = pred_cfg['hidden_dims']
        output_dim = pred_cfg['output_dim']
        use_bidirectional = pred_cfg.get('use_bidirectional', False)

        layers = []
        input_dim = fusion_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(gnn_cfg['dropout'])
            ])
            input_dim = hidden_dim

        if use_bidirectional:
            layers.append(BidirectionalModule(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, output_dim))

        self.prediction_head = nn.Sequential(*layers)

        # MC Dropout configuration
        self.mc_samples = mc_cfg['n_samples']
        self.mc_dropout_rate = mc_cfg['dropout_rate']

    def forward(self, graph_data, ecfp, dropout=False, return_bond_predictions=False):
        """
        Forward pass

        Args:
            graph_data: PyG Data object with x, edge_index, edge_attr, batch
            ecfp: ECFP4 fingerprint [batch_size, 4096]
            dropout: Whether to apply dropout (for MC Dropout)
            return_bond_predictions: Whether to return bond predictions (for pretraining)

        Returns:
            spectrum: Predicted spectrum [batch_size, output_dim]
            bond_predictions: Bond masking predictions [num_masked_bonds, 4] (optional)
        """
        # GNN Branch
        if return_bond_predictions and self.use_bond_breaking:
            gnn_emb, node_features, edge_index_processed, edge_attr_emb, edge_mask = self.gnn_branch(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr,
                graph_data.batch,
                dropout=dropout,
                return_node_features=True
            )
        else:
            gnn_emb = self.gnn_branch(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr,
                graph_data.batch,
                dropout=dropout
            )  # [batch_size, 768]

        # ECFP Branch
        ecfp_emb = self.ecfp_branch(ecfp)  # [batch_size, 512]

        # Fusion
        fused = torch.cat([gnn_emb, ecfp_emb], dim=-1)  # [batch_size, 1280]

        # Prediction
        spectrum = self.prediction_head(fused)  # [batch_size, output_dim]

        # Bond masking predictions (for pretraining)
        if return_bond_predictions and self.use_bond_breaking:
            # Use BondBreakingAttention to predict bond breaking probabilities
            # Use the processed edge_index (after DropEdge) to match edge_attr_emb size
            bond_probs = self.bond_breaking(
                node_features,
                edge_index_processed,
                edge_attr_emb
            )  # [E', 1] - breaking probabilities (E' after DropEdge)

            # Predict bond features (type, conjugated, aromatic, in_ring)
            bond_predictions = self.bond_feature_head(bond_probs)  # [E', 4]

            # Filter to only masked bonds if mask_indices available
            # Also return a mask indicating which original mask_indices survived DropEdge
            valid_bond_mask = None
            if hasattr(graph_data, 'mask_indices') and graph_data.mask_indices.numel() > 0:
                # Adjust mask_indices based on edge_mask (DropEdge)
                # Create mapping from old indices to new indices
                old_to_new_idx = torch.cumsum(edge_mask, dim=0) - 1

                # Filter mask_indices: keep only those that were not dropped
                valid_bond_mask = edge_mask[graph_data.mask_indices]
                adjusted_mask_indices = old_to_new_idx[graph_data.mask_indices[valid_bond_mask]]

                # Only select bond predictions for valid masked bonds
                if adjusted_mask_indices.numel() > 0:
                    bond_predictions = bond_predictions[adjusted_mask_indices]
                else:
                    # No masked bonds survived DropEdge
                    bond_predictions = torch.zeros((0, 4), device=bond_predictions.device)

            # Store valid_bond_mask in graph_data for trainer to use
            if valid_bond_mask is not None:
                graph_data.valid_bond_mask = valid_bond_mask

            return spectrum, bond_predictions

        return spectrum

    def predict_with_uncertainty(
        self,
        graph_data,
        ecfp,
        n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout uncertainty estimation

        Args:
            graph_data: PyG Data object
            ecfp: ECFP4 fingerprint
            n_samples: Number of MC samples (default: self.mc_samples)

        Returns:
            mean_spectrum: Mean prediction [batch_size, output_dim]
            std_spectrum: Standard deviation [batch_size, output_dim]
        """
        n_samples = n_samples or self.mc_samples

        # Enable dropout for MC sampling
        self.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(graph_data, ecfp, dropout=True)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [n_samples, batch_size, output_dim]

        mean_spectrum = predictions.mean(dim=0)
        std_spectrum = predictions.std(dim=0)

        self.eval()

        return mean_spectrum, std_spectrum

    def get_ecfp_embedding(self, ecfp):
        """Get ECFP embedding for feature-level distillation"""
        return self.ecfp_branch(ecfp)


def compute_ecfp4(smiles: str, fingerprint_size: int = 4096) -> np.ndarray:
    """
    Compute ECFP4 fingerprint from SMILES

    Args:
        smiles: SMILES string
        fingerprint_size: Fingerprint size (default: 4096)

    Returns:
        ecfp: Binary fingerprint array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fingerprint_size)

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fingerprint_size)
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp)
