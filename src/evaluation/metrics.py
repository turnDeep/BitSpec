#!/usr/bin/env python3
# src/evaluation/metrics.py
"""
NEIMS v2.0 Evaluation Metrics

Implements Recall@K, Spectral Similarity, and other metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List


def recall_at_k(predicted: torch.Tensor, target: torch.Tensor, k: int = 10) -> float:
    """
    Recall@K: Fraction of top-k target peaks found in top-k predictions
    
    Args:
        predicted: Predicted spectrum [batch_size, 501]
        target: Target spectrum [batch_size, 501]
        k: Number of top peaks
        
    Returns:
        recall: Average recall score
    """
    batch_size = predicted.size(0)
    recalls = []
    
    for i in range(batch_size):
        # Get top-k indices
        target_topk = torch.topk(target[i], k).indices.cpu().numpy()
        pred_topk = torch.topk(predicted[i], k).indices.cpu().numpy()
        
        # Compute overlap
        overlap = len(set(target_topk) & set(pred_topk))
        recall = overlap / k
        recalls.append(recall)
    
    return np.mean(recalls)


def spectral_similarity(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """
    Spectral Similarity (Cosine Similarity)
    
    Returns:
        similarity: Average cosine similarity
    """
    similarity = F.cosine_similarity(predicted, target, dim=-1)
    return similarity.mean().item()


def compute_all_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Compute all evaluation metrics
    
    Returns:
        metrics: Dictionary of all metrics
    """
    metrics = {}
    
    # Recall@K for different K values
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(predicted, target, k)
    
    # Spectral similarity
    metrics['spectral_similarity'] = spectral_similarity(predicted, target)
    
    # MAE and MSE
    metrics['mae'] = F.l1_loss(predicted, target).item()
    metrics['mse'] = F.mse_loss(predicted, target).item()
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    return metrics
