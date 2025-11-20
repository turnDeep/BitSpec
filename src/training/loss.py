# src/training/loss.py
"""
損失関数の定義
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCosineLoss(nn.Module):
    """
    Weighted Cosine Similarity Loss for EI-MS Spectrum Prediction

    通常のコサイン類似度ベースの損失関数。
    EI-MSスペクトル予測に適した設計。

    注意: EI-MSでは、MS/MSと異なり、Shifted Matching（Neutral loss考慮）は
    適用しません。EI-MSはイオン化と同時にフラグメンテーションが発生するため、
    明確なプリカーサー-フラグメント関係が存在しないためです。
    """

    def __init__(self):
        """
        損失関数の初期化

        シンプルなコサイン類似度損失を使用。
        将来的にNIST標準の重み付け（m/z重み、強度重み）を追加可能。
        """
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        損失を計算

        Args:
            pred: 予測スペクトル [batch_size, spectrum_dim]
            target: 正解スペクトル [batch_size, spectrum_dim]

        Returns:
            損失値（1 - コサイン類似度）
        """
        # コサイン類似度を計算
        cosine_sim = F.cosine_similarity(pred, target, dim=1).mean()

        # 損失として返す（類似度が高いほど損失が小さい）
        return 1.0 - cosine_sim
