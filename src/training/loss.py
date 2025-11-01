# src/training/loss.py
"""
損失関数の定義
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ModifiedCosineLoss(nn.Module):
    """
    Modified Cosine Loss

    通常のコサイン類似度に加えて、neutral loss（中性損失）を考慮した
    Modified Cosine Similarityを計算する損失関数。
    プリカーサーイオンの質量差を利用してピークのシフトマッチングを行う。
    """

    def __init__(self, tolerance: float = 0.1):
        """
        Args:
            tolerance: ピークマッチングの許容誤差（m/z単位）
        """
        super().__init__()
        self.tolerance = tolerance

    def _compute_shifted_matching(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        precursor_mz_diff: torch.Tensor,
        tolerance: float
    ) -> torch.Tensor:
        """
        プリカーサー質量差を考慮したシフトマッチングスコアを計算

        Args:
            pred: 予測スペクトル [batch_size, spectrum_dim]
            target: 正解スペクトル [batch_size, spectrum_dim]
            precursor_mz_diff: プリカーサーm/z差 [batch_size]
            tolerance: 許容誤差

        Returns:
            シフトマッチングスコア
        """
        batch_size, spectrum_dim = pred.shape
        device = pred.device

        # m/z値の配列を作成 (0, 1, 2, ..., spectrum_dim-1)
        mz_values = torch.arange(spectrum_dim, device=device).float()

        scores = []
        for i in range(batch_size):
            # シフト量（プリカーサー質量差）
            shift = precursor_mz_diff[i].item()
            shift_bins = int(round(shift))

            # シフトされた予測スペクトル
            if shift_bins > 0:
                # 正のシフト：スペクトルを右にシフト
                shifted_pred = torch.cat([
                    torch.zeros(shift_bins, device=device),
                    pred[i, :spectrum_dim - shift_bins]
                ])
            elif shift_bins < 0:
                # 負のシフト：スペクトルを左にシフト
                shifted_pred = torch.cat([
                    pred[i, -shift_bins:],
                    torch.zeros(-shift_bins, device=device)
                ])
            else:
                # シフトなし
                shifted_pred = pred[i]

            # シフトされたスペクトルと正解スペクトルの類似度を計算
            # コサイン類似度を使用
            cos_sim = F.cosine_similarity(
                shifted_pred.unsqueeze(0),
                target[i].unsqueeze(0),
                dim=1
            )
            scores.append(cos_sim)

        # バッチ全体の平均スコア
        return torch.stack(scores).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        precursor_mz_diff: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        損失を計算

        Args:
            pred: 予測スペクトル [batch_size, spectrum_dim]
            target: 正解スペクトル [batch_size, spectrum_dim]
            precursor_mz_diff: プリカーサーm/z差 [batch_size] (オプション)

        Returns:
            損失値（1 - 類似度）
        """
        # 通常のコサイン類似度
        cosine_sim = F.cosine_similarity(pred, target, dim=1).mean()

        # Modified cosine: neutral lossを考慮
        if precursor_mz_diff is not None:
            # プリカーサー質量差でピークをシフトしてマッチング
            shifted_matching = self._compute_shifted_matching(
                pred, target, precursor_mz_diff, self.tolerance
            )
            cosine_sim = (cosine_sim + shifted_matching) / 2

        return 1.0 - cosine_sim  # 損失として返す
