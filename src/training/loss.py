# src/training/loss.py
"""
損失関数の定義
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectrumLoss(nn.Module):
    """マススペクトル予測用の損失関数"""
    
    def __init__(
        self,
        loss_type: str = "cosine",
        alpha: float = 0.5,
        beta: float = 0.5
    ):
        """
        Args:
            loss_type: 損失関数のタイプ
                - "cosine": コサイン類似度損失
                - "mse": 平均二乗誤差
                - "mae": 平均絶対誤差
                - "combined": 複数の損失の組み合わせ
            alpha: MSE損失の重み（combinedの場合）
            beta: コサイン損失の重み（combinedの場合）
        """
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        
    def cosine_similarity_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        コサイン類似度損失
        論文で使用されているメイン指標
        
        Args:
            pred: 予測スペクトル [batch_size, spectrum_dim]
            target: 正解スペクトル [batch_size, spectrum_dim]
            
        Returns:
            損失値
        """
        # コサイン類似度を計算
        cos_sim = F.cosine_similarity(pred, target, dim=1)
        
        # 損失は 1 - cos_sim
        loss = 1.0 - cos_sim.mean()
        
        return loss
    
    def weighted_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_factor: float = 2.0
    ) -> torch.Tensor:
        """
        重み付きMSE損失
        強度の高いピークにより大きな重みを付ける
        
        Args:
            pred: 予測スペクトル
            target: 正解スペクトル
            weight_factor: 重み付け係数
            
        Returns:
            損失値
        """
        # 強度に基づいて重みを計算
        weights = 1.0 + (target * weight_factor)
        
        # 重み付きMSE
        loss = (weights * (pred - target) ** 2).mean()
        
        return loss
    
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
            損失値
        """
        if self.loss_type == "cosine":
            return self.cosine_similarity_loss(pred, target)
        
        elif self.loss_type == "mse":
            return F.mse_loss(pred, target)
        
        elif self.loss_type == "mae":
            return F.l1_loss(pred, target)
        
        elif self.loss_type == "weighted_mse":
            return self.weighted_mse_loss(pred, target)
        
        elif self.loss_type == "combined":
            # 複数の損失を組み合わせ
            mse_loss = F.mse_loss(pred, target)
            cosine_loss = self.cosine_similarity_loss(pred, target)
            
            return self.alpha * mse_loss + self.beta * cosine_loss
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class CombinedSpectrumLoss(nn.Module):
    """複数の損失関数を組み合わせたマススペクトル損失"""

    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        kl_weight: float = 0.1
    ):
        """
        Args:
            mse_weight: MSE損失の重み
            cosine_weight: コサイン類似度損失の重み
            kl_weight: KLダイバージェンス損失の重み
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.kl_weight = kl_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        損失を計算

        Args:
            pred: 予測スペクトル [batch_size, spectrum_dim]
            target: 正解スペクトル [batch_size, spectrum_dim]

        Returns:
            損失値
        """
        # MSE損失
        mse_loss = F.mse_loss(pred, target)

        # コサイン類似度損失
        cos_sim = F.cosine_similarity(pred, target, dim=1)
        cosine_loss = 1.0 - cos_sim.mean()

        # KLダイバージェンス損失（分布の類似性）
        # 安定性のために小さな値を足す
        eps = 1e-10
        pred_norm = pred + eps
        target_norm = target + eps
        kl_loss = F.kl_div(pred_norm.log(), target_norm, reduction='batchmean')

        # 複合損失
        total_loss = (
            self.mse_weight * mse_loss +
            self.cosine_weight * cosine_loss +
            self.kl_weight * kl_loss
        )

        return total_loss


class FocalLoss(nn.Module):
    """Focal Loss（難しいサンプルに注目）"""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測スペクトル
            target: 正解スペクトル

        Returns:
            損失値
        """
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


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


if __name__ == "__main__":
    # テスト
    print("Testing SpectrumLoss...")
    
    # ダミーデータ
    batch_size = 8
    spectrum_dim = 1000
    
    pred = torch.randn(batch_size, spectrum_dim).sigmoid()
    target = torch.randn(batch_size, spectrum_dim).sigmoid()
    
    # 各損失関数をテスト
    for loss_type in ["cosine", "mse", "mae", "weighted_mse", "combined"]:
        criterion = SpectrumLoss(loss_type=loss_type)
        loss = criterion(pred, target)
        print(f"{loss_type} loss: {loss.item():.4f}")
    
    print("\nLoss test passed!")
