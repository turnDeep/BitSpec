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
