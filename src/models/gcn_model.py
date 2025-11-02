# src/models/gcn_model.py
"""
Graph Convolutional Networkベースのマススペクトル予測モデル
論文のメインAIの実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, 
    global_mean_pool, global_max_pool, global_add_pool,
    AttentionalAggregation
)
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, List

class GraphConvBlock(nn.Module):
    """グラフ畳み込みブロック"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str = "GCNConv",
        activation: str = "relu",
        batch_norm: bool = True,
        dropout: float = 0.1,
        residual: bool = True
    ):
        super().__init__()
        
        self.residual = residual and (in_channels == out_channels)
        
        # グラフ畳み込み層の選択
        if conv_type == "GCNConv":
            self.conv = GCNConv(in_channels, out_channels)
        elif conv_type == "SAGEConv":
            self.conv = SAGEConv(in_channels, out_channels)
        elif conv_type == "GATConv":
            self.conv = GATConv(in_channels, out_channels, heads=4, concat=False)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        # バッチ正規化
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        
        # 活性化関数
        self.activation = self._get_activation(activation)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def _get_activation(self, activation: str):
        """活性化関数の取得"""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: ノード特徴 [N, in_channels]
            edge_index: エッジインデックス [2, E]
            
        Returns:
            更新されたノード特徴 [N, out_channels]
        """
        identity = x
        
        # グラフ畳み込み
        x = self.conv(x, edge_index)
        
        # バッチ正規化
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        # 活性化関数
        x = self.activation(x)
        
        # ドロップアウト
        if self.dropout is not None:
            x = self.dropout(x)
        
        # 残差接続
        if self.residual:
            x = x + identity
        
        return x


class GCNMassSpecPredictor(nn.Module):
    """
    GCNベースのマススペクトル予測モデル
    論文のFig. 4で説明されているアーキテクチャを実装
    """
    
    def __init__(
        self,
        node_features: int = 44,  # 原子特徴の次元
        edge_features: int = 12,  # 結合特徴の次元
        hidden_dim: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
        spectrum_dim: int = 1000,
        conv_type: str = "GCNConv",
        pooling: str = "attention",
        activation: str = "relu",
        batch_norm: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.spectrum_dim = spectrum_dim
        self.pooling_type = pooling
        
        # 入力埋め込み層
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # グラフ畳み込み層（論文のConvolution部分）
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                GraphConvBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    conv_type=conv_type,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    residual=residual
                )
            )
        
        # グラフプーリング層（論文のPooling部分）
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            nn_transform = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.pool = AttentionalAggregation(gate_nn, nn_transform)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # マススペクトル予測ヘッド
        self.spectrum_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, spectrum_dim),
            nn.Sigmoid()  # 強度は0-1の範囲
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_graph_features(self, data: Data) -> torch.Tensor:
        """
        事前学習用：グラフレベル表現のみを抽出

        Args:
            data: PyG Data object
                - x: ノード特徴 [N, node_features]
                - edge_index: エッジインデックス [2, E]
                - batch: バッチインデックス [N]

        Returns:
            グラフレベル特徴量 [batch_size, hidden_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 入力埋め込み
        x = self.node_embedding(x)

        # グラフ畳み込み
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)

        # グラフプーリング
        if self.pooling_type == "attention":
            graph_features = self.pool(x, batch)
        else:
            graph_features = self.pool(x, batch)

        return graph_features

    def forward(
        self,
        data: Data
    ) -> torch.Tensor:
        """
        Forward pass (EI-MS予測用：完全なパイプライン)

        Args:
            data: PyG Data object
                - x: ノード特徴 [N, node_features]
                - edge_index: エッジインデックス [2, E]
                - edge_attr: エッジ特徴 [E, edge_features]
                - batch: バッチインデックス [N]

        Returns:
            予測マススペクトル [batch_size, spectrum_dim]
        """
        # グラフ特徴量を抽出
        graph_features = self.extract_graph_features(data)

        # マススペクトル予測
        spectrum = self.spectrum_predictor(graph_features)

        return spectrum
    
    def predict_fragments(
        self,
        data: Data,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        部分構造情報を含む詳細な予測
        
        Args:
            data: PyG Data object
            return_attention: アテンションスコアを返すか
            
        Returns:
            spectrum: 予測マススペクトル
            attention: アテンションスコア (optional)
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        
        # 入力埋め込み
        x = self.node_embedding(x)
        
        # グラフ畳み込みで特徴抽出
        node_features_list = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
            node_features_list.append(x)
        
        # プーリング
        if self.pooling_type == "attention" and return_attention:
            graph_features, attention_weights = self.pool(x, batch, return_attention=True)
        else:
            graph_features = self.pool(x, batch) if self.pooling_type != "attention" else self.pool(x, batch)
            attention_weights = None
        
        # スペクトル予測
        spectrum = self.spectrum_predictor(graph_features)
        
        if return_attention:
            return spectrum, attention_weights
        return spectrum, None


class PretrainHead(nn.Module):
    """
    事前学習用のヘッドモデル
    量子化学的性質（HOMO-LUMO gap）を予測
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        """
        Args:
            hidden_dim: 入力特徴量の次元
            dropout: ドロップアウト率
        """
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # HOMO-LUMO gap
        )

    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            graph_features: グラフレベル特徴量 [batch_size, hidden_dim]

        Returns:
            予測HOMO-LUMO gap [batch_size, 1]
        """
        return self.predictor(graph_features)


class MultiTaskPretrainHead(nn.Module):
    """
    マルチタスク事前学習用のヘッドモデル
    複数の量子化学的性質を同時予測
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        """
        Args:
            hidden_dim: 入力特徴量の次元
            dropout: ドロップアウト率
        """
        super().__init__()

        # 共通の特徴抽出層
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # タスク固有のヘッド
        self.homo_lumo_head = nn.Linear(hidden_dim, 1)  # HOMO-LUMO gap
        self.dipole_head = nn.Linear(hidden_dim, 3)      # 双極子モーメント (x, y, z)
        self.energy_head = nn.Linear(hidden_dim, 1)      # 全エネルギー

    def forward(
        self,
        graph_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            graph_features: グラフレベル特徴量 [batch_size, hidden_dim]

        Returns:
            homo_lumo: 予測HOMO-LUMO gap [batch_size, 1]
            dipole: 予測双極子モーメント [batch_size, 3]
            energy: 予測全エネルギー [batch_size, 1]
        """
        # 共通の特徴抽出
        shared_features = self.shared_layers(graph_features)

        # 各タスクの予測
        homo_lumo = self.homo_lumo_head(shared_features)
        dipole = self.dipole_head(shared_features)
        energy = self.energy_head(shared_features)

        return homo_lumo, dipole, energy


class GCNEnsemble(nn.Module):
    """複数のGCNモデルのアンサンブル"""

    def __init__(
        self,
        models: List[GCNMassSpecPredictor],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models)
            self.weights = weights

    def forward(self, data: Data) -> torch.Tensor:
        """アンサンブル予測"""
        predictions = []

        for model in self.models:
            with torch.no_grad():
                pred = model(data)
            predictions.append(pred)

        # 重み付き平均
        ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))
        return ensemble_pred


if __name__ == "__main__":
    # モデルのテスト
    print("Testing GCNMassSpecPredictor...")
    
    # ダミーデータ
    x = torch.randn(10, 44)  # 10ノード、44次元特徴
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 12)
    batch = torch.zeros(10, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    # モデル初期化
    model = GCNMassSpecPredictor(
        node_features=44,
        hidden_dim=256,
        num_layers=5,
        spectrum_dim=1000
    )
    
    # 順伝播
    spectrum = model(data)
    print(f"Input: {x.shape}")
    print(f"Output spectrum: {spectrum.shape}")
    print(f"Spectrum range: [{spectrum.min():.4f}, {spectrum.max():.4f}]")
    print("\nModel test passed!")
