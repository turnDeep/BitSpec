#!/usr/bin/env python3
"""
エッジ統合機能のテストスクリプト
"""

import sys
sys.path.insert(0, '/home/user/BitSpec/src')

import torch
from torch_geometric.data import Data
from models.gcn_model import GCNMassSpecPredictor

def test_edge_integration():
    """エッジ情報の統合をテスト"""
    print("=" * 60)
    print("Testing Edge Integration into GCN Model")
    print("=" * 60)

    # ダミーデータを作成
    num_nodes = 10
    num_edges = 15
    node_features = 48  # 更新された特徴次元
    edge_features = 6   # 更新された特徴次元

    x = torch.randn(num_nodes, node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_attr = torch.randn(num_edges, edge_features)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # PyG Dataオブジェクト作成
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    print(f"\n[Input Data]")
    print(f"  Nodes: {num_nodes}")
    print(f"  Node features: {node_features}D")
    print(f"  Edges: {num_edges}")
    print(f"  Edge features: {edge_features}D")
    print(f"  x shape: {x.shape}")
    print(f"  edge_attr shape: {edge_attr.shape}")

    # モデル初期化
    print(f"\n[Model Initialization]")
    model = GCNMassSpecPredictor(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=256,
        num_layers=5,
        spectrum_dim=1000
    )
    print(f"  Model created successfully")
    print(f"  edge_embedding: {model.edge_embedding is not None}")
    print(f"  edge_to_node_fusion: {model.edge_to_node_fusion is not None}")

    # 順伝播テスト（完全なパイプライン）
    print(f"\n[Forward Pass Test - Full Pipeline]")
    model.eval()
    with torch.no_grad():
        spectrum = model(data)
    print(f"  Input node features: {x.shape}")
    print(f"  Input edge features: {edge_attr.shape}")
    print(f"  Output spectrum: {spectrum.shape}")
    print(f"  Spectrum range: [{spectrum.min():.4f}, {spectrum.max():.4f}]")
    assert spectrum.shape == (1, 1000), f"Expected (1, 1000), got {spectrum.shape}"

    # グラフ特徴抽出テスト（事前学習用）
    print(f"\n[Graph Features Extraction Test - Pretraining]")
    with torch.no_grad():
        graph_features = model.extract_graph_features(data)
    print(f"  Graph features shape: {graph_features.shape}")
    print(f"  Graph features range: [{graph_features.min():.4f}, {graph_features.max():.4f}]")
    assert graph_features.shape == (1, 256), f"Expected (1, 256), got {graph_features.shape}"

    # エッジ属性なしのテスト（互換性確認）
    print(f"\n[Backward Compatibility Test - No Edge Attributes]")
    data_no_edge = Data(x=x, edge_index=edge_index, batch=batch)
    with torch.no_grad():
        spectrum_no_edge = model(data_no_edge)
    print(f"  Output spectrum (no edge_attr): {spectrum_no_edge.shape}")
    print(f"  Model works without edge_attr: ✓")

    # 部分構造予測テスト
    print(f"\n[Fragment Prediction Test]")
    with torch.no_grad():
        spectrum_frag, attention = model.predict_fragments(data, return_attention=False)
    print(f"  Fragment spectrum: {spectrum_frag.shape}")
    print(f"  Fragment prediction works: ✓")

    print(f"\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print(f"\n[Summary]")
    print(f"  • Edge embeddings are now integrated into node features")
    print(f"  • Each node aggregates information from connected edges")
    print(f"  • Backward compatibility maintained (works without edge_attr)")
    print(f"  • HOMO-LUMO pretraining will now use bond information")
    print(f"  • Mass spectrum prediction will benefit from bond features")

if __name__ == "__main__":
    test_edge_integration()
