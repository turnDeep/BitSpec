#!/usr/bin/env python3
"""
官能基特徴量統合のテストスクリプト
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from rdkit import Chem
from src.data.features import MolecularFeaturizer, SubstructureFeaturizer
from src.models.gcn_model import GCNMassSpecPredictor
from torch_geometric.data import Data, Batch

def test_substructure_featurizer():
    """SubstructureFeaturizerのテスト"""
    print("=" * 60)
    print("Testing SubstructureFeaturizer...")
    print("=" * 60)

    featurizer = SubstructureFeaturizer()

    # テスト分子: ベンゼン、トルエン、フェノール
    test_smiles = [
        ("Benzene", "c1ccccc1"),
        ("Toluene", "Cc1ccccc1"),
        ("Phenol", "Oc1ccccc1"),
        ("Acetone", "CC(=O)C"),
        ("Ethanol", "CCO")
    ]

    for name, smiles in test_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  ✗ {name}: Invalid SMILES")
            continue

        fingerprint = featurizer.get_substructure_fingerprint(mol)
        num_matches = int(fingerprint.sum())

        print(f"  ✓ {name} ({smiles}): {num_matches} functional groups detected")
        print(f"    Fingerprint shape: {fingerprint.shape}")

    print()

def test_molecular_featurizer_with_functional_groups():
    """MolecularFeaturizerとSubstructureFeaturizerの統合テスト"""
    print("=" * 60)
    print("Testing MolecularFeaturizer + SubstructureFeaturizer...")
    print("=" * 60)

    mol_featurizer = MolecularFeaturizer()
    sub_featurizer = SubstructureFeaturizer()

    # テスト分子
    smiles = "Oc1ccccc1"  # フェノール
    mol = Chem.MolFromSmiles(smiles)

    # グラフデータ生成
    graph_data = mol_featurizer.mol_to_graph(mol)

    # 官能基フィンガープリント追加
    fg_fingerprint = sub_featurizer.get_substructure_fingerprint(mol)
    graph_data.functional_groups = torch.tensor(fg_fingerprint, dtype=torch.float32)

    print(f"  ✓ Graph data created for phenol")
    print(f"    Node features shape: {graph_data.x.shape}")
    print(f"    Edge features shape: {graph_data.edge_attr.shape}")
    print(f"    Functional groups shape: {graph_data.functional_groups.shape}")
    print(f"    Functional groups detected: {int(graph_data.functional_groups.sum())}")
    print()

def test_gcn_model_with_functional_groups():
    """GCNモデルでの官能基特徴量使用テスト"""
    print("=" * 60)
    print("Testing GCNMassSpecPredictor with functional groups...")
    print("=" * 60)

    # ダミーデータ作成
    batch_size = 4
    num_nodes_per_graph = 10

    # グラフデータ作成
    graphs = []
    for i in range(batch_size):
        x = torch.randn(num_nodes_per_graph, 48)  # ノード特徴
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 6)  # エッジ特徴
        functional_groups = torch.rand(48)  # 官能基フィンガープリント (0-1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.functional_groups = functional_groups
        graphs.append(data)

    # バッチ化
    batch = Batch.from_data_list(graphs)

    print(f"  Batch created:")
    print(f"    Batch size: {batch_size}")
    print(f"    Total nodes: {batch.x.shape[0]}")
    print(f"    Functional groups shape: {batch.functional_groups.shape}")

    # モデル作成（官能基使用）
    model_with_fg = GCNMassSpecPredictor(
        node_features=48,
        edge_features=6,
        hidden_dim=256,
        num_layers=3,
        spectrum_dim=1000,
        dropout=0.1,
        pooling="mean",
        use_functional_groups=True,
        num_functional_groups=48
    )

    # モデル作成（官能基不使用）
    model_without_fg = GCNMassSpecPredictor(
        node_features=48,
        edge_features=6,
        hidden_dim=256,
        num_layers=3,
        spectrum_dim=1000,
        dropout=0.1,
        pooling="mean",
        use_functional_groups=False,
        num_functional_groups=48
    )

    model_with_fg.eval()
    model_without_fg.eval()

    with torch.no_grad():
        # 官能基ありで予測
        output_with_fg = model_with_fg(batch)

        # 官能基なしで予測
        output_without_fg = model_without_fg(batch)

    print(f"\n  ✓ Model with functional groups:")
    print(f"    Input: graph features (256) + functional groups (48)")
    print(f"    Output shape: {output_with_fg.shape}")
    print(f"    Output range: [{output_with_fg.min():.4f}, {output_with_fg.max():.4f}]")

    print(f"\n  ✓ Model without functional groups:")
    print(f"    Input: graph features (256) only")
    print(f"    Output shape: {output_without_fg.shape}")
    print(f"    Output range: [{output_without_fg.min():.4f}, {output_without_fg.max():.4f}]")

    # パラメータ数の比較
    params_with_fg = sum(p.numel() for p in model_with_fg.parameters())
    params_without_fg = sum(p.numel() for p in model_without_fg.parameters())

    print(f"\n  Model parameters:")
    print(f"    With functional groups: {params_with_fg:,}")
    print(f"    Without functional groups: {params_without_fg:,}")
    print(f"    Difference: {params_with_fg - params_without_fg:,} ({(params_with_fg - params_without_fg) / params_without_fg * 100:.2f}%)")
    print()

def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("  Functional Groups Feature Integration Test")
    print("=" * 60 + "\n")

    try:
        # テスト1: SubstructureFeaturizer
        test_substructure_featurizer()

        # テスト2: MolecularFeaturizer + SubstructureFeaturizer
        test_molecular_featurizer_with_functional_groups()

        # テスト3: GCNモデル
        test_gcn_model_with_functional_groups()

        print("=" * 60)
        print("  ✓ All tests passed successfully!")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
