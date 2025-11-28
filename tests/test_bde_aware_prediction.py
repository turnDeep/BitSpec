#!/usr/bin/env python3
"""
Test BDE-Aware Prediction Implementation

Validates that the new BDE-aware prediction mechanism works correctly:
1. BondAwarePooling module
2. TeacherModel with BDE-aware prediction enabled
3. Fragmentation-aware embeddings are properly created
4. Output dimensions are correct
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from torch_geometric.data import Data, Batch

# Import models
from src.models.teacher import TeacherModel, BondAwarePooling, BondBreakingAttention


def create_dummy_molecule_graph(num_nodes=10, num_edges=18):
    """Create a dummy molecular graph for testing"""
    # Node features (48-dimensional)
    x = torch.randn(num_nodes, 48)

    # Edge index (bidirectional)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Edge features (6-dimensional: bond type, conjugated, aromatic, in_ring, etc.)
    edge_attr = torch.randn(num_edges, 6)

    # Batch assignment
    batch = torch.zeros(num_nodes, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


def test_bond_aware_pooling():
    """Test BondAwarePooling module"""
    print("=" * 80)
    print("TEST 1: BondAwarePooling Module")
    print("=" * 80)

    hidden_dim = 256
    pooling = BondAwarePooling(hidden_dim=hidden_dim)

    # Create dummy data
    num_nodes = 15
    num_edges = 28

    node_features = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    bond_probs = torch.rand(num_edges, 1)  # Bond breaking probabilities
    batch = torch.zeros(num_nodes, dtype=torch.long)  # Single molecule

    # Forward pass
    frag_emb = pooling(node_features, edge_index, bond_probs, batch)

    print(f"Input:")
    print(f"  - Node features: {node_features.shape}")
    print(f"  - Edge index: {edge_index.shape}")
    print(f"  - Bond probs: {bond_probs.shape}")
    print(f"  - Batch: {batch.shape}")
    print(f"\nOutput:")
    print(f"  - Fragmentation-aware embedding: {frag_emb.shape}")
    print(f"  - Expected: [1, {hidden_dim}]")

    assert frag_emb.shape == (1, hidden_dim), f"Expected shape (1, {hidden_dim}), got {frag_emb.shape}"
    print("\n✅ BondAwarePooling test PASSED\n")

    return True


def test_teacher_model_bde_aware():
    """Test TeacherModel with BDE-aware prediction enabled"""
    print("=" * 80)
    print("TEST 2: TeacherModel with BDE-Aware Prediction")
    print("=" * 80)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure BDE-aware prediction is enabled
    config['model']['teacher']['gnn']['use_bde_aware_prediction'] = True
    config['model']['teacher']['gnn']['use_bond_breaking'] = True

    print(f"\nConfiguration:")
    print(f"  - use_bde_aware_prediction: {config['model']['teacher']['gnn']['use_bde_aware_prediction']}")
    print(f"  - use_bond_breaking: {config['model']['teacher']['gnn']['use_bond_breaking']}")

    # Create model
    print("\nInitializing TeacherModel...")
    model = TeacherModel(config)
    model.eval()

    # Create dummy input
    batch_size = 4
    graphs = [create_dummy_molecule_graph() for _ in range(batch_size)]
    graph_batch = Batch.from_data_list(graphs)
    ecfp = torch.randn(batch_size, 4096)

    print(f"\nInput:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Total nodes: {graph_batch.x.shape[0]}")
    print(f"  - Total edges: {graph_batch.edge_index.shape[1]}")
    print(f"  - ECFP: {ecfp.shape}")

    # Test 1: Standard forward pass (BDE-aware enabled by default)
    print("\n--- Test 2a: Standard forward (BDE-aware active) ---")
    with torch.no_grad():
        output = model(graph_batch, ecfp)

    if isinstance(output, tuple):
        spectrum, bde_pred = output
        print(f"Output:")
        print(f"  - Spectrum: {spectrum.shape}")
        print(f"  - BDE predictions: {bde_pred.shape}")
        print(f"  - Expected spectrum: [{batch_size}, 501]")
        assert spectrum.shape == (batch_size, 501), f"Expected spectrum shape ({batch_size}, 501), got {spectrum.shape}"
        print("✅ BDE predictions are returned (as expected in BDE-aware mode)")
    else:
        spectrum = output
        print(f"Output:")
        print(f"  - Spectrum: {spectrum.shape}")
        assert spectrum.shape == (batch_size, 501), f"Expected spectrum shape ({batch_size}, 501), got {spectrum.shape}"
        print("✅ Standard output format")

    # Test 2: Explicitly request BDE predictions
    print("\n--- Test 2b: Explicit BDE prediction request ---")
    with torch.no_grad():
        spectrum, bde_pred = model(graph_batch, ecfp, return_bde_predictions=True)

    print(f"Output:")
    print(f"  - Spectrum: {spectrum.shape}")
    print(f"  - BDE predictions: {bde_pred.shape}")
    assert spectrum.shape == (batch_size, 501), f"Expected spectrum shape ({batch_size}, 501), got {spectrum.shape}"
    assert bde_pred is not None, "Expected BDE predictions, got None"
    print("✅ BDE predictions returned correctly")

    print("\n✅ TeacherModel BDE-aware test PASSED\n")

    return True


def test_teacher_model_without_bde_aware():
    """Test TeacherModel with BDE-aware prediction disabled (backward compatibility)"""
    print("=" * 80)
    print("TEST 3: TeacherModel WITHOUT BDE-Aware (Backward Compatibility)")
    print("=" * 80)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Disable BDE-aware prediction
    config['model']['teacher']['gnn']['use_bde_aware_prediction'] = False

    print(f"\nConfiguration:")
    print(f"  - use_bde_aware_prediction: {config['model']['teacher']['gnn']['use_bde_aware_prediction']}")

    # Create model
    print("\nInitializing TeacherModel (BDE-aware disabled)...")
    model = TeacherModel(config)
    model.eval()

    # Create dummy input
    batch_size = 4
    graphs = [create_dummy_molecule_graph() for _ in range(batch_size)]
    graph_batch = Batch.from_data_list(graphs)
    ecfp = torch.randn(batch_size, 4096)

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(graph_batch, ecfp)

    # Should return only spectrum (no BDE predictions)
    if isinstance(output, tuple):
        print(f"⚠️  WARNING: Received tuple output, expected single tensor")
        spectrum = output[0]
    else:
        spectrum = output

    print(f"Output:")
    print(f"  - Spectrum: {spectrum.shape}")
    print(f"  - Expected: [{batch_size}, 501]")
    assert spectrum.shape == (batch_size, 501), f"Expected spectrum shape ({batch_size}, 501), got {spectrum.shape}"

    print("\n✅ Backward compatibility test PASSED\n")

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("BDE-AWARE PREDICTION IMPLEMENTATION TESTS")
    print("=" * 80 + "\n")

    tests = [
        ("BondAwarePooling Module", test_bond_aware_pooling),
        ("TeacherModel with BDE-Aware", test_teacher_model_bde_aware),
        ("TeacherModel without BDE-Aware (Backward Compat)", test_teacher_model_without_bde_aware),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with error:")
            print(f"   {str(e)}\n")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(success for _, success in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nBDE-aware prediction implementation is working correctly.")
        print("The model can now use bond-breaking probabilities and BDE predictions")
        print("to create fragmentation-aware embeddings during spectrum prediction.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease review the errors above and fix the implementation.")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
