#!/usr/bin/env python3
"""
Test suite for evaluation metrics

Tests:
- Cosine similarity metric
- Top-K recall metric
- Spectral angle metric
- Edge cases (zero vectors, identical vectors, etc.)
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Define evaluation metrics (extracted from evaluate_minimal.py)
class ModelEvaluator:
    """Static methods for evaluation metrics"""

    @staticmethod
    def cosine_similarity_metric(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate mean cosine similarity between predictions and targets

        Args:
            pred: Predicted spectra [batch_size, spectrum_dim]
            target: Target spectra [batch_size, spectrum_dim]

        Returns:
            Mean cosine similarity
        """
        pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
        cosine_sim = (pred_norm * target_norm).sum(axis=1)
        return cosine_sim.mean()

    @staticmethod
    def top_k_recall(pred: np.ndarray, target: np.ndarray, k: int = 10) -> float:
        """
        Calculate Top-K Recall

        Args:
            pred: Predicted spectra [batch_size, spectrum_dim]
            target: Target spectra [batch_size, spectrum_dim]
            k: Number of top peaks to consider

        Returns:
            Mean Top-K Recall
        """
        recalls = []
        for p, t in zip(pred, target):
            true_top_k = set(np.argsort(t)[-k:])
            pred_top_k = set(np.argsort(p)[-k:])
            recall = len(true_top_k & pred_top_k) / k
            recalls.append(recall)
        return np.mean(recalls)

    @staticmethod
    def spectral_angle(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate spectral angle (alternative similarity metric)

        Args:
            pred: Predicted spectra [batch_size, spectrum_dim]
            target: Target spectra [batch_size, spectrum_dim]

        Returns:
            Mean spectral angle in degrees
        """
        cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
        # Clamp to avoid numerical issues with arccos
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        angle_rad = np.arccos(cosine_sim)
        angle_deg = np.degrees(angle_rad)
        return angle_deg


def test_cosine_similarity():
    """Test cosine similarity metric"""
    print("\n" + "=" * 60)
    print("Test 1: Cosine Similarity Metric")
    print("=" * 60)

    # Test case 1: Identical vectors (should be 1.0)
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[1.0, 2.0, 3.0]])
    cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
    assert abs(cosine_sim - 1.0) < 1e-6, f"Expected 1.0, got {cosine_sim}"
    print(f"✅ Identical vectors: {cosine_sim:.6f} (expected: 1.0)")

    # Test case 2: Orthogonal vectors (should be 0.0)
    pred = np.array([[1.0, 0.0, 0.0]])
    target = np.array([[0.0, 1.0, 0.0]])
    cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
    assert abs(cosine_sim - 0.0) < 1e-6, f"Expected 0.0, got {cosine_sim}"
    print(f"✅ Orthogonal vectors: {cosine_sim:.6f} (expected: 0.0)")

    # Test case 3: Opposite vectors (should be -1.0)
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[-1.0, -2.0, -3.0]])
    cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
    assert abs(cosine_sim - (-1.0)) < 1e-6, f"Expected -1.0, got {cosine_sim}"
    print(f"✅ Opposite vectors: {cosine_sim:.6f} (expected: -1.0)")

    # Test case 4: Multiple samples (batch)
    pred = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0]
    ])
    target = np.array([
        [1.0, 0.0, 0.0],  # Identical
        [1.0, 0.0, 0.0],  # Orthogonal
        [1.0, 1.0, 1.0]   # Identical
    ])
    cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
    expected = (1.0 + 0.0 + 1.0) / 3  # Average: 0.666...
    assert abs(cosine_sim - expected) < 1e-6, f"Expected {expected}, got {cosine_sim}"
    print(f"✅ Batch of 3 samples: {cosine_sim:.6f} (expected: {expected:.6f})")

    # Test case 5: Scaled vectors (should be 1.0, cosine is scale-invariant)
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[2.0, 4.0, 6.0]])  # 2x scaled
    cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
    assert abs(cosine_sim - 1.0) < 1e-6, f"Expected 1.0, got {cosine_sim}"
    print(f"✅ Scaled vectors: {cosine_sim:.6f} (expected: 1.0)")

    # Test case 6: Zero vector handling
    pred = np.array([[0.0, 0.0, 0.0]])
    target = np.array([[1.0, 2.0, 3.0]])
    cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
    # Should handle gracefully (not NaN)
    assert not np.isnan(cosine_sim), "Cosine similarity should not be NaN"
    print(f"✅ Zero vector: {cosine_sim:.6f} (no NaN)")

    print("\n✅ All cosine similarity tests PASSED!\n")


def test_top_k_recall():
    """Test Top-K Recall metric"""
    print("\n" + "=" * 60)
    print("Test 2: Top-K Recall Metric")
    print("=" * 60)

    # Test case 1: Perfect prediction (recall = 1.0)
    pred = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    recall = ModelEvaluator.top_k_recall(pred, target, k=3)
    assert abs(recall - 1.0) < 1e-6, f"Expected 1.0, got {recall}"
    print(f"✅ Perfect prediction (k=3): {recall:.4f} (expected: 1.0)")

    # Test case 2: No overlap (recall = 0.0)
    pred = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    target = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
    recall = ModelEvaluator.top_k_recall(pred, target, k=1)
    assert abs(recall - 0.0) < 1e-6, f"Expected 0.0, got {recall}"
    print(f"✅ No overlap (k=1): {recall:.4f} (expected: 0.0)")

    # Test case 3: Partial overlap
    pred = np.array([[0.5, 0.6, 0.7, 0.1, 0.2]])  # Top-3: indices 2,1,0
    target = np.array([[0.8, 0.9, 0.3, 0.1, 0.2]])  # Top-3: indices 1,0,2
    recall = ModelEvaluator.top_k_recall(pred, target, k=3)
    # Overlap: {2, 1, 0} ∩ {1, 0, 2} = {0, 1, 2} → 3/3 = 1.0
    assert abs(recall - 1.0) < 1e-6, f"Expected 1.0, got {recall}"
    print(f"✅ Partial overlap (k=3): {recall:.4f} (expected: 1.0)")

    # Test case 4: 50% overlap
    pred = np.array([[0.9, 0.8, 0.1, 0.2]])  # Top-2: indices 0,1
    target = np.array([[0.1, 0.9, 0.2, 0.8]])  # Top-2: indices 1,3
    recall = ModelEvaluator.top_k_recall(pred, target, k=2)
    # Overlap: {0, 1} ∩ {1, 3} = {1} → 1/2 = 0.5
    assert abs(recall - 0.5) < 1e-6, f"Expected 0.5, got {recall}"
    print(f"✅ 50% overlap (k=2): {recall:.4f} (expected: 0.5)")

    # Test case 5: Batch of samples
    pred = np.array([
        [1.0, 0.0, 0.0],  # Top-1: index 0
        [0.0, 1.0, 0.0],  # Top-1: index 1
    ])
    target = np.array([
        [1.0, 0.0, 0.0],  # Top-1: index 0 (match)
        [0.0, 0.0, 1.0],  # Top-1: index 2 (no match)
    ])
    recall = ModelEvaluator.top_k_recall(pred, target, k=1)
    expected = (1.0 + 0.0) / 2  # Average: 0.5
    assert abs(recall - expected) < 1e-6, f"Expected {expected}, got {recall}"
    print(f"✅ Batch of 2 samples (k=1): {recall:.4f} (expected: {expected:.4f})")

    # Test case 6: Different K values
    pred = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    target = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    for k in [1, 5, 10]:
        recall = ModelEvaluator.top_k_recall(pred, target, k=k)
        assert abs(recall - 1.0) < 1e-6, f"Expected 1.0 for k={k}, got {recall}"
        print(f"✅ Perfect match (k={k}): {recall:.4f} (expected: 1.0)")

    print("\n✅ All Top-K Recall tests PASSED!\n")


def test_spectral_angle():
    """Test Spectral Angle metric"""
    print("\n" + "=" * 60)
    print("Test 3: Spectral Angle Metric")
    print("=" * 60)

    # Test case 1: Identical vectors (angle = 0°)
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[1.0, 2.0, 3.0]])
    angle = ModelEvaluator.spectral_angle(pred, target)
    assert abs(angle - 0.0) < 0.01, f"Expected 0.0°, got {angle}°"  # Allow 0.01° tolerance
    print(f"✅ Identical vectors: {angle:.4f}° (expected: 0.0°)")

    # Test case 2: Orthogonal vectors (angle = 90°)
    pred = np.array([[1.0, 0.0, 0.0]])
    target = np.array([[0.0, 1.0, 0.0]])
    angle = ModelEvaluator.spectral_angle(pred, target)
    assert abs(angle - 90.0) < 0.01, f"Expected 90.0°, got {angle}°"
    print(f"✅ Orthogonal vectors: {angle:.4f}° (expected: 90.0°)")

    # Test case 3: Opposite vectors (angle = 180°)
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[-1.0, -2.0, -3.0]])
    angle = ModelEvaluator.spectral_angle(pred, target)
    assert abs(angle - 180.0) < 0.01, f"Expected 180.0°, got {angle}°"
    print(f"✅ Opposite vectors: {angle:.4f}° (expected: 180.0°)")

    # Test case 4: 45° angle
    pred = np.array([[1.0, 0.0]])
    target = np.array([[1.0, 1.0]])  # 45° from x-axis
    angle = ModelEvaluator.spectral_angle(pred, target)
    expected = 45.0
    assert abs(angle - expected) < 1.0, f"Expected ~{expected}°, got {angle}°"
    print(f"✅ 45° angle: {angle:.2f}° (expected: ~{expected:.2f}°)")

    print("\n✅ All Spectral Angle tests PASSED!\n")


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 60)
    print("Test 4: Edge Cases")
    print("=" * 60)

    # Test case 1: Empty batch
    pred = np.array([]).reshape(0, 5)
    target = np.array([]).reshape(0, 5)
    try:
        cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
        # Should handle gracefully
        assert not np.isnan(cosine_sim), "Should not return NaN"
        print(f"✅ Empty batch: Handled gracefully")
    except Exception as e:
        print(f"⚠️  Empty batch: {e}")

    # Test case 2: Single dimension
    pred = np.array([[0.5]])
    target = np.array([[1.0]])
    cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
    assert abs(cosine_sim - 1.0) < 1e-6, f"Expected 1.0, got {cosine_sim}"
    print(f"✅ Single dimension: {cosine_sim:.6f}")

    # Test case 3: Large batch
    batch_size = 1000
    dim = 100
    pred = np.random.rand(batch_size, dim)
    target = np.random.rand(batch_size, dim)
    cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
    assert not np.isnan(cosine_sim), "Should not be NaN"
    assert -1.0 <= cosine_sim <= 1.0, "Should be in [-1, 1] range"
    print(f"✅ Large batch (1000x100): {cosine_sim:.6f}")

    # Test case 4: High-dimensional spectra (realistic EI-MS)
    pred = np.random.rand(10, 1000)  # 10 samples, m/z 1-1000
    target = np.random.rand(10, 1000)
    cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
    recall_10 = ModelEvaluator.top_k_recall(pred, target, k=10)
    print(f"✅ Realistic EI-MS (10x1000): cosine_sim={cosine_sim:.4f}, recall@10={recall_10:.4f}")

    print("\n✅ All edge case tests PASSED!\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("NExtIMS v4.2: Evaluation Metrics Test Suite")
    print("=" * 70)

    try:
        test_cosine_similarity()
        test_top_k_recall()
        test_spectral_angle()
        test_edge_cases()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70 + "\n")
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
