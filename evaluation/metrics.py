import torch
import torch.nn.functional as F
import numpy as np

def weighted_cosine_similarity(pred_intensities, true_intensities, intensity_power=0.5, mz_power=1.3):
    """
    Calculates the weighted cosine similarity between predicted and true spectra.

    Args:
        pred_intensities (torch.Tensor): Predicted spectra, shape [batch_size, max_mz].
        true_intensities (torch.Tensor): Ground truth spectra, shape [batch_size, max_mz].
        intensity_power (float): The power for intensity weighting.
        mz_power (float): The power for m/z weighting.

    Returns:
        torch.Tensor: A tensor of shape [batch_size] containing the similarity score for each spectrum.
    """
    max_mz = pred_intensities.shape[1]
    device = pred_intensities.device

    # Pre-calculate m/z weights
    mz_vals = torch.arange(1, max_mz + 1, dtype=torch.float32, device=device)
    mz_weights = (mz_vals / max_mz) ** mz_power

    def _apply_weights(intensities):
        """Applies intensity and m/z weights."""
        # Normalize intensities first to handle scaling differences
        intensities_norm = F.normalize(intensities, p=2, dim=1)
        weighted_intensities = intensities_norm ** intensity_power
        return weighted_intensities * mz_weights

    pred_weighted = _apply_weights(pred_intensities)
    true_weighted = _apply_weights(true_intensities)

    similarity = F.cosine_similarity(pred_weighted, true_weighted, dim=-1)

    return similarity

def top_n_accuracy(pred_intensities, true_intensities, n=10, tolerance=1):
    """
    Calculates the Top-N peak accuracy.

    This metric measures the proportion of the top N true peaks that are "found"
    within the top N predicted peaks, within a given m/z tolerance.

    Args:
        pred_intensities (torch.Tensor): Predicted spectra, shape [batch_size, max_mz].
        true_intensities (torch.Tensor): Ground truth spectra, shape [batch_size, max_mz].
        n (int): The number of top peaks to consider.
        tolerance (int): The m/z tolerance for a peak to be considered a match.

    Returns:
        float: The mean accuracy across the batch.
    """
    accuracies = []
    for i in range(pred_intensities.shape[0]):
        pred_spec = pred_intensities[i]
        true_spec = true_intensities[i]

        # Get the indices (m/z values) of the top N peaks for both predicted and true spectra
        _, top_pred_indices = torch.topk(pred_spec, k=n)
        _, top_true_indices = torch.topk(true_spec, k=n)

        # Sort the indices to make comparison easier
        top_pred_indices, _ = torch.sort(top_pred_indices)
        top_true_indices, _ = torch.sort(top_true_indices)

        matches = 0
        for true_peak_mz in top_true_indices:
            # Check if there is a predicted peak within the tolerance window
            found = torch.any(torch.abs(top_pred_indices - true_peak_mz) <= tolerance)
            if found:
                matches += 1

        accuracies.append(matches / n)

    return np.mean(accuracies)


if __name__ == '__main__':
    # --- Example Usage & Test ---
    print("--- Testing Evaluation Metrics ---")

    batch_size = 4
    num_bins = 1000

    # Create dummy prediction and target tensors
    pred_spec = torch.rand(batch_size, num_bins) * 100
    true_spec = torch.rand(batch_size, num_bins) * 100

    # --- Test Weighted Cosine Similarity ---
    print("\n--- Testing weighted_cosine_similarity ---")
    similarity_identical = weighted_cosine_similarity(pred_spec, pred_spec)
    print(f"Similarity for identical spectra (mean): {similarity_identical.mean().item():.4f}")
    assert torch.all(torch.isclose(similarity_identical, torch.tensor(1.0))), "Similarity for identical spectra should be 1."

    similarity_different = weighted_cosine_similarity(pred_spec, true_spec)
    print(f"Similarity for different spectra (mean): {similarity_different.mean().item():.4f}")
    assert torch.all(similarity_different < 1.0), "Similarity for different spectra should be less than 1."

    # --- Test Top-N Accuracy ---
    print("\n--- Testing top_n_accuracy ---")

    # Case 1: Identical spectra should have 100% accuracy
    acc_identical = top_n_accuracy(pred_spec, pred_spec, n=10)
    print(f"Top-10 accuracy for identical spectra: {acc_identical:.2f}")
    assert acc_identical == 1.0, "Accuracy for identical spectra should be 1.0"

    # Case 2: No overlap
    spec_a = torch.zeros_like(pred_spec)
    spec_a[:, :10] = torch.arange(10, 0, -1, dtype=torch.float) # Peaks at m/z 0-9
    spec_b = torch.zeros_like(pred_spec)
    spec_b[:, 50:60] = torch.arange(10, 0, -1, dtype=torch.float) # Peaks at m/z 50-59
    acc_no_overlap = top_n_accuracy(spec_a, spec_b, n=10)
    print(f"Top-10 accuracy for non-overlapping spectra: {acc_no_overlap:.2f}")
    assert acc_no_overlap == 0.0, "Accuracy for non-overlapping spectra should be 0.0"

    # Case 3: Partial overlap
    spec_c = torch.zeros_like(pred_spec)
    spec_c[:, 5:15] = torch.arange(10, 0, -1, dtype=torch.float) # Top 10 peaks at m/z 5-14
    # spec_a has top 10 peaks at m/z 0-9
    # With tolerance=1, peaks 5-9 in spec_a should match peaks 5-9 in spec_c (5 matches)
    acc_partial = top_n_accuracy(spec_a, spec_c, n=10, tolerance=1)
    print(f"Top-10 accuracy for partial overlap (tolerance=1): {acc_partial:.2f}")
    assert acc_partial == 0.6, "Partial overlap accuracy should be 0.6"

    print("\n--- Evaluation metrics test completed successfully! ---")