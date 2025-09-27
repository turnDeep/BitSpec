import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCosineLoss(nn.Module):
    """
    Calculates the weighted cosine similarity loss between predicted and true spectra.

    This loss function is designed to work with dense intensity vectors where the
    index corresponds to the m/z value. This aligns with the output of the preprocessor
    and is a more robust approach than predicting sparse m/z values.

    The weighting scheme is based on the specification document:
    - Intensity is weighted by `intensity ** intensity_power`.
    - The vector is then weighted by `(m/z / max_mz) ** mz_power`.
    """
    def __init__(self, intensity_power=0.5, mz_power=1.3, max_mz=1000):
        """
        Args:
            intensity_power (float): The power to raise intensity values to.
            mz_power (float): The power to raise m/z values to for weighting.
            max_mz (int): The maximum m/z value, defining the length of the intensity vector.
        """
        super().__init__()
        self.intensity_power = intensity_power
        self.mz_power = mz_power
        self.max_mz = max_mz

        # Create a buffer for the m/z weights so it's moved to the correct device automatically.
        # The weights are pre-calculated for efficiency.
        mz_vals = torch.arange(1, max_mz + 1, dtype=torch.float32)
        mz_weights = (mz_vals / max_mz) ** self.mz_power
        self.register_buffer('mz_weights', mz_weights)

    def _apply_weights(self, intensities):
        """Applies intensity and m/z weights to a batch of spectra."""
        # Apply intensity weighting
        weighted_intensities = intensities ** self.intensity_power

        # Apply m/z weighting (broadcasts over the batch dimension)
        return weighted_intensities * self.mz_weights

    def forward(self, pred_intensities, true_intensities):
        """
        Calculates the loss.

        Args:
            pred_intensities (torch.Tensor): Predicted spectra, shape [batch_size, max_mz].
            true_intensities (torch.Tensor): Ground truth spectra, shape [batch_size, max_mz].

        Returns:
            torch.Tensor: The mean loss value (1 - similarity).
        """
        # Ensure inputs are normalized, as cosine similarity is sensitive to magnitude.
        # The model's output (sigmoid * 100) and target (normalized to 100) are already scaled.
        # We can apply an L2 norm for stability, though it's not strictly required if
        # the input scaling is consistent.
        pred_norm = F.normalize(pred_intensities, p=2, dim=1)
        true_norm = F.normalize(true_intensities, p=2, dim=1)

        # Apply the weighting scheme to both predicted and true spectra
        pred_weighted = self._apply_weights(pred_norm)
        true_weighted = self._apply_weights(true_norm)

        # Calculate cosine similarity along the last dimension (the spectrum itself)
        # The result is a tensor of shape [batch_size].
        similarity = F.cosine_similarity(pred_weighted, true_weighted, dim=-1)

        # The loss is 1 minus the similarity. We want to maximize similarity, so we minimize 1 - similarity.
        # We take the mean over the batch.
        loss = 1.0 - similarity.mean()

        return loss

if __name__ == '__main__':
    # --- Example Usage & Test ---
    print("--- Testing WeightedCosineLoss ---")

    # Parameters
    batch_size = 4
    num_bins = 1000

    # Create the loss function
    loss_fn = WeightedCosineLoss()
    print("Loss function:", loss_fn)
    print("m/z weights shape:", loss_fn.mz_weights.shape)

    # Create dummy prediction and target tensors
    # These simulate the output of the model and the preprocessor
    pred_spec = torch.rand(batch_size, num_bins) * 100
    true_spec = torch.rand(batch_size, num_bins) * 100

    # --- Test Case 1: Identical Spectra ---
    # The loss for identical spectra should be close to 0.
    loss_identical = loss_fn(pred_spec, pred_spec)
    print(f"\nLoss for identical spectra: {loss_identical.item():.6f}")
    assert torch.isclose(loss_identical, torch.tensor(0.0), atol=1e-6), "Loss for identical spectra should be 0."

    # --- Test Case 2: Different Spectra ---
    # The loss should be a positive value between 0 and 2.
    loss_different = loss_fn(pred_spec, true_spec)
    print(f"Loss for different spectra: {loss_different.item():.4f}")
    assert loss_different.item() > 0, "Loss for different spectra should be positive."

    # --- Test Case 3: Orthogonal (uncorrelated) Spectra ---
    # Create a spectrum that is zero where the other is non-zero
    spec_a = torch.zeros(batch_size, num_bins)
    spec_a[:, :num_bins//2] = torch.rand(batch_size, num_bins//2)
    spec_b = torch.zeros(batch_size, num_bins)
    spec_b[:, num_bins//2:] = torch.rand(batch_size, num_bins//2)
    loss_orthogonal = loss_fn(spec_a, spec_b)
    # The weighted cosine similarity should be close to 0, so the loss should be close to 1.
    print(f"Loss for orthogonal spectra: {loss_orthogonal.item():.4f}")
    assert torch.isclose(loss_orthogonal, torch.tensor(1.0), atol=1e-4), "Loss for orthogonal spectra should be close to 1."

    print("\n--- WeightedCosineLoss test completed successfully! ---")