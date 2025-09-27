import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    """
    A hybrid loss function for comparing predicted and true mass spectra,
    based on the design specification.

    It combines:
    1. Cosine similarity for overall shape.
    2. Peak existence loss (BCE).
    3. Peak intensity loss (Huber).
    4. A penalty for physics violations (handled separately).
    """
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # Weight for cosine loss
        self.beta = beta    # Weight for peak existence loss
        self.gamma = gamma  # Weight for intensity loss

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.huber_loss = nn.HuberLoss()

    def forward(self, pred_mz, pred_intensity, true_mz, true_intensity):
        """
        Calculates the hybrid loss.

        This is a complex operation because predicted and true spectra do not
        have aligned m/z bins. A common technique is to "bin" both spectra into
        a fixed-size vector before comparing.

        Args:
            pred_mz (Tensor): Predicted m/z values.
            pred_intensity (Tensor): Predicted intensities.
            true_mz (Tensor): Ground truth m/z values.
            true_intensity (Tensor): Ground truth intensities.
        """

        # --- This is a simplified placeholder implementation ---
        # A full implementation requires binning the spectra into a common m/z axis.

        # Assume spectra are pre-binned to a fixed vector of size N for simplicity.
        # pred_binned and true_binned would be tensors of shape (batch_size, num_bins).

        # Let's simulate some binned spectra for placeholder calculation
        num_bins = 1000 # Example bin count
        pred_binned = self.bin_spectrum(pred_mz, pred_intensity, num_bins)
        true_binned = self.bin_spectrum(true_mz, true_intensity, num_bins)

        # 1. Cosine Similarity Loss
        # We want to MAXIMIZE cosine similarity, so we use (1 - sim) as loss.
        # Add a small epsilon to prevent division by zero.
        cos_sim = F.cosine_similarity(pred_binned, true_binned, dim=-1)
        loss_cosine = (1 - cos_sim).mean()

        # 2. Peak Existence Loss (Binary Cross-Entropy)
        # Treat peak existence as a binary classification problem.
        # A "peak exists" if its binned intensity is > 0.
        pred_exists = (pred_binned > 0).float()
        true_exists = (true_binned > 0).float()
        loss_peak = self.bce_loss(pred_exists, true_exists)

        # 3. Intensity Regression Loss (Huber)
        # Only compare intensities where a true peak exists.
        mask = true_exists > 0
        loss_intensity = self.huber_loss(pred_binned[mask], true_binned[mask])

        # Combine losses
        total_loss = (self.alpha * loss_cosine +
                      self.beta * loss_peak +
                      self.gamma * loss_intensity)

        return total_loss

    def bin_spectrum(self, mz, intensity, num_bins, max_mz=1000.0):
        """
        A simplified function to bin a sparse spectrum into a dense vector.
        """
        binned_spectrum = torch.zeros(mz.size(0), num_bins, device=mz.device)

        # Normalize m/z values to bin indices
        bin_indices = (mz / max_mz * (num_bins - 1)).long()

        # Use scatter_add_ to handle multiple peaks falling into the same bin
        # This is a common operation in spectral processing.
        # We need to flatten the batch for scatter_add_ to work easily
        batch_indices = torch.arange(mz.size(0), device=mz.device).unsqueeze(1).expand_as(mz)

        # This operation is complex with batching, so we loop for simplicity.
        for i in range(mz.size(0)):
            binned_spectrum[i].scatter_add_(0, bin_indices[i], intensity[i])

        return binned_spectrum
