import torch
import torch.nn.functional as F

class EvaluationMetrics:
    """
    A class to compute various evaluation metrics for mass spectra prediction.
    """
    def __init__(self, at_k=[10, 20, 50]):
        self.at_k = at_k

    def _bin_spectrum(self, mz, intensity, num_bins=1000, max_mz=1000.0):
        """
        A simplified function to bin a sparse spectrum into a dense vector.
        This should be consistent with the one used in the loss function.
        """
        binned_spectrum = torch.zeros(mz.size(0), num_bins, device=mz.device)
        bin_indices = (mz / max_mz * (num_bins - 1)).long()
        for i in range(mz.size(0)):
            binned_spectrum[i].scatter_add_(0, bin_indices[i], intensity[i])
        return binned_spectrum

    def calculate_all(self, pred_mz, pred_intensity, true_mz, true_intensity):
        """
        Calculates all defined metrics.

        Args:
            pred_mz, pred_intensity: The predicted spectrum.
            true_mz, true_intensity: The ground truth spectrum.
        """
        # For simplicity, we assume inputs are single spectra (not batches)
        # and already on the correct device.

        # Step 1: Bin spectra to a common m/z axis
        pred_binned = self._bin_spectrum(pred_mz.unsqueeze(0), pred_intensity.unsqueeze(0))
        true_binned = self._bin_spectrum(true_mz.unsqueeze(0), true_intensity.unsqueeze(0))

        # Step 2: Calculate metrics
        cosine_sim = self.cosine_similarity(pred_binned, true_binned)
        prec, recall, f1 = self.precision_recall_f1(pred_binned, true_binned, self.at_k)
        mae = self.mean_absolute_error(pred_binned, true_binned)

        return {
            'cosine_similarity': cosine_sim.item(),
            'precision_at_k': prec,
            'recall_at_k': recall,
            'f1_at_k': f1,
            'mae': mae.item()
        }

    def cosine_similarity(self, pred_binned, true_binned):
        return F.cosine_similarity(pred_binned, true_binned).mean()

    def precision_recall_f1(self, pred_binned, true_binned, at_k_list):
        # Placeholder logic
        # A real implementation would identify top-k peaks and compare
        precisions, recalls, f1s = {}, {}, {}
        for k in at_k_list:
            precisions[f'@{k}'] = 0.9 # Dummy value
            recalls[f'@{k}'] = 0.8    # Dummy value
            f1s[f'@{k}'] = 0.85       # Dummy value
        return precisions, recalls, f1s

    def mean_absolute_error(self, pred_binned, true_binned):
        # Only calculate MAE on peaks that exist in the true spectrum
        mask = true_binned > 0
        if torch.sum(mask) > 0:
            return (pred_binned[mask] - true_binned[mask]).abs().mean()
        return torch.tensor(0.0)
