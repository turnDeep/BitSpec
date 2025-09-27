import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def plot_spectrum_comparison(pred_intensities, true_intensities, mol_id=None, save_path=None):
    """
    Plots a comparison of a predicted spectrum and a true spectrum.

    Args:
        pred_intensities (torch.Tensor or np.ndarray): 1D array of predicted intensities.
        true_intensities (torch.Tensor or np.ndarray): 1D array of true intensities.
        mol_id (str, optional): The ID of the molecule, for the plot title.
        save_path (str, optional): If provided, saves the plot to this file path.
    """
    if isinstance(pred_intensities, torch.Tensor):
        pred_intensities = pred_intensities.cpu().numpy()
    if isinstance(true_intensities, torch.Tensor):
        true_intensities = true_intensities.cpu().numpy()

    # Normalize for better visual comparison if they aren't already
    if np.max(pred_intensities) > 0:
        pred_intensities = pred_intensities / np.max(pred_intensities) * 100
    if np.max(true_intensities) > 0:
        true_intensities = true_intensities / np.max(true_intensities) * 100

    mz_values = np.arange(1, len(true_intensities) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot true spectrum (ground truth)
    markerline_true, stemlines_true, baseline_true = ax.stem(
        mz_values, true_intensities,
        linefmt='-', markerfmt=' ', basefmt=' ',
        label='True Spectrum'
    )
    plt.setp(stemlines_true, 'color', 'b', 'linewidth', 1.5)

    # Plot predicted spectrum
    markerline_pred, stemlines_pred, baseline_pred = ax.stem(
        mz_values, -pred_intensities, # Plot predicted spectrum downwards
        linefmt='-', markerfmt=' ', basefmt=' ',
        label='Predicted Spectrum'
    )
    plt.setp(stemlines_pred, 'color', 'r', 'linewidth', 1.0)

    # Formatting
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative Intensity")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero') # Set x-axis at y=0
    ax.get_yaxis().set_ticks([-100, -50, 0, 50, 100])
    ax.get_yaxis().set_ticklabels([100, 50, 0, 50, 100])

    title = "Spectrum Comparison"
    if mol_id:
        title += f" for Molecule ID: {mol_id}"
    ax.set_title(title)

    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    # --- Example Usage & Test ---
    print("--- Testing Spectrum Visualizer ---")

    num_bins = 500 # Use a smaller number of bins for a clearer test plot

    # Create a dummy true spectrum with a few distinct peaks
    true_spec = np.zeros(num_bins)
    true_spec[50] = 100
    true_spec[120] = 75
    true_spec[250] = 50
    true_spec[400] = 20

    # Create a dummy predicted spectrum that is similar but not identical
    pred_spec = np.zeros(num_bins)
    pred_spec[52] = 95  # Slightly off m/z, slightly lower intensity
    pred_spec[120] = 80 # Correct m/z, slightly higher intensity
    pred_spec[300] = 40 # Missed peak at 250, new peak at 300
    pred_spec[401] = 25 # Slightly off m/z, slightly higher intensity

    # Add some noise
    pred_spec += np.random.rand(num_bins) * 5

    # Define where to save the test plot
    output_dir = "test_plots"
    output_path = os.path.join(output_dir, "spectrum_comparison_test.png")

    # Generate the plot
    plot_spectrum_comparison(pred_spec, true_spec, mol_id="TEST001", save_path=output_path)

    # Check if the file was created
    assert os.path.exists(output_path), f"Plot file was not created at {output_path}"

    print(f"\nTest plot generated at '{output_path}'. Please inspect it visually.")

    # Clean up the created directory and file
    import shutil
    shutil.rmtree(output_dir)
    print(f"Cleaned up test directory: {output_dir}")

    print("\n--- Visualizer test completed successfully! ---")