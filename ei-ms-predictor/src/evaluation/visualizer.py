import matplotlib.pyplot as plt
import numpy as np

class SpectrumVisualizer:
    """
    Provides methods to visualize mass spectra.
    """
    def __init__(self):
        pass

    def plot_spectrum(self, mz_values, intensities, title="Mass Spectrum", color='blue', label=None):
        """
        Helper function to plot a single mass spectrum as a stem plot.
        """
        # Create a stem plot
        markerline, stemlines, baseline = plt.stem(
            mz_values,
            intensities,
            linefmt=f'{color}-',
            markerfmt=f'{color}o',
            basefmt='gray'
        )
        # Style the plot
        plt.setp(stemlines, 'linewidth', 1.5)
        plt.setp(markerline, 'markersize', 3)
        plt.setp(baseline, 'linewidth', 0.5)
        if label:
            # Create a proxy artist for the legend
            plt.plot([], [], color=color, label=label)

    def plot_comparison(self, pred_mz, pred_intensity, true_mz, true_intensity, output_path=None):
        """
        Plots a comparison of a predicted spectrum and a true spectrum.

        Args:
            pred_mz, pred_intensity (np.array): The predicted spectrum.
            true_mz, true_intensity (np.array): The ground truth spectrum.
            output_path (str, optional): If provided, saves the plot to this file.
        """
        plt.figure(figsize=(12, 6))

        # Normalize intensities for better comparison
        if np.max(pred_intensity) > 0:
            pred_intensity = 100 * pred_intensity / np.max(pred_intensity)
        if np.max(true_intensity) > 0:
            true_intensity = 100 * true_intensity / np.max(true_intensity)

        # Plot true spectrum (inverted, in gray)
        self.plot_spectrum(true_mz, -true_intensity, color='gray', label='True Spectrum')

        # Plot predicted spectrum (in blue)
        self.plot_spectrum(pred_mz, pred_intensity, color='blue', label='Predicted Spectrum')

        # Final plot styling
        plt.title("Predicted vs. True Spectrum")
        plt.xlabel("m/z")
        plt.ylabel("Relative Intensity")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(0, color='black', linewidth=0.5) # Zero line
        plt.legend()

        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

# Example Usage:
if __name__ == '__main__':
    # Create some dummy data
    true_mz = np.array([29, 45, 31, 27])
    true_intensity = np.array([60, 100, 70, 40])

    pred_mz = np.array([29, 45, 31, 50])
    pred_intensity = np.array([55, 95, 80, 10])

    visualizer = SpectrumVisualizer()
    visualizer.plot_comparison(pred_mz, pred_intensity, true_mz, true_intensity, "spectrum_comparison_example.png")
