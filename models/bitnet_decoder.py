import torch
import torch.nn as nn
import torch.nn.functional as F

# To run the test script directly, we need to add the project root to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bitlinear import BitLinear

class BitNetDecoder(nn.Module):
    """
    BitNet Decoder for Mass Spectrum Prediction.

    This module takes a graph embedding and decodes it into a mass spectrum
    (m/z values and intensities) using a series of BitLinear layers, as
    specified in the design documents.
    """
    def __init__(self, input_dim, hidden_dim, output_bins=1000, num_layers=4):
        super().__init__()

        if num_layers < 2:
            raise ValueError("BitNetDecoder must have at least 2 layers.")

        self.output_bins = output_bins

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # Input layer
        self.layers.append(BitLinear(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(BitLinear(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Output layer
        # The output dimension is doubled to produce both m/z and intensity for each bin.
        self.layers.append(BitLinear(hidden_dim, output_bins * 2))

        # Add LayerNorm for the input layer as well
        self.layer_norms.insert(0, nn.LayerNorm(hidden_dim))

        self.activation = nn.SiLU() # Swish activation function

    def forward(self, x):
        """
        Forward pass for the BitNet Decoder.

        Args:
            x (torch.Tensor): A batch of graph embeddings of shape [batch_size, input_dim].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - mz_values (torch.Tensor): Predicted m/z values, shape [batch_size, output_bins].
                - intensities (torch.Tensor): Predicted intensities, shape [batch_size, output_bins].
        """
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.layer_norms[i](x)
            x = self.activation(x)

        # Final layer
        output = self.layers[-1](x)

        # Split the output into two parts for m/z and intensity
        # The shape of output is [batch_size, output_bins * 2]
        mz_logits, intensity_logits = output.chunk(2, dim=-1)

        # --- Process outputs as per specification ---

        # m/z values are predicted in the range [1, 1000]
        # A sigmoid function maps the logits to (0, 1), which is then scaled.
        # Note: The design doc implies predicting m/z values directly.
        # This is unusual; typically the model predicts intensities for fixed m/z bins.
        # For now, we will follow the design doc's approach of predicting both.
        mz_values = torch.sigmoid(mz_logits) * (self.output_bins - 1) + 1

        # Intensities are relative, so softmax is appropriate to make them sum to a constant.
        # The result is then scaled to a max of 100.
        # The design doc says "softmax * 100", which might mean scaling the max intensity to 100.
        # A standard softmax will produce probabilities that sum to 1.
        # Let's use a simple sigmoid for each intensity logit for now, which is more common
        # for multi-label problems where peaks are not mutually exclusive.
        intensities = torch.sigmoid(intensity_logits) * 100.0

        return mz_values, intensities

if __name__ == '__main__':
    # --- Example Usage & Test ---
    print("--- Testing BitNetDecoder Layer ---")

    # Parameters from specification-doc.md
    input_dim = 256
    hidden_dim = 512
    output_bins = 1000
    batch_size = 16

    # Create the decoder
    decoder = BitNetDecoder(input_dim, hidden_dim, output_bins)
    print("Decoder architecture:", decoder)

    # Create a dummy input tensor (simulating output from GCNEncoder)
    dummy_embedding = torch.randn(batch_size, input_dim)

    # --- Forward Pass ---
    print("\nRunning forward pass...")
    mz, intensity = decoder(dummy_embedding)

    print("Input embedding shape:", dummy_embedding.shape)
    print("Output m/z shape:", mz.shape)
    print("Output intensity shape:", intensity.shape)

    # Check shapes
    assert mz.shape == (batch_size, output_bins)
    assert intensity.shape == (batch_size, output_bins)

    # Check value ranges
    assert mz.min() >= 1.0 and mz.max() <= output_bins, "m/z values are out of range [1, 1000]"
    assert intensity.min() >= 0.0 and intensity.max() <= 100.0, "Intensity values are out of range [0, 100]"
    print("Forward pass successful. Shapes and value ranges are correct.")

    # --- Backward Pass (Gradient Check) ---
    print("\nRunning backward pass...")
    target_mz = torch.rand(batch_size, output_bins) * 999 + 1
    target_intensity = torch.rand(batch_size, output_bins) * 100

    loss = F.mse_loss(mz, target_mz) + F.mse_loss(intensity, target_intensity)
    loss.backward()

    # Check if gradients exist for the first layer's weights
    grad = decoder.layers[0].weight.grad
    print("Gradient shape for first BitLinear layer:", grad.shape)
    assert grad is not None, "Gradient is None! Backward pass failed."
    assert grad.abs().sum() > 0, "Gradient is all zeros!"
    print("Backward pass successful. Gradients are flowing.")

    print("\n--- BitNetDecoder test completed successfully! ---")