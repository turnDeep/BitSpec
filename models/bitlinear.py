import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Module):
    """
    Custom BitLinear layer with 1.58-bit (ternary) weight quantization and 8-bit activation quantization.

    This implementation is based on the design and specification documents.
    - Weights are quantized to {-1, 0, 1}.
    - Activations are quantized to 8-bit integers.
    - A Straight-Through Estimator (STE) is used for the weight quantization to allow gradient flow during backpropagation.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision weights, stored for training and updated by the optimizer.
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # --- Weight Quantization (Ternary: -1, 0, 1) ---
        # This logic is derived from `specification-doc.md`.

        # 1. Calculate the threshold for ternary quantization.
        # .detach() is used to ensure this calculation does not affect the gradient computation.
        abs_mean = self.weight.abs().mean().detach()
        threshold = 0.7 * abs_mean

        # 2. Quantize weights to -1, 0, or 1.
        w_quant = torch.sign(self.weight)
        w_quant[self.weight.abs() < threshold] = 0

        # 3. Calculate the scaling factor for the quantized weights.
        # This compensates for the change in magnitude caused by quantization.
        # Adding a small epsilon to avoid division by zero if all weights are quantized to 0.
        non_zero_mean = (w_quant != 0).float().mean() + 1e-8
        scale_w = self.weight.abs().mean().detach() / non_zero_mean

        # 4. Apply the scaling factor to the quantized weights.
        w_scaled = w_quant * scale_w

        # 5. Apply Straight-Through Estimator (STE).
        # In the forward pass, we use the quantized and scaled weights (w_scaled).
        # In the backward pass, the gradient is passed through to the full-precision weights (self.weight).
        # This is a standard and essential technique for training quantized networks.
        w_ste = (w_scaled - self.weight).detach() + self.weight

        # --- Activation Quantization (8-bit) ---
        # This logic is derived from `design-doc.md`.

        # 1. Calculate the scaling factor for activations.
        # We use symmetric quantization for the activations.
        # Adding epsilon to avoid division by zero.
        x_abs_max = x.abs().max().detach()
        if x_abs_max == 0:
            # If input is all zeros, the output will also be all zeros.
            if self.bias is not None:
                return self.bias.clone()
            else:
                return torch.zeros(x.shape[0], self.out_features, device=x.device, dtype=x.dtype)

        q_max = 127.0
        scale_x = q_max / x_abs_max

        # 2. Quantize and de-quantize the activations.
        x_quant = torch.round(x * scale_x)
        x_dequant = x_quant / scale_x

        # 3. Apply STE for activation quantization.
        # This is crucial for gradients to flow back to previous layers.
        # We use the quantized value in the forward pass, but treat the operation
        # as an identity function during the backward pass for gradient purposes.
        x_ste = (x_dequant - x).detach() + x

        # --- Linear Operation ---
        # Perform the linear operation using the STE-applied activations and weights.
        output = F.linear(x_ste, w_ste, self.bias)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

if __name__ == '__main__':
    # --- Example Usage & Test ---
    print("--- Testing BitLinear Layer ---")

    # Parameters
    in_features = 128
    out_features = 256
    batch_size = 16

    # Create layer and input tensor
    bit_linear_layer = BitLinear(in_features, out_features)
    input_tensor = torch.randn(batch_size, in_features)

    print("Layer:", bit_linear_layer)

    # --- Forward Pass ---
    print("\nRunning forward pass...")
    output = bit_linear_layer(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    assert output.shape == (batch_size, out_features)
    print("Forward pass successful.")

    # --- Backward Pass (Gradient Check) ---
    print("\nRunning backward pass...")
    # We need to ensure gradients are flowing to the full-precision `weight` parameter.
    target = torch.randn(batch_size, out_features)
    loss = (output - target).pow(2).mean()
    loss.backward()

    grad = bit_linear_layer.weight.grad
    print("Gradient shape:", grad.shape)
    assert grad is not None, "Gradient is None! STE might be implemented incorrectly."
    assert grad.shape == (out_features, in_features)
    assert grad.abs().sum() > 0, "Gradient is all zeros! Check STE."
    print("Backward pass successful. Gradients are flowing.")

    print("\n--- BitLinear Layer test completed successfully! ---")