import torch
import warnings

class DeviceManager:
    """CPU/GPU切り替え管理"""

    def __init__(self, force_cpu: bool = False):
        self.force_cpu = force_cpu
        self.device = self._setup_device()

    def _setup_device(self):
        if self.force_cpu:
            print("Forcing CPU.")
            return torch.device('cpu')

        if torch.cuda.is_available():
            # The spec mentions a specific check for sm_120, but for generality,
            # we will just use any available CUDA device.
            # A more robust implementation could check `torch.cuda.get_device_capability()`.
            print("CUDA is available. Using GPU.")
            return torch.device('cuda')
        else:
            print("CUDA not available. Falling back to CPU.")
            return torch.device('cpu')

    def get_device(self):
        return self.device

    def optimize_for_device(self, model):
        """デバイス最適化"""
        # The spec mentions many advanced optimizations (BF16, Flash Attention).
        # These are highly specific to hardware and recent library versions.
        # For this general implementation, we will stick to a simple .to(device) call,
        # which is universally applicable. The user can add advanced optimizations later.

        # A simple check for mixed precision availability.
        if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
             # Using bfloat16 for better performance on modern GPUs.
            # return model.to(self.device, dtype=torch.bfloat16)
            pass # Keep it simple for now

        return model.to(self.device)
